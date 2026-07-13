"""
Iterative Schema Refinement Pipeline using SGLang

Key design decisions:
- Two-stage evaluation: separate extraction from critique
- Reduced score dimensions (8 → 5) to reduce correlation/noise
- Reduced feedback categories (6 → 4) to reduce overlap
- Schema constraints enforced in code, not in prompts
- Refinement encourages generalization (prevents schema bloat)
- Chain-of-thought reasoning in refinement step
- Captures model reasoning_content (when reasoning is activated)

PREREQUISITE:
    Start the SGLang server BEFORE running this script:
        python -m sglang.launch_server \
            --model-path meta-llama/Llama-3.1-8B-Instruct \
            --port 30000

OUTPUT:
    results/iterative_schema/<model_name>/
        ├── final_schema.json
        ├── schema_evolution.json
        ├── batch_aggregations.csv
        ├── evaluations.csv
        ├── filled_test.csv
        ├── reasoning_traces.json   ← NEW: all captured reasoning
        └── experiment_metrics.csv
"""

import json
import re
import statistics
import time
import psutil
import os
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from jsondiff import diff as jsondiff_compare
from sglang_client import SGLangClient


# ============================================================
# Data Classes
# ============================================================

@dataclass
class EvaluationResult:
    """Results from evaluating a single clinical note against a schema."""
    evalid: str
    coverage: float           # How well does schema capture the note?
    accuracy: float           # Does extraction faithfully represent the note?
    clarity: float            # Is the structure logical and clear?
    conciseness: float        # Is it free of redundancy?
    utility: float            # Is it clinically useful?
    feedback: Dict[str, Any]
    filled_instance: Dict[str, Any]
    extraction_reasoning: Optional[str] = None   # NEW
    evaluation_reasoning: Optional[str] = None   # NEW
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class BatchAggregation:
    avg_scores: Dict[str, float]
    feedback_summary: Dict[str, Any]
    sample_count: int
    batch_id: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class SchemaEvolution:
    version: int
    schema: Dict[str, Any]
    changes_made: List[str]
    schema_diff: Optional[Dict[str, Any]] = None
    batch_id: Optional[str] = None
    rationale: str = ""
    refinement_reasoning: Optional[str] = None   # NEW
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class FillResult:
    """NEW: Track fill outputs with reasoning."""
    note_index: int
    filled_schema: str
    fill_reasoning: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ExperimentMetrics:
    model_name: str
    method: str = "iterative_schema"
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    total_duration: Optional[float] = None
    peak_memory_mb: float = 0.0
    api_calls: int = 0
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_calls: int = 0   # NEW: count of calls that returned reasoning

    def update_memory(self):
        process = psutil.Process(os.getpid())
        current_mb = process.memory_info().rss / 1024 / 1024
        self.peak_memory_mb = max(self.peak_memory_mb, current_mb)

    def update_from_response(self, response: Dict[str, Any]):
        self.api_calls += 1
        usage = response.get("usage", {})
        self.total_tokens += usage.get("total_tokens", 0)
        self.prompt_tokens += usage.get("prompt_tokens", 0)
        self.completion_tokens += usage.get("completion_tokens", 0)
        if response.get("reasoning"):
            self.reasoning_calls += 1
        self.update_memory()

    def finalize(self):
        self.end_time = time.time()
        self.total_duration = self.end_time - self.start_time



# ============================================================
# Constants & Helpers
# ============================================================

def parse_llm_json(response_text: str) -> Dict[str, Any]:
    """Extract and parse JSON from LLM response text."""
    text = re.sub(r'```json\s*', '', response_text)
    text = re.sub(r'```\s*', '', text)
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if not json_match:
        raise ValueError("No JSON found in LLM response")
    try:
        return json.loads(json_match.group())
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in LLM response: {e}")

def _coerce_score(value: Any, default: float = 0.0) -> float:
    """Coerce LLM-produced score to float. LLMs sometimes return '4', '4/5', 'N/A', etc."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        # Pull the first number out of strings like "4", "4/5", "4 - good"
        m = re.search(r'-?\d+(?:\.\d+)?', value)
        if m:
            try:
                return float(m.group())
            except ValueError:
                return default
    return default


# Reduced from 6 categories to 4 (less overlap, more stable aggregation)
FEEDBACK_CATEGORIES = [
    "missing_information",     # Things in the note not captured by schema
    "representation_issues",   # Wrong types, awkward field structure
    "redundancy",              # Duplicate or unnecessary fields
    "improvements",            # Concrete suggestions
]

# Reduced from 8 scores to 5 (less correlation, statistically cleaner)
SCORE_NAMES = [
    "coverage",      # Was completeness + thoroughness
    "accuracy",      # Was medical_accuracy
    "clarity",       # Was schema_clarity + consistency
    "conciseness",   # Kept as is
    "utility",       # Was usefulness + flexibility
]

# Brief design philosophy (replaces long technical SCHEMA_DESIGN_PRINCIPLES)
SCHEMA_PHILOSOPHY = """
Schema design principles:
- Use simple, clinically meaningful field names
- Use arrays only when repetition naturally occurs
- Keep field types simple (string, number, integer, boolean, object, array)
- Add descriptions to clarify clinical purpose
"""


def enforce_schema_constraints(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Programmatically enforce schema constraints.

    Done in code, not prompts, because:
    1. Teaching LLM these rules is unreliable
    2. Frees prompt space for clinical reasoning
    3. Deterministic and reproducible

    Constraints:
    - No 'required' arrays (all fields optional → reduces hallucination)
    - additionalProperties: false at root only (prevents new top-level fields)
    - Type arrays like ["string", "null"] normalized to "string"
    """
    def clean(obj, is_root=False):
        if isinstance(obj, dict):
            obj.pop('required', None)

            if 'type' in obj:
                obj_type = obj['type']
                if isinstance(obj_type, list):
                    non_null = [t for t in obj_type if t != 'null']
                    obj['type'] = non_null[0] if non_null else 'string'

            if is_root and 'properties' in obj:
                obj['additionalProperties'] = False
            else:
                obj.pop('additionalProperties', None)

            for value in obj.values():
                if isinstance(value, dict):
                    clean(value, is_root=False)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            clean(item, is_root=False)
        return obj

    schema = clean(schema, is_root=True)
    if 'type' not in schema:
        schema['type'] = 'object'
    return schema


# ============================================================
# Evaluator (Two-Stage: Extract → Evaluate)
# ============================================================

class ClinicalNoteEvaluator:
    """
    Two-stage evaluator:
    1. Extract data using the schema (no memory - independent)
    2. Evaluate the schema based on extraction (uses memory within batch)

    Separation reduces cognitive load and produces better feedback.
    """

    def __init__(self, client: SGLangClient, metrics: ExperimentMetrics):
        self.client = client
        self.metrics = metrics
        self.batch_memory: List[EvaluationResult] = []

        self.extraction_system = (
            "You are a clinical data extraction specialist. "
            "Extract structured data faithfully from clinical notes. "
            "Always output valid JSON."
        )
        self.evaluation_system = (
            "You are a clinical informatics expert evaluating JSON schemas. "
            "You provide concise, actionable critique. "
            "Always output valid JSON."
        )

    def _extract(self, clinical_note: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Stage 1: Extract data from note using schema. Returns (parsed, reasoning)."""
        prompt = f"""Extract structured data from the clinical note using the provided schema.

Rules:
- Only extract information explicitly stated in the note
- Omit fields that have no corresponding information in the note
- Do not infer, guess, or add fields not in the schema
- Match the data types specified in the schema

SCHEMA:
{json.dumps(schema, indent=2)}

CLINICAL NOTE:
{clinical_note}

Output the extracted JSON object. No explanation, no markdown."""

        response = self.client.generate(
            prompt,
            system_prompt=self.extraction_system,
            use_memory=False,
            response_format={"type": "json_object"}
        )
        self.metrics.update_from_response(response)
        return parse_llm_json(response["reply"]), response.get("reasoning")

    def _evaluate(self, clinical_note: str, schema: Dict[str, Any],
                  extracted: Dict[str, Any]) -> Dict[str, Any]:
        """Stage 2: Evaluate the schema given the extraction. Returns (parsed, reasoning)."""
        prompt = f"""Evaluate this JSON schema based on how it captured information from the clinical note.

SCHEMA:
{json.dumps(schema, indent=2)}

CLINICAL NOTE:
{clinical_note}

EXTRACTED DATA (using current schema):
{json.dumps(extracted, indent=2)}

Compare the note against the extracted data. Score the schema (0-5) on:
- coverage: How well the schema captures relevant clinical information
- accuracy: How faithfully the extraction represents the note
- clarity: How logical and easy to interpret the schema is
- conciseness: How free from redundancy or unnecessary fields
- utility: How clinically practical and useful the schema is

Identify problems by category:
- missing_information: Important things in the note not captured
- representation_issues: Wrong types, awkward structure, fields that don't fit data
- redundancy: Duplicate, overlapping, or unnecessary fields
- improvements: Concrete, actionable suggestions

Output JSON:
{{
    "scores": {{"coverage": 0-5, "accuracy": 0-5, "clarity": 0-5, "conciseness": 0-5, "utility": 0-5}},
    "feedback": {{
        "missing_information": ["specific items not captured"],
        "representation_issues": ["specific problems with how data is represented"],
        "redundancy": ["specific redundancies"],
        "improvements": ["specific actionable suggestions"]
    }}
}}"""

        response = self.client.generate(
            prompt,
            system_prompt=self.evaluation_system,
            use_memory=True,  # Memory within batch for context
            response_format={"type": "json_object"}
        )
        self.metrics.update_from_response(response)
        return parse_llm_json(response["reply"]), response.get("reasoning")


    def evaluate_note(self, clinical_note: str, note_id: str,
                     schema: Dict[str, Any]) -> EvaluationResult:
        """Two-stage evaluation: extract first, then evaluate."""
        try:
            
            extracted, extraction_reasoning = self._extract(clinical_note, schema)
            eval_data, evaluation_reasoning = self._evaluate(clinical_note, schema, extracted)

            raw_scores = eval_data.get("scores") or {}
            if not isinstance(raw_scores, dict):
                raw_scores = {}
            scores = {s: _coerce_score(raw_scores.get(s, 0)) for s in SCORE_NAMES}

            raw_feedback = eval_data.get("feedback") or {}
            if not isinstance(raw_feedback, dict):
                raw_feedback = {"error": f"feedback not a dict: {type(raw_feedback).__name__}"}

            evaluation = EvaluationResult(
                evalid=note_id,
                **scores,
                feedback=raw_feedback,
                filled_instance=extracted if isinstance(extracted, dict) else {},
                extraction_reasoning=extraction_reasoning,
                evaluation_reasoning=evaluation_reasoning,
            )
        except Exception as e:
            print(f"  Evaluation error for note {note_id}: {e}")
            evaluation = EvaluationResult(
                evalid=note_id,
                **{s: 0 for s in SCORE_NAMES},
                feedback={"error": str(e)},
                filled_instance={},
                extraction_reasoning=None,
                evaluation_reasoning=None,
            )

        self.batch_memory.append(evaluation)
        return evaluation

    def aggregate_batch(self, batch_id: str) -> BatchAggregation:
        """Aggregate evaluation results using mean (more sensitive than median)."""
        if not self.batch_memory:
            return BatchAggregation({}, {}, 0, batch_id)

        avg_scores = {
            s: statistics.mean([getattr(r, s) for r in self.batch_memory])
            for s in SCORE_NAMES
        }

        feedback_summary = {}
        for cat in FEEDBACK_CATEGORIES:
            items = []
            for r in self.batch_memory:
                items.extend(r.feedback.get(cat, []))
            feedback_summary[cat] = list(dict.fromkeys(items))

        return BatchAggregation(
            avg_scores=avg_scores,
            feedback_summary=feedback_summary,
            sample_count=len(self.batch_memory),
            batch_id=batch_id,
        )

    def clear_batch_memory(self):
        """Clear batch memory and conversation history between batches."""
        self.batch_memory.clear()
        self.client.reset_conversation()


# ============================================================
# Refiner (Generalization-Oriented with CoT)
# ============================================================

class SchemaRefiner:
    """
    Refines schemas based on aggregated feedback.
    Uses chain-of-thought reasoning and encourages generalization
    over schema expansion to prevent bloat across iterations.
    """

    def __init__(self, client: SGLangClient, metrics: ExperimentMetrics):
        self.client = client
        self.metrics = metrics
        self.system_prompt = (
            "You are a clinical informatics expert refining JSON schemas. "
            "You favor generalization, simplicity, and clinical practicality. "
            "Always output valid JSON."
        )

    def refine_schema(self, current_schema: Dict[str, Any],
                     aggregation: BatchAggregation) -> SchemaEvolution:
        """Refine a schema based on aggregated batch feedback."""

        prompt = f"""You are refining a clinical JSON schema based on evaluation feedback from {aggregation.sample_count} clinical notes.

CURRENT SCHEMA:
{json.dumps(current_schema, indent=2)}

AVERAGE SCORES (0-5):
{json.dumps(aggregation.avg_scores, indent=2)}

FEEDBACK FROM EVALUATION:
Missing information:    {aggregation.feedback_summary.get('missing_information', [])}
Representation issues:  {aggregation.feedback_summary.get('representation_issues', [])}
Redundancy:             {aggregation.feedback_summary.get('redundancy', [])}
Improvements:           {aggregation.feedback_summary.get('improvements', [])}

{SCHEMA_PHILOSOPHY}

Refinement priorities:
1. Address frequently-mentioned issues (appearing multiple times in feedback)
2. Prefer GENERAL fields over highly specific ones
3. MERGE semantically similar fields when possible
4. REMOVE fields that appear only rarely or are redundant
5. Only ADD fields that capture commonly missing information
6. Keep changes conservative - don't restructure unnecessarily

Think step by step:
1. What are the 2-3 most important issues to address?
2. For each, decide: add, remove, rename, merge, or restructure?
3. Will the change generalize to other clinical notes, or only fix one case?

Output JSON:
{{
    "reasoning": "brief analysis of feedback and decisions",
    "refined_schema": {{...the improved schema...}},
    "changes_made": ["specific change 1", "specific change 2", ...],
    "rationale": "why these changes improve the schema"
}}"""

        try:
            response = self.client.generate(
                prompt,
                system_prompt=self.system_prompt,
                use_memory=False,
                response_format={"type": "json_object"}
            )

            self.metrics.update_from_response(response)
            result_data = parse_llm_json(response["reply"])
            refined_schema = result_data["refined_schema"]

            # Code-level enforcement (not in prompts)
            refined_schema = enforce_schema_constraints(refined_schema)

            return SchemaEvolution(
                version=0,
                schema=refined_schema,
                changes_made=result_data["changes_made"],
                schema_diff=jsondiff_compare(current_schema, refined_schema),
                batch_id=aggregation.batch_id,
                rationale=f"{result_data.get('rationale', '')} | Reasoning: {result_data.get('reasoning', '')}",
                refinement_reasoning=response.get("reasoning"),
            )
        except Exception as e:
            print(f"  Refinement error: {e}")
            return SchemaEvolution(
                version=0,
                schema=deepcopy(current_schema),
                changes_made=[f"Refinement failed: {str(e)}"],
                schema_diff={},
                batch_id=aggregation.batch_id,
                rationale="Refinement failed",
                refinement_reasoning=None,
            )


# ============================================================
# Schema Filler (Independent, No Memory)
# ============================================================

class SchemaFiller:
    """Fills the final schema from clinical notes."""

    def __init__(self, client: SGLangClient, metrics: ExperimentMetrics):
        self.client = client
        self.metrics = metrics
        self.system_prompt = (
            "You are a clinical data extraction specialist. "
            "Extract structured data faithfully from clinical notes. "
            "Always output valid JSON."
        )

    def fill(self, clinical_note: str, schema: Dict[str, Any],
             example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract data from a clinical note using the final schema.
        Returns dict with 'reply' and 'reasoning'.
        """

        example_section = ""
        if example and len(json.dumps(example)) > 50:
            example_section = f"\nReference example output:\n{json.dumps(example, indent=2)}\n"

        prompt = f"""Extract structured data from the clinical note using the schema.

Rules:
- Only extract information explicitly stated in the note
- Omit fields that have no corresponding information in the note
- Do not infer, guess, or invent values
- Match the data types specified in the schema
{example_section}
SCHEMA:
{json.dumps(schema, indent=2)}

CLINICAL NOTE:
{clinical_note}

Output the extracted JSON object. No explanation, no markdown."""

        try:
            response = self.client.generate(
                prompt,
                system_prompt=self.system_prompt,
                use_memory=False,
                response_format={"type": "json_object"}
            )
            self.metrics.update_from_response(response)
            return {
                "reply": response["reply"],
                "reasoning": response.get("reasoning"),
            }
        except Exception as e:
            print(f"  Error filling schema: {e}")
            return {
                "reply": json.dumps({"error": str(e)}),
                "reasoning": None,
            }


# ============================================================
# Pipeline Coordinator
# ============================================================

class IterativeSchemaRefinementPipeline:
    """Orchestrates the iterative schema refinement pipeline."""

    def __init__(self, model_name: str, output_dir: str,
                 base_url: str = "http://127.0.0.1:30000/v1"):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.metrics = ExperimentMetrics(model_name=model_name)

        # Three separate clients - separates conversation histories per role
        self.eval_client = SGLangClient(model=model_name, base_url=base_url, temperature=0.1)
        self.refine_client = SGLangClient(model=model_name, base_url=base_url, temperature=0.1)
        self.fill_client = SGLangClient(model=model_name, base_url=base_url, temperature=0.2)

        self.evaluator = ClinicalNoteEvaluator(self.eval_client, self.metrics)
        self.refiner = SchemaRefiner(self.refine_client, self.metrics)
        self.filler = SchemaFiller(self.fill_client, self.metrics)

        self.schema_evolution: List[SchemaEvolution] = []
        self.batch_results: List[BatchAggregation] = []
        self.all_evaluations: List[EvaluationResult] = []
        self.fill_results: List[FillResult] = []   # NEW
        self.initial_schema_reasoning: Optional[str] = None   # NEW

    def get_initial_schema(self) -> Dict[str, Any]:
        """Generate a minimal starting schema."""
        prompt = f"""Generate a JSON schema for clinical notes covering common sections:
- Patient demographics
- Chief complaint
- History of present illness
- Vital signs
- Physical examination
- Assessment / diagnosis
- Treatment plan
- Medications

{SCHEMA_PHILOSOPHY}

Output a complete JSON schema. Use simple types and brief descriptions."""

        system_prompt = (
            "You are a clinical informatics expert designing JSON schemas. "
            "Always output valid JSON."
        )

        response = self.eval_client.generate(
            prompt,
            system_prompt=system_prompt,
            use_memory=False,
            response_format={"type": "json_object"}
        )
        self.metrics.update_from_response(response)
        self.initial_schema_reasoning = response.get("reasoning")

        try:
            schema = json.loads(response['reply'])
        except json.JSONDecodeError:
            schema = parse_llm_json(response['reply'])

        return enforce_schema_constraints(schema)

    def run_refinement(self, train_data: pd.DataFrame,
                      num_clusters: Optional[int] = None) -> Dict[str, Any]:
        """Run iterative schema refinement over cluster batches."""

        current_schema = self.get_initial_schema()
        schema_version = 0

        self.schema_evolution.append(SchemaEvolution(
            version=schema_version,
            schema=deepcopy(current_schema),
            changes_made=["Initial schema"],
            schema_diff={},
            rationale="Starting schema generated from clinical informatics knowledge",
            refinement_reasoning=self.initial_schema_reasoning,
        ))

        clusters = sorted(train_data['cluster'].unique())
        if num_clusters:
            clusters = clusters[0:num_clusters+1]
        else:
            clusters = clusters[0:]

        print(f"\nProcessing {len(clusters)} clusters: {clusters}")

        for cluster_id in clusters:
            batch_notes = train_data[train_data['cluster'] == cluster_id]['clinical_notes'].dropna().tolist()
            batch_id = f"cluster_{cluster_id}"
            print(f"\n=== Processing {batch_id} ({len(batch_notes)} notes) ===")

            self.evaluator.clear_batch_memory()

            for note_idx, note in enumerate(batch_notes):
                print(f"  Evaluating note {note_idx + 1}/{len(batch_notes)}")
                note_id = f"{cluster_id}_{note_idx}"
                result = self.evaluator.evaluate_note(note, note_id=note_id, schema=current_schema)
                self.all_evaluations.append(result)

            aggregation = self.evaluator.aggregate_batch(batch_id)
            self.batch_results.append(aggregation)

            print(f"  Refining schema based on {aggregation.sample_count} evaluations...")
            evolution = self.refiner.refine_schema(current_schema, aggregation)

            schema_version += 1
            evolution.version = schema_version
            self.schema_evolution.append(evolution)
            current_schema = evolution.schema

            print(f"  Schema v{schema_version} created with {len(evolution.changes_made)} changes")

        self._save_refinement_results(current_schema)
        print(f"\nRefinement complete. Final schema is v{schema_version}")
        return current_schema

    def run_filling(self, test_data: pd.DataFrame,
                   final_schema: Dict[str, Any]) -> pd.DataFrame:
        """Fill the final schema using test data."""
        # Find a good example from training extractions
        example = {}
        if self.all_evaluations:
            for e in reversed(self.all_evaluations):
                if e.filled_instance and e.coverage >= 3:
                    example = e.filled_instance
                    break

        filled_results: List[str] = []
        fill_reasonings: List[Optional[str]] = []
        print(f"\n=== Filling schema for {len(test_data)} test notes ===")

        for idx, note in enumerate(test_data['clinical_notes'].values):
            print(f"  Filling note {idx + 1}/{len(test_data)}")
            fill_output = self.filler.fill(note, schema=final_schema, example=example)
            filled_results.append(fill_output["reply"])
            fill_reasonings.append(fill_output["reasoning"])
            self.fill_results.append(FillResult(
                note_index=idx,
                filled_schema=fill_output["reply"],
                fill_reasoning=fill_output["reasoning"],
            ))

        test_data = test_data.copy()
        test_data['filled_schema'] = filled_results
        test_data['fill_reasoning'] = fill_reasonings
        test_data.to_csv(self.output_dir / 'filled_test.csv', index=False)
        print(f"  Saved → {self.output_dir / 'filled_test.csv'}")

        # Save reasoning traces (consolidated)
        self._save_reasoning_traces()
        return test_data

    def _save_refinement_results(self, final_schema: Dict[str, Any]):
        """Save all refinement artifacts."""
        # Final schema
        with open(self.output_dir / 'final_schema.json', 'w') as f:
            json.dump(final_schema, f, indent=2)

        # Schema evolution (includes refinement_reasoning per version)
        evolution_df = pd.DataFrame([asdict(e) for e in self.schema_evolution])
        evolution_df.to_json(self.output_dir / 'schema_evolution.json',
                            orient='records', indent=2)

        # Batch aggregations - guard against empty
        if self.batch_results:
            batch_df = pd.DataFrame([asdict(e) for e in self.batch_results])
            batch_df = pd.concat([
                batch_df.drop(columns=['avg_scores', 'feedback_summary']),
                pd.json_normalize(batch_df['avg_scores']),
                pd.json_normalize(batch_df['feedback_summary']),
            ], axis=1)
        else:
            batch_df = pd.DataFrame()
        batch_df.to_csv(self.output_dir / 'batch_aggregations.csv', index=False)

        # Evaluations - guard against empty (now includes reasoning columns)
        if self.all_evaluations:
            evaluations_df = pd.DataFrame([asdict(e) for e in self.all_evaluations])
            evaluations_df = pd.concat([
                evaluations_df.drop(columns='feedback'),
                pd.json_normalize(evaluations_df['feedback']),
            ], axis=1)
        else:
            evaluations_df = pd.DataFrame()
        evaluations_df.to_csv(self.output_dir / 'evaluations.csv', index=False)

        print(f"\nArtifacts saved to {self.output_dir}/")

    def _save_reasoning_traces(self):
        """
        Consolidate all reasoning traces into one JSON file.
        Entries with no reasoning content (None / empty) are filtered out, so
        non-reasoning models produce a small file rather than a wall of nulls.
        If no reasoning was captured anywhere, no file is written.
        """
        refinements = [
            {
                "version": e.version,
                "batch_id": e.batch_id,
                "reasoning": e.refinement_reasoning,
                "changes_made": e.changes_made,
            }
            for e in self.schema_evolution
            if e.refinement_reasoning
        ]

        evaluations = []
        for e in self.all_evaluations:
            entry = {"evalid": e.evalid}
            if e.extraction_reasoning:
                entry["extraction_reasoning"] = e.extraction_reasoning
            if e.evaluation_reasoning:
                entry["evaluation_reasoning"] = e.evaluation_reasoning
            # Only keep the entry if at least one stage produced reasoning
            if len(entry) > 1:
                evaluations.append(entry)

        fills = [
            {
                "note_index": f.note_index,
                "fill_reasoning": f.fill_reasoning,
            }
            for f in self.fill_results
            if f.fill_reasoning
        ]

        has_any_reasoning = bool(
            self.initial_schema_reasoning or refinements or evaluations or fills
        )

        if not has_any_reasoning:
            print(
                "  No reasoning content captured (model likely doesn't emit "
                "reasoning_content); skipping reasoning_traces.json"
            )
            return

        traces = {
            "model": self.model_name,
            "reasoning_calls": self.metrics.reasoning_calls,
            "total_api_calls": self.metrics.api_calls,
        }
        if self.initial_schema_reasoning:
            traces["initial_schema_reasoning"] = self.initial_schema_reasoning
        if refinements:
            traces["refinements"] = refinements
        if evaluations:
            traces["evaluations"] = evaluations
        if fills:
            traces["fills"] = fills

        with open(self.output_dir / 'reasoning_traces.json', 'w') as f:
            json.dump(traces, f, indent=2)
        print(f"  Saved → {self.output_dir / 'reasoning_traces.json'}")

    def save_experiment_metrics(self):
        """Save experiment computational metrics."""
        self.metrics.finalize()

        metrics_dict = {
            'model': self.metrics.model_name,
            'method': self.metrics.method,
            'total_duration_seconds': self.metrics.total_duration,
            'total_duration_minutes': self.metrics.total_duration / 60,
            'total_duration_hours': self.metrics.total_duration / 3600,
            'peak_memory_mb': self.metrics.peak_memory_mb,
            'peak_memory_gb': self.metrics.peak_memory_mb / 1024,
            'total_api_calls': self.metrics.api_calls,
            'reasoning_calls': self.metrics.reasoning_calls,
            'reasoning_call_ratio': self.metrics.reasoning_calls / max(self.metrics.api_calls, 1),
            'total_tokens': self.metrics.total_tokens,
            'prompt_tokens': self.metrics.prompt_tokens,
            'completion_tokens': self.metrics.completion_tokens,
            'avg_tokens_per_call': self.metrics.total_tokens / max(self.metrics.api_calls, 1),
        }

        pd.DataFrame([metrics_dict]).to_csv(self.output_dir / 'experiment_metrics.csv', index=False)

        print(f"\n{'='*60}")
        print("EXPERIMENT METRICS")
        print(f"{'='*60}")
        print(f"Model: {self.metrics.model_name}")
        print(f"Duration: {self.metrics.total_duration/60:.2f} min")
        print(f"Peak Memory: {self.metrics.peak_memory_mb/1024:.2f} GB")
        print(f"API Calls: {self.metrics.api_calls}")
        print(f"Reasoning-bearing calls: {self.metrics.reasoning_calls} "
              f"({100*self.metrics.reasoning_calls/max(self.metrics.api_calls,1):.1f}%)")
        print(f"Total Tokens: {self.metrics.total_tokens:,}")
    
    


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    # ---------- Configuration ----------

    # Model name as known to the running SGLang server.
    # If launched with --model-path meta-llama/Llama-3.1-8B-Instruct,
    # use the last path segment here.
    MODEL_NAME = os.environ.get("MODEL_NAME", "Llama-4-Scout-17B-16E-Instruct-FP8")

    # SGLang server (must already be running)
    SGLANG_BASE_URL = "http://127.0.0.1:30000/v1"

    # Output: results/iterative_schema/<model_name>/
    OUTPUT_DIR = f"./results/iterative_schema/{MODEL_NAME}"

    NUM_CLUSTERS = None

    TRAIN_DATA_PATH = os.environ.get("TRAIN_DATA_PATH", "datasets/cluster_training_data.csv")
    TEST_DATA_PATH = os.environ.get("TEST_DATA_PATH", "datasets/cluster_testing_data.csv")

    # ---------- Run ----------

    print("="*80)
    print(f"ITERATIVE SCHEMA REFINEMENT")
    print(f"Model:      {MODEL_NAME}")
    print(f"Server:     {SGLANG_BASE_URL}")
    print(f"Output:     {OUTPUT_DIR}")
    print(f"Note: SGLang server must already be running.")
    print("="*80)

    try:
        train_data = pd.read_csv(TRAIN_DATA_PATH)
        test_data = pd.read_csv(TEST_DATA_PATH, index_col=0)

        print(f"\nLoaded {len(train_data)} training notes, {len(test_data)} test notes")

        pipeline = IterativeSchemaRefinementPipeline(
            model_name=MODEL_NAME,
            output_dir=OUTPUT_DIR,
            base_url=SGLANG_BASE_URL,
        )

        final_schema = pipeline.run_refinement(train_data, num_clusters=NUM_CLUSTERS)
        pipeline.run_filling(test_data, final_schema)
        pipeline.save_experiment_metrics()

        print("\n✓ Experiment completed successfully!")
        print(f"\nNext steps:")
        print(f"  - Analyze: python analyze_experiment.py {OUTPUT_DIR}")
        print(f"  - Evaluate: python run_evaluation.py")

    except Exception as e:
        print(f"\n✗ Experiment failed: {e}")
        import traceback
        traceback.print_exc()