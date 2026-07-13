"""
Baseline: One-Shot In-Context Learning (ICL) Schema Generation

Single-pass schema generation: analyze patterns in training notes once,
generate a schema once, fill test notes against that schema.
No iteration, no feedback loop.

PREREQUISITE:
    Start the SGLang server BEFORE running this script:
        python -m sglang.launch_server \
            --model-path meta-llama/Llama-3.1-8B-Instruct \
            --port 30000

    NOTE: This baseline concatenates many training notes for pattern analysis,
    so launch the server with a LARGER context length (e.g., 32768).

OUTPUT:
    results/oneshot_icl_schema/<model_name>/
        ├── pattern_analysis.txt        Stage 1 output (free text)
        ├── final_schema.json           Stage 2 output (the schema)
        ├── filled_test.csv             Stage 3 output (test notes + filled JSON)
        └── experiment_metrics.csv      Runtime, memory, tokens
"""

import json
import re
import time
import psutil
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd
from sglang_client import SGLangClient


# ============================================================
# Metrics Tracking
# ============================================================

@dataclass
class ExperimentMetrics:
    model_name: str
    method: str = "oneshot_icl_schema"
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    total_duration: Optional[float] = None
    peak_memory_mb: float = 0.0
    api_calls: int = 0
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    
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
        self.update_memory()
    
    def finalize(self):
        self.end_time = time.time()
        self.total_duration = self.end_time - self.start_time


# ============================================================
# Helpers
# ============================================================

def parse_llm_json(response_text: str) -> Dict[str, Any]:
    """Extract and parse JSON from LLM response."""
    text = re.sub(r'```json\s*', '', response_text)
    text = re.sub(r'```\s*', '', text)
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if not json_match:
        raise ValueError("No JSON found in LLM response")
    return json.loads(json_match.group())


def enforce_schema_constraints(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Programmatically enforce schema constraints.
    Same as iterative pipeline - ensures fair comparison.
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
# One-Shot Schema Generator
# ============================================================

class OneShotSchemaGenerator:
    """
    Three-stage approach (single pass each, no feedback loop):
    1. analyze_patterns(notes)  - free-text pattern analysis (uses memory for stage 2)
    2. generate_schema()        - schema based on analysis (uses memory)
    3. fill_schema(note)        - per-note extraction (no memory between calls)
    """
    
    def __init__(self, model_name: str, output_dir: str, 
                 base_url: str = "http://0.0.0.0:30000/v1"):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics = ExperimentMetrics(model_name=model_name)
        
        # Two clients: analysis+generation (memory needed),
        # filling (no memory needed - each fill independent)
        self.client = SGLangClient(model=model_name, base_url=base_url, temperature=0.1)
        self.filler_client = SGLangClient(model=model_name, base_url=base_url, temperature=0.1)
        
        self.system_prompt = (
            "You are a clinical informatics expert specializing in analyzing "
            "clinical notes and designing JSON schemas. Always output valid JSON when requested."
        )
        
        self.pattern_analysis: Optional[str] = None
        self.schema: Optional[Dict[str, Any]] = None
    
    def analyze_patterns(self, notes: pd.Series, max_notes: int = 50) -> str:
        """
        Stage 1: Analyze patterns from training notes.
        
        Args:
            notes: Series of clinical notes
            max_notes: Cap to avoid context overflow.
        """
        notes_data = '\n---\n'.join(notes.dropna().values[:max_notes])
        
        prompt = f"""Analyze these clinical notes to identify common patterns for schema design.

Identify:
1. Common sections (chief complaint, HPI, vitals, assessment, plan, etc.)
2. Section ordering and structure
3. Key data types (dates, vitals, medications, diagnoses)
4. Recurring fields and their formats
5. Variations and edge cases

Provide a structured analysis with specific examples from the notes.

CLINICAL NOTES:
{notes_data}"""
        
        response = self.client.generate(
            prompt,
            temperature=0.2,
            system_prompt=self.system_prompt,
            use_memory=True,  # Stage 2 will reference this
        )
        
        self.metrics.update_from_response(response)
        self.pattern_analysis = response['reply']
        
        with open(self.output_dir / 'pattern_analysis.txt', 'w') as f:
            f.write(self.pattern_analysis)
        
        return self.pattern_analysis
    
    def generate_schema(self) -> Dict[str, Any]:
        """Stage 2: Generate schema from previously-stored analysis."""
        prompt = """Based on your previous analysis, create a JSON schema for clinical notes.

Schema design principles:
- Use simple, clinically meaningful field names
- Use arrays only when repetition naturally occurs
- Keep field types simple (string, number, integer, boolean, object, array)
- Add brief descriptions to clarify clinical purpose

Output a complete JSON schema only. No explanation."""
        
        response = self.client.generate(
            prompt,
            system_prompt=self.system_prompt,
            temperature=0.1,
            use_memory=True,  # Reference analysis from stage 1
            response_format={"type": "json_object"}
        )
        
        self.metrics.update_from_response(response)
        
        try:
            self.schema = json.loads(response['reply'])
        except json.JSONDecodeError:
            self.schema = parse_llm_json(response['reply'])
            (self.output_dir / 'schema_raw.txt').write_text(response['reply'])
            
        
        # Same constraints as iterative pipeline (fair comparison)
        self.schema = enforce_schema_constraints(self.schema)
        
        with open(self.output_dir / 'final_schema.json', 'w') as f:
            json.dump(self.schema, f, indent=2)
        
        # Free analysis context now that we have the schema
        self.client.reset_conversation()
        
        return self.schema
    
    def fill_schema(self, clinical_note: str) -> str:
        """Stage 3: Fill schema for a single note. No memory between calls."""
        if self.schema is None:
            raise ValueError("Schema must be generated first")
        
        prompt = f"""Extract structured data from the clinical note using the schema.

Rules:
- Only extract information explicitly stated in the note
- Omit fields that have no corresponding information in the note
- Do not infer, guess, or invent values
- Match the data types specified in the schema

SCHEMA:
{json.dumps(self.schema, indent=2)}

CLINICAL NOTE:
{clinical_note}

Output the extracted JSON object. No explanation, no markdown."""
        
        try:
            response = self.filler_client.generate(
                prompt,
                system_prompt=self.system_prompt,
                temperature=0.1,
                use_memory=False,
                response_format={"type": "json_object"}
            )
            self.metrics.update_from_response(response)
            return response['reply']
        except Exception as e:
            print(f"  Error filling: {e}")
            return json.dumps({"error": str(e)})
    
    def save_metrics(self):
        """Save experiment metrics."""
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
            'total_tokens': self.metrics.total_tokens,
            'prompt_tokens': self.metrics.prompt_tokens,
            'completion_tokens': self.metrics.completion_tokens,
            'avg_tokens_per_call': self.metrics.total_tokens / max(self.metrics.api_calls, 1),
        }
        
        pd.DataFrame([metrics_dict]).to_csv(self.output_dir / 'experiment_metrics.csv', index=False)
        
        print(f"\n{'='*60}")
        print(f"BASELINE METRICS: One-Shot ICL")
        print(f"{'='*60}")
        print(f"Duration: {self.metrics.total_duration/60:.2f} min")
        print(f"Peak Memory: {self.metrics.peak_memory_mb/1024:.2f} GB")
        print(f"API Calls: {self.metrics.api_calls}")
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
    
    # SGLang server (must already be running, ideally with --context-length 32768)
    SGLANG_BASE_URL = "http://0.0.0.0:30000/v1"
    
    # Output: results/oneshot_icl_schema/<model_name>/
    OUTPUT_DIR = f"./results/oneshot_icl_schema/{MODEL_NAME}"
    
    TRAIN_DATA_PATH = os.environ.get("TRAIN_DATA_PATH", "datasets/cluster_training_data.csv")
    TEST_DATA_PATH = os.environ.get("TEST_DATA_PATH", "datasets/cluster_testing_data.csv")
    
    # ---------- Run ----------
    
    print("="*80)
    print(f"BASELINE: One-Shot ICL Schema Generation")
    print(f"Model:  {MODEL_NAME}")
    print(f"Server: {SGLANG_BASE_URL}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Note: SGLang server must already be running.")
    print("="*80)
    
    try:
        train_data = pd.read_csv(TRAIN_DATA_PATH)
        test_data = pd.read_csv(TEST_DATA_PATH, index_col=0)
        
        print(f"\nLoaded {len(train_data)} training notes, {len(test_data)} test notes")
        
        generator = OneShotSchemaGenerator(
            model_name=MODEL_NAME,
            output_dir=OUTPUT_DIR,
            base_url=SGLANG_BASE_URL,
        )
        
        # Stage 1: Pattern analysis
        print("\n=== Step 1: Pattern Analysis ===")
        generator.analyze_patterns(train_data['clinical_notes'], max_notes=15)
        print("✓ Pattern analysis complete")
        
        # Stage 2: Schema generation
        print("\n=== Step 2: Schema Generation ===")
        generator.generate_schema()
        print("✓ Schema generated")
        
        # Stage 3: Fill test data
        print(f"\n=== Step 3: Filling {len(test_data)} test notes ===")
        filled_results = []
        for idx, note in enumerate(test_data['clinical_notes'].values):
            print(f"  Filling note {idx + 1}/{len(test_data)}")
            filled_results.append(generator.fill_schema(note))
        
        test_data = test_data.copy()
        test_data['filled_schema'] = filled_results
        test_data.to_csv(generator.output_dir / 'filled_test.csv', index=False)
        
        generator.save_metrics()
        
        print("\n✓ Baseline experiment completed!")
        print(f"\nNext step: python run_evaluation.py")
        
    except Exception as e:
        print(f"\n✗ Failed: {e}")
        import traceback
        traceback.print_exc()