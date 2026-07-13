"""
DeepEval-based evaluation — schema-aware, NaN-safe.

Metrics:
  - CustomSummarizationMetric  (precision/recall F1; fmt-aware prompts)
  - GEval Hallucination        (precision of clinical claims ↑)
  - GEval Omission             (recall of clinical content   ↑)
  - DAGHallucination           (DAG precision ↑, 0–1)
  - DAGCoverage                (DAG recall    ↑, 0–1)

Design notes
------------
* The custom metric is run in its OWN evaluate() pass, separate from the
  GEval/DAG metrics. Even though the custom metric is now defensive, this
  isolation guarantees that any unexpected failure inside it can never
  poison the sibling GEval coroutines for the same row (the original
  "everything NaN except DAG" bug).
* All metrics receive `fmt`. For fmt == "json" the prompts are told the
  candidate is a FILLED CLINICAL SCHEMA so structured key/value output is
  judged on clinical content, not prose style — this makes the
  schema-vs-summary comparison fair.
* DAG WAS a single multi-root metric. In the installed DeepEval a leaf
  VerdictNode sets `metric.score = self.score / 10` by ASSIGNMENT and the
  root nodes run sequentially, so the coverage branch silently overwrote
  the hallucination branch — the old `dag_score` was coverage-only and the
  hallucination judge calls were wasted. It is now TWO single-root metrics
  (DAGHallucination, DAGCoverage) recorded in separate columns, so DAG
  precision and recall are both real and independently auditable.
* Hallucination and Omission GEval rubrics are now STANDARDISED: identical
  4-tier structure, identical score bands, mirrored wording (precision vs
  recall). This makes geval_hallucination and geval_omission directly
  comparable and a clean F1 of the two well-defined.
* The custom metric's simple-coverage path (coverage_score /
  coverage_verdicts) never entered any score and always serialised to
  None; those columns are removed. complex_coverage_score carries recall.
* Logging goes through eval_logging (set EVAL_LOG_LEVEL=DEBUG for detail).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import pandas as pd

from config import DEEPEVAL_N_COMPLEX_QUESTIONS, JUDGE_MODEL
from custom_metric import CustomSummarizationMetric
from eval_logging import get_logger
from flatten import to_candidate_text

log = get_logger(__name__)


# ═════════════════════════════════════════════════════════════
# Schema-aware prompt fragments
# ═════════════════════════════════════════════════════════════
def _schema_note(fmt: str) -> str:
    """Instruction appended to GEval/DAG steps so a filled JSON schema is
    not penalised for lacking prose narrative."""
    if fmt != "json":
        return ""
    return (
        " NOTE: the actual_output is a FILLED STRUCTURED CLINICAL SCHEMA "
        "(key/value fields), not a prose summary. Judge it ONLY on clinical "
        "content fidelity. Do not penalise it for absence of narrative flow, "
        "section prose, or headings. Treat empty / 'N/A' / 'not documented' "
        "fields as 'information not asserted', never as a hallucination."
    )


# ─────────────────────────────────────────────────────────────
# Standardised rubric.
#
# Hallucination (precision) and Omission (recall) previously had rubrics
# whose tiers were worded asymmetrically ("multiple/one hallucinated fact"
# vs "major/important/minor missing"). Same 0–10 bands, but the semantics
# per band were not mirror images, so the two scores were not on a truly
# common scale and an F1 of them mixed two different rating philosophies.
#
# This single shared 4-tier scaffold is parameterised by the error noun so
# both metrics use IDENTICAL band boundaries and IDENTICAL severity logic
# (count + clinical importance of the error), differing only in whether the
# error is "unsupported content" (precision) or "omitted content" (recall).
# ─────────────────────────────────────────────────────────────
def _standard_rubric(error_kind: str):
    """error_kind: short noun phrase for the failure, e.g.
    'unsupported (hallucinated) clinical content' or
    'omitted clinically important content'."""
    from deepeval.metrics.g_eval import Rubric
    return [
        Rubric(score_range=(0, 2),
                expected_outcome=(f"Severe: multiple instances of {error_kind}, "
                                  f"or {error_kind} affecting a clinically "
                                  f"critical element.")),
        Rubric(score_range=(3, 5),
                expected_outcome=(f"Major: one clear instance of {error_kind} "
                                  f"affecting an important clinical element.")),
        Rubric(score_range=(6, 8),
                expected_outcome=(f"Minor: only small/low-importance {error_kind}; "
                                  f"all clinically critical elements correct.")),
        Rubric(score_range=(9, 10),
                expected_outcome=(f"None: no {error_kind}; the output is fully "
                                  f"correct on this axis.")),
    ]


# ═════════════════════════════════════════════════════════════
# Metric builders
# ═════════════════════════════════════════════════════════════
def build_custom_summarization(fmt: str = "text",
                               judge_model: str = JUDGE_MODEL,
                               n_complex_questions: int = DEEPEVAL_N_COMPLEX_QUESTIONS):
    return CustomSummarizationMetric(
        threshold=0.5,
        model=judge_model,
        n_complex_questions=n_complex_questions,
        verbose_mode=True,   # scores + questions live in verbose_logs JSON
        fmt=fmt,             # drives schema-fair prompt wording
    )


def build_hallucination_geval(fmt: str = "text", judge_model: str = JUDGE_MODEL):
    from deepeval.metrics import GEval
    from deepeval.test_case import LLMTestCaseParams
    note = _schema_note(fmt)
    return GEval(
        name="Hallucination",
        model=judge_model,
        evaluation_steps=[
            "Extract every clinical claim from the actual_output. A clinical "
            "claim is any statement about the patient's condition, history, "
            "medications, findings, diagnosis, or management." + note,
            "For each extracted claim, verify whether it is directly stated or "
            "clearly inferable from the input (source clinical note). A claim "
            "is hallucinated if it introduces diagnoses, medications, findings, "
            "numerical values, or events not present in the source.",
            "Assign a score using the rubric, weighting by BOTH the number of "
            "hallucinated claims AND the clinical importance of what they "
            "misstate (a single critical fabrication outranks several trivial "
            "ones).",
        ],
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        rubric=_standard_rubric("unsupported (hallucinated) clinical content"),
        verbose_mode=False,
    )


def build_omission_geval(fmt: str = "text", judge_model: str = JUDGE_MODEL):
    from deepeval.metrics import GEval
    from deepeval.test_case import LLMTestCaseParams
    note = _schema_note(fmt)
    return GEval(
        name="Omission",
        model=judge_model,
        evaluation_steps=[
            "Read the input (source clinical note) and identify all clinically "
            "important information: patient background, reason for visit, "
            "relevant history, examination/investigative findings, diagnostic "
            "conclusions, and management or follow-up plan. Do not assume any "
            "fixed section structure.",
            "For each identified clinical concept, determine whether it is "
            "captured in the actual_output, even if expressed differently "
            "(a schema field counts as captured)." + note,
            "Assign a score using the rubric, weighting by BOTH the number of "
            "omitted concepts AND their clinical importance (omitting one "
            "critical concept outranks omitting several trivial ones).",
        ],
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        rubric=_standard_rubric("omitted clinically important content"),
        verbose_mode=False,
    )


# ── DAG: split into two SINGLE-root metrics ──────────────────
def build_dag_hallucination(fmt: str = "text", judge_model: str = JUDGE_MODEL):
    """DAG precision (faithfulness) — SINGLE root.

    Single-root is mandatory: a leaf VerdictNode sets
    `metric.score = self.score / 10` by ASSIGNMENT and DeepEval executes
    root nodes sequentially, so the previous combined two-root DAG had this
    branch silently overwritten by the coverage branch. Separate single-
    root metrics make each score real and recorded independently.
    """
    from deepeval.metrics import DAGMetric
    from deepeval.metrics.dag import (
        BinaryJudgementNode,
        DeepAcyclicGraph,
        NonBinaryJudgementNode,
        TaskNode,
        VerdictNode,
    )
    from deepeval.test_case import LLMTestCaseParams
    note = _schema_note(fmt)

    grounding_degree = NonBinaryJudgementNode(
        criteria=("Given the extracted clinical claims and the original "
                  "clinical note (input), to what degree are the claims "
                  "grounded in the source note?"),
        evaluation_params=[LLMTestCaseParams.INPUT],
        children=[
            VerdictNode(verdict="All claims are fully grounded",         score=10),
            VerdictNode(verdict="Mostly grounded with minor inferences", score=6),
            VerdictNode(verdict="Some claims lack grounding",            score=3),
        ],
    )
    hallucination_check = BinaryJudgementNode(
        criteria=("Do any of the extracted clinical claims introduce "
                  "information (diagnoses, medications, findings, values, or "
                  "events) NOT present in the original clinical note (input)?"),
        evaluation_params=[LLMTestCaseParams.INPUT],
        children=[
            VerdictNode(verdict=True,  score=0),
            VerdictNode(verdict=False, child=grounding_degree),
        ],
    )
    extract_claims = TaskNode(
        instructions=("Extract every distinct clinical claim from the "
                       "actual_output. A clinical claim is any statement about "
                       "the patient's condition, background, history, findings, "
                       "diagnosis, or management. List each on its own line."
                       + note),
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
        output_label="Extracted clinical claims",
        children=[hallucination_check],
    )

    dag = DeepAcyclicGraph(root_nodes=[extract_claims])
    return DAGMetric(name="DAGHallucination", dag=dag,
                     model=judge_model, include_reason=True, verbose_mode=False)


def build_dag_coverage(fmt: str = "text", judge_model: str = JUDGE_MODEL):
    """DAG recall (coverage) — SINGLE root. See build_dag_hallucination
    for why the previously combined DAG had to be split."""
    from deepeval.metrics import DAGMetric
    from deepeval.metrics.dag import (
        DeepAcyclicGraph,
        NonBinaryJudgementNode,
        TaskNode,
        VerdictNode,
    )
    from deepeval.test_case import LLMTestCaseParams

    coverage_degree = NonBinaryJudgementNode(
        criteria=("Given the key clinical facts from the source note (input) "
                  "and the actual_output, what proportion of those facts is "
                  "captured in the output?"),
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
        children=[
            VerdictNode(verdict="All key facts are present",                    score=10),
            VerdictNode(verdict="Most key facts present, minor details missing", score=6),
            VerdictNode(verdict="Major clinical facts are missing",              score=2),
        ],
    )
    extract_key_facts = TaskNode(
        instructions=("Read the input (source clinical note) and identify all "
                       "key clinical facts: patient background, reason for "
                       "visit, relevant history, findings, diagnostic "
                       "conclusions, and management plan. Do not assume fixed "
                       "headings. List each key fact on its own line."),
        evaluation_params=[LLMTestCaseParams.INPUT],
        output_label="Key clinical facts from source note",
        children=[coverage_degree],
    )

    dag = DeepAcyclicGraph(root_nodes=[extract_key_facts])
    return DAGMetric(name="DAGCoverage", dag=dag,
                     model=judge_model, include_reason=True, verbose_mode=False)


def build_geval_dag_metrics(fmt: str, judge_model: str = JUDGE_MODEL) -> List[Any]:
    """First-party metrics — run together (they don't poison each other).

    DAG is two single-root metrics (precision + recall) instead of one
    combined metric whose precision branch was silently discarded.
    """
    return [
        build_hallucination_geval(fmt, judge_model),
        build_omission_geval(fmt, judge_model),
        build_dag_hallucination(fmt, judge_model),
        build_dag_coverage(fmt, judge_model),
    ]


# ═════════════════════════════════════════════════════════════
# verbose_logs parser for CustomSummarizationMetric
#
# coverage_score / coverage_verdicts (the simple yes/no path) are
# intentionally NOT extracted: that path does not enter the score and
# always serialised to None. complex_coverage_* is the recall signal.
# ═════════════════════════════════════════════════════════════
def _parse_verbose_logs(verbose_logs_str: Optional[str]) -> Dict[str, Any]:
    empty = {
        "alignment_score": None,
        "complex_coverage_score": None,
        "assessment_questions": None,
        "alignment_verdicts": None,
        "complex_assessment_questions": None,
        "complex_coverage_verdicts": None,
    }
    if not verbose_logs_str:
        return empty
    try:
        logs = json.loads(verbose_logs_str)
    except (json.JSONDecodeError, TypeError):
        return empty

    def fmt_alignment(vs):
        if not vs:
            return None
        out = []
        for i, v in enumerate(vs):
            verdict = v.get("verdict", "")
            mark = {"yes": "✓", "no": "✗"}.get(verdict, "?")
            reason = v.get("reason") or ""
            out.append(f"Claim {i+1}: {mark} {verdict}" + (f" — {reason}" if reason else ""))
        return "\n".join(out)

    def fmt_complex_qs(qs):
        if not qs:
            return None
        return "\n".join(
            f"Q: {q.get('question','')} | A: {q.get('answer','')} | "
            f"importance={q.get('importance','')}" for q in qs
        )

    def fmt_complex_verdicts(vs):
        if not vs:
            return None
        return "\n".join(
            f"Q: {v.get('question','')} | score={v.get('score','')} | "
            f"expected='{v.get('original_answer','')}' | "
            f"got='{v.get('summary_answer','')}' | reason: {v.get('reason','')}"
            for v in vs
        )

    return {
        "alignment_score":              logs.get("alignment_score"),
        "complex_coverage_score":       logs.get("complex_coverage_score"),
        "assessment_questions":         json.dumps(logs.get("assessment_questions", [])),
        "alignment_verdicts":           fmt_alignment(logs.get("alignment_verdicts")),
        "complex_assessment_questions": fmt_complex_qs(logs.get("complex_assessment_questions")),
        "complex_coverage_verdicts":    fmt_complex_verdicts(logs.get("complex_coverage_verdicts")),
    }


# ═════════════════════════════════════════════════════════════
# Per-metric result parser
# ═════════════════════════════════════════════════════════════
def _round(x):
    return round(x, 4) if x is not None else None


def _norm_dag(score) -> Optional[float]:
    """DAG metric.score is ALREADY 0–1: the leaf VerdictNode does
    `metric.score = self.score / 10` internally. We therefore only clamp
    defensively; we MUST NOT divide by 10 again.

    The original code's `score / 10 if score > 1.0 else score` was a latent
    correctness bug: it assumed the raw score was 0–10 and conditionally
    re-divided, which (had the assumption held) would have turned a poor
    0–10 result like 1.0 into 0.1 while leaving good results unscaled —
    silently corrupting the low end. It was only ever harmless because the
    real score is already <=1 so the branch never fired. Replaced with an
    unconditional defensive clamp."""
    if score is None:
        return None
    try:
        s = float(score)
    except (TypeError, ValueError):
        return None
    return round(min(max(s, 0.0), 1.0), 4)


def _parse_metric(m) -> Dict[str, Any]:
    name, score = m.name, m.score
    err = getattr(m, "error", None)
    if score is None and err:
        # Surface WHY a metric produced nothing — the missing piece in the
        # original code that made NaNs opaque.
        log.warn("metric '%s' returned no score: %s", name, str(err)[:160])

    out: Dict[str, Any] = {}
    if name == "Custom Summarization Metric":
        out["summ_score"] = _round(score)
        out["summ_reason"] = m.reason
        out.update(_parse_verbose_logs(getattr(m, "verbose_logs", None)))
    elif name == "Hallucination [GEval]":
        out["geval_hallucination"] = _round(score)
        out["geval_hall_reason"] = m.reason
    elif name == "Omission [GEval]":
        out["geval_omission"] = _round(score)
        out["geval_omit_reason"] = m.reason
    elif name == "DAGHallucination [DAG]":
        out["dag_hallucination"] = _norm_dag(score)
        out["dag_hall_reason"] = m.reason
    elif name == "DAGCoverage [DAG]":
        out["dag_coverage"] = _norm_dag(score)
        out["dag_cov_reason"] = m.reason
    else:
        log.warn("unrecognised metric '%s' (score=%s)", name, score)
    return out


EMPTY_RESULT_TEMPLATE = {
    "summ_score": None, "summ_reason": None, "alignment_score": None,
    "complex_coverage_score": None,
    "assessment_questions": None,
    "alignment_verdicts": None, "complex_assessment_questions": None,
    "complex_coverage_verdicts": None, "geval_hallucination": None,
    "geval_hall_reason": None, "geval_omission": None,
    "geval_omit_reason": None,
    "dag_hallucination": None, "dag_hall_reason": None,
    "dag_coverage": None, "dag_cov_reason": None,
}


# ═════════════════════════════════════════════════════════════
# Chunk runner
# ═════════════════════════════════════════════════════════════
def _run_chunk(test_cases, metrics, label: str, max_concurrent: int = 5):
    """One evaluate() call over a chunk. Returns EvaluationResult or None.

    ignore_errors=True keeps a single bad case from killing the chunk; the
    metric then reports score=None + an error which _parse_metric logs.
    """
    from deepeval import evaluate
    from deepeval.evaluate.configs import AsyncConfig, DisplayConfig, ErrorConfig

    log.info("%s: submitting %d cases (max_concurrent=%d)",
             label, len(test_cases), max_concurrent)
    try:
        return evaluate(
            test_cases=test_cases,
            metrics=metrics,
            async_config=AsyncConfig(run_async=True, throttle_value=0,
                                     max_concurrent=max_concurrent),
            error_config=ErrorConfig(ignore_errors=True,
                                     skip_on_missing_params=False),
            display_config=DisplayConfig(show_indicator=False,
                                         print_results=False),
        )
    except Exception as e:  # noqa: BLE001 - chunk-level catastrophic failure
        log.error("%s failed: %s: %s", label, type(e).__name__, e)
        return None


def _merge_results(results, chunk_cases, chunk_indices, df, approach,
                   model_name, fmt, into: Dict[int, dict]):
    """Fold one evaluate() result set into the per-row accumulator `into`.

    CRITICAL: DeepEval does NOT guarantee that results.test_results is the
    same length as, or in the same order as, the submitted test cases.
    Under ignore_errors=True it silently drops failed cases and may reorder
    the rest. Mapping positionally (test_results[k] -> chunk_indices[k]) is
    therefore a correctness bug: it attributes scores to the wrong note_id.

    We map via TestResult.index (submission-order position DeepEval sets),
    with an input-text fallback and an input-text cross-check that refuses
    to write a score whose input does not match the mapped case.

    `into` is keyed by doc_index so the custom pass and the geval/dag pass
    write into the SAME row.
    """
    if results is None or not getattr(results, "test_results", None):
        return

    n = len(chunk_cases)
    seen = 0
    for result in results.test_results:
        ridx = getattr(result, "index", None)

        if isinstance(ridx, int) and 0 <= ridx < n:
            pos = ridx
        else:
            pos = None
            r_in = getattr(result, "input", None)
            if r_in is not None:
                for p, tc in enumerate(chunk_cases):
                    if tc.input == r_in:
                        pos = p
                        break
            if pos is None:
                log.warn("could not map a TestResult back to a case "
                         "(index=%s); skipping it", ridx)
                continue

        r_in = getattr(result, "input", None)
        if r_in is not None and r_in != chunk_cases[pos].input:
            log.error("input mismatch at pos %d (index=%s) — refusing to "
                      "attribute scores to avoid corrupting note_id", pos, ridx)
            continue

        doc_idx = chunk_indices[pos]
        src = df.iloc[doc_idx]
        row = into.get(doc_idx)
        if row is None:
            row = {
                "doc_index": doc_idx, "note_id": src["note_id"],
                "approach": approach, "model": model_name,
                "summary_format": fmt, **EMPTY_RESULT_TEMPLATE,
            }
            into[doc_idx] = row
        for m in result.metrics_data:
            row.update(_parse_metric(m))
        seen += 1

    if seen < n:
        log.warn("evaluate() returned %d usable results for %d submitted "
                 "cases (%d dropped by DeepEval, likely judge/metric errors)",
                 seen, n, n - seen)


# ═════════════════════════════════════════════════════════════
# Public API
# ═════════════════════════════════════════════════════════════
def score_run(df: pd.DataFrame, fmt: str,
              judge_model: str = JUDGE_MODEL,
              chunk_size: int = 10,
              max_concurrent: int = 5,
              checkpoint_path: Optional[str] = None,
              ) -> pd.DataFrame:
    """Run all DeepEval metrics over a loaded run dataframe.

    Expects standardized columns: source_text, model_output, note_id,
    approach, model. Evaluates in chunks; within each chunk the custom
    metric runs in its own evaluate() pass (isolation), then the
    GEval/DAG metrics run together. Both passes write into the same row.
    Checkpoints after every chunk for crash-safe resume.
    """
    from deepeval.test_case import LLMTestCase
    from pathlib import Path

    approach = df["approach"].iloc[0] if len(df) else "?"
    model_name = df["model"].iloc[0] if len(df) else "?"

    # Build test cases (skip content-free outputs, not just empty ones).
    test_cases, index_map = [], []
    for i, row in df.iterrows():
        actual = to_candidate_text(str(row["model_output"]), fmt)
        if len(actual.strip()) < 3 or not any(c.isalpha() for c in actual):
            log.warn("skip row %s note_id=%s: empty/contentless after flatten",
                     i, row["note_id"])
            continue
        test_cases.append(LLMTestCase(input=str(row["source_text"]),
                                       actual_output=actual))
        index_map.append(i)

    if not test_cases:
        log.warn("no test cases to evaluate")
        return pd.DataFrame()

    # Resume support.
    completed: set = set()
    if checkpoint_path:
        cp = Path(checkpoint_path)
        if cp.exists():
            try:
                completed = set(pd.read_csv(cp)["doc_index"].astype(int))
                log.info("resume: %d rows already in %s", len(completed), cp)
            except Exception as e:  # noqa: BLE001
                log.warn("could not read checkpoint %s: %s", cp, e)

    remaining = [(tc, idx) for tc, idx in zip(test_cases, index_map)
                 if idx not in completed]
    if not remaining:
        log.info("resume: all %d cases already complete", len(index_map))
        return pd.read_csv(checkpoint_path) if checkpoint_path else pd.DataFrame()

    tcs = [tc for tc, _ in remaining]
    idxs = [idx for _, idx in remaining]

    # Two metric groups: custom (isolated) + geval/dag (together).
    custom_metric = [build_custom_summarization(fmt, judge_model)]
    geval_dag = build_geval_dag_metrics(fmt, judge_model)

    all_rows: List[dict] = []
    n_chunks = (len(tcs) + chunk_size - 1) // chunk_size
    log.info("scoring %d cases in %d chunk(s), fmt=%s", len(tcs), n_chunks, fmt)

    for c in range(n_chunks):
        s, e = c * chunk_size, min((c + 1) * chunk_size, len(tcs))
        chunk_cases, chunk_idx = tcs[s:e], idxs[s:e]
        base = f"chunk {c+1}/{n_chunks} (cases {s}-{e-1})"

        acc: Dict[int, dict] = {}

        # Pass 1 — custom metric in isolation. A failure here cannot reach
        # the geval/dag coroutines because they run in a separate call.
        r1 = _run_chunk(chunk_cases, custom_metric, base + " [custom]",
                        max_concurrent)
        _merge_results(r1, chunk_cases, chunk_idx, df, approach,
                       model_name, fmt, acc)

        # Pass 2 — geval + dag (both single-root DAGs) together.
        r2 = _run_chunk(chunk_cases, geval_dag, base + " [geval+dag]",
                        max_concurrent)
        _merge_results(r2, chunk_cases, chunk_idx, df, approach,
                       model_name, fmt, acc)

        # Ensure every case in the chunk yields a row even if both passes
        # failed for it (so a resume doesn't loop forever on bad rows).
        chunk_rows = []
        for i in chunk_idx:
            if i not in acc:
                src = df.iloc[i]
                acc[i] = {
                    "doc_index": i, "note_id": src["note_id"],
                    "approach": approach, "model": model_name,
                    "summary_format": fmt, **EMPTY_RESULT_TEMPLATE,
                }
                log.warn("row %s produced no metric data (both passes empty)", i)
            chunk_rows.append(acc[i])

        for r in chunk_rows:
            log.info("row %4d → summ=%s hall=%s omit=%s dagH=%s dagC=%s",
                     r["doc_index"], r["summ_score"], r["geval_hallucination"],
                     r["geval_omission"], r["dag_hallucination"],
                     r["dag_coverage"])

        all_rows.extend(chunk_rows)

        if checkpoint_path and chunk_rows:
            cp = Path(checkpoint_path)
            cdf = pd.DataFrame(chunk_rows)
            if cp.exists():
                # Align to the existing header so a schema change (the DAG
                # split, removed coverage_* cols) cannot silently misalign
                # appended columns. Warn loudly if the on-disk schema is the
                # OLD one — that file should be regenerated with --overwrite.
                try:
                    existing_cols = list(pd.read_csv(cp, nrows=0).columns)
                    if set(existing_cols) != set(cdf.columns):
                        log.warn("checkpoint %s has a DIFFERENT column schema "
                                 "(likely a pre-DAG-split file). Appending "
                                 "aligned to the OLD header; rerun with "
                                 "--overwrite for a clean, consistent CSV.",
                                 cp)
                    cdf = cdf.reindex(columns=existing_cols)
                except Exception:  # noqa: BLE001
                    pass
                cdf.to_csv(cp, mode="a", header=False, index=False)
            else:
                cp.parent.mkdir(parents=True, exist_ok=True)
                cdf.to_csv(cp, index=False)
            log.info("checkpoint: wrote %d rows → %s", len(chunk_rows), cp)

    if checkpoint_path and Path(checkpoint_path).exists():
        return pd.read_csv(checkpoint_path)
    return pd.DataFrame(all_rows)