"""
Central configuration for evaluation runs.

Edit RESULTS_ROOT and JUDGE_MODEL to suit your environment.
The APPROACHES table is the single source of truth for:
  - which file to read for each approach
  - which column holds the source note
  - which column holds the model output
  - whether the output is JSON (filled schema) or plain text
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# Paths
#
# RESULTS_ROOT is the parent of {approach}/{model}/... produced by the
# generation step. Override without editing this file via the environment:
#     export RESULTS_ROOT=/path/to/results
# ─────────────────────────────────────────────────────────────
RESULTS_ROOT = Path(os.environ.get("RESULTS_ROOT", "results"))          # parent of {approach}/{model}/...
EVAL_OUTPUT_ROOT = Path(os.environ.get("EVAL_OUTPUT_ROOT", "eval_outputs"))  # per-run CSVs land here

# ─────────────────────────────────────────────────────────────
# Judge / scoring config
# ─────────────────────────────────────────────────────────────
JUDGE_MODEL = "gpt-5-mini"   # used by DeepEval (GEval, DAG, CustomSummarization)
MQAG_NUM_QUESTIONS = 15
SELFCHECK_BERT_CONSISTENT_THRESHOLD = 0.5  # bertscore < threshold counts as "consistent"
DEEPEVAL_N_COMPLEX_QUESTIONS = 8
SEED = 42

# ─────────────────────────────────────────────────────────────
# Approach metadata
#
# Each entry describes one approach folder under RESULTS_ROOT.
#   file        : filename inside results/{approach}/{model}/
#   source_col  : column with the original clinical note
#   output_col  : column with the generated summary or filled schema
#   format      : "text" or "json" — controls how output is flattened
# ─────────────────────────────────────────────────────────────
APPROACHES = {
    "base_extractive": {
        "file":       "summaries.csv",
        "source_col": "clinical_notes",
        "output_col": "summary",
        "format":     "text",
    },
    "cot_abstractive": {
        "file":       "summaries.csv",
        "source_col": "clinical_notes",
        "output_col": "summary",
        "format":     "text",
    },
    "standard_abstractive": {
        "file":       "summaries.csv",
        "source_col": "clinical_notes",
        "output_col": "summary",
        "format":     "text",
    },
    "iterative_schema": {
        "file":       "filled_test.csv",
        "source_col": "clinical_notes",
        "output_col": "filled_schema",
        "format":     "json",
    },
    "oneshot_icl_schema": {
        "file":       "filled_test.csv",
        "source_col": "clinical_notes",
        "output_col": "filled_schema",
        "format":     "json",
    },
}


def approach_config(approach: str) -> dict:
    """Lookup with a clear error if an unknown approach is passed."""
    if approach not in APPROACHES:
        raise KeyError(
            f"Unknown approach '{approach}'. Known: {list(APPROACHES)}. "
            f"Add it to APPROACHES in config.py."
        )
    return APPROACHES[approach]
