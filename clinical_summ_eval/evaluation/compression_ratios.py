"""
Compute character- and word-level compression ratios across all runs.

Reads existing generation outputs directly (results/{approach}/{model}/...):
  - Summarization approaches: uses precomputed summary_word_count, summary_char_count,
    original_word_count, original_char_count from summaries.csv when available.
  - Schema approaches: computes length from the filled_schema column. By default
    uses the JSON-as-string length; pass --schema-mode flattened to use the
    declarative statements (matching how the schemas are scored in the eval).

Usage:
  python compression_ratios.py                          # all runs, JSON mode
  python compression_ratios.py --schema-mode flattened  # flattened statements
  python compression_ratios.py --approaches iterative_schema

Outputs:
  eval_outputs/compression/per_document.csv   one row per (approach, model, note)
  eval_outputs/compression/summary.csv        mean/std/min/max per (approach, model)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

from config import EVAL_OUTPUT_ROOT, approach_config
from flatten import flatten_json
from io_utils import discover_runs, load_run

# ─────────────────────────────────────────────────────────────
# Length helpers
# ─────────────────────────────────────────────────────────────
def text_lengths(s: str) -> tuple[int, int]:
    """Return (char_count, word_count) for a string."""
    if not isinstance(s, str) or not s.strip():
        return 0, 0
    return len(s), len(s.split())


def schema_lengths(filled_json: str, mode: str) -> tuple[int, int]:
    """Length of a filled schema.

    mode='json'        → length of the raw JSON string (compact, no extra whitespace)
    mode='flattened'   → length of the declarative 'key > sub: value' statements
                         joined into a single string (same form scored by SelfCheck/DeepEval)
    """
    if not isinstance(filled_json, str) or not filled_json.strip():
        return 0, 0
    try:
        obj = json.loads(filled_json)
    except (json.JSONDecodeError, TypeError):
        return 0, 0

    if mode == "json":
        # Compact JSON: keep it apples-to-apples vs. a prose summary by stripping
        # the artificial whitespace from pretty-printing.
        s = json.dumps(obj, separators=(",", ":"))
    else:  # flattened
        stmts = flatten_json(obj)
        s = ". ".join(stmts) + ("." if stmts else "")
    return len(s), len(s.split())


# ─────────────────────────────────────────────────────────────
# Per-run processing
# ─────────────────────────────────────────────────────────────
def compute_for_run(run, schema_mode: str) -> Optional[pd.DataFrame]:
    """Return per-document compression dataframe for a single run, or None."""
    cfg = approach_config(run.approach)

    # Use load_run so columns are standardised and blank rows are dropped
    df = load_run(run)
    if df.empty:
        return None

    rows = []
    for _, row in df.iterrows():
        src = str(row["source_text"])
        src_char, src_word = text_lengths(src)

        if run.fmt == "json":
            out_char, out_word = schema_lengths(row["model_output"], schema_mode)
        else:
            out_char, out_word = text_lengths(str(row["model_output"]))

        # Guard against zero-length source (shouldn't happen post-load_run, but safe)
        char_ratio = out_char / src_char if src_char else None
        word_ratio = out_word / src_word if src_word else None

        rows.append({
            "approach":           run.approach,
            "model":              run.model,
            "note_id":            row["note_id"],
            "fmt":                run.fmt,
            "schema_mode":        schema_mode if run.fmt == "json" else None,
            "original_char_count": src_char,
            "original_word_count": src_word,
            "output_char_count":  out_char,
            "output_word_count":  out_word,
            "char_compression":   round(char_ratio, 4) if char_ratio is not None else None,
            "word_compression":   round(word_ratio, 4) if word_ratio is not None else None,
            "char_reduction":     round(1 - char_ratio, 4) if char_ratio is not None else None,
            "word_reduction":     round(1 - word_ratio, 4) if word_ratio is not None else None,
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────────────────────────
def summarise(per_doc: pd.DataFrame) -> pd.DataFrame:
    """mean / std / min / max per (approach, model) over the four ratio columns."""
    metric_cols = ["char_compression", "word_compression",
                   "char_reduction",   "word_reduction",
                   "output_char_count", "output_word_count"]
    grouped = per_doc.groupby(["approach", "model"], dropna=False)
    agg = grouped[metric_cols].agg(["mean", "std", "min", "max", "count"])
    agg.columns = [f"{m}_{stat}" for m, stat in agg.columns]
    return agg.reset_index()


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--approaches", nargs="*", default=None)
    p.add_argument("--models", nargs="*", default=None)
    p.add_argument("--schema-mode", choices=["json", "flattened"], default="json",
                   help="How to measure schema output length. "
                        "'json' (default) = raw compact JSON string; "
                        "'flattened' = declarative statements as scored by the eval.")
    args = p.parse_args()

    runs = discover_runs(approaches=args.approaches, models=args.models)
    if not runs:
        print("No runs discovered. Check RESULTS_ROOT and filters.")
        sys.exit(1)

    print(f"Discovered {len(runs)} run(s). Schema mode: {args.schema_mode}\n")

    all_rows = []
    for run in runs:
        print(f"  • {run.slug}")
        df = compute_for_run(run, schema_mode=args.schema_mode)
        if df is None or df.empty:
            print(f"    [Skip] empty after loading")
            continue
        print(f"    {len(df)} docs  "
              f"mean char_compression={df['char_compression'].mean():.3f}  "
              f"mean word_compression={df['word_compression'].mean():.3f}")
        all_rows.append(df)

    if not all_rows:
        print("No data produced.")
        return

    per_doc = pd.concat(all_rows, ignore_index=True)
    out_dir = EVAL_OUTPUT_ROOT / "compression"
    out_dir.mkdir(parents=True, exist_ok=True)
    per_doc_path = out_dir / "per_document.csv"
    summary_path = out_dir / "summary.csv"
    per_doc.to_csv(per_doc_path, index=False)
    print(f"\n[Saved] {per_doc_path} ({len(per_doc)} rows)")

    summary = summarise(per_doc)
    summary.to_csv(summary_path, index=False)
    print(f"[Saved] {summary_path} ({len(summary)} rows)\n")

    print("Per-(approach, model) means:")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        cols = ["approach", "model",
                "char_compression_mean", "char_compression_std",
                "word_compression_mean", "word_compression_std",
                "output_word_count_mean", "output_word_count_count"]
        print(summary[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()