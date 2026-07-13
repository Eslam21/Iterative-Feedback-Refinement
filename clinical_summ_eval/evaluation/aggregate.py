"""
Aggregate per-run eval CSVs into a single paper-ready table.

  python aggregate.py                          # both selfcheck + deepeval
  python aggregate.py --source selfcheck       # only selfcheck
  python aggregate.py --source deepeval        # only deepeval

Outputs (under eval_outputs/aggregated/):
  selfcheck_all_rows.csv          — every per-doc row, with approach/model cols
  selfcheck_summary.csv           — mean ± std per (approach, model)
  deepeval_all_rows.csv           — per-doc rows + derived F1 columns
  deepeval_summary.csv

Derived DeepEval columns added per row BEFORE aggregation
---------------------------------------------------------
  summ_f1   Row-level F1 recomputed from alignment_score (precision) and
            complex_coverage_score (recall). The custom metric already
            stores this as `summ_score` (internally computed as the
            harmonic mean of the two components), so summ_f1 is a
            verification of summ_score. They agree row-by-row except on
            partial/failed rows where summ_score is null but the
            components survived; summ_f1 recovers those.

  geval_f1  Row-level F1 of geval_hallucination (precision) and
            geval_omission (recall). The two GEval metrics share an
            identical four-level rubric structure, so their harmonic mean
            is well defined.

  dag_f1    Row-level F1 of dag_hallucination (precision) and
            dag_coverage (recall).

All three F1s are NaN-aware: NaN if either component is NaN (no F1 from a
missing axis), 0.0 when p + r == 0 (both zero -> F1 defined as 0, not
division by zero). Aggregation then produces _mean, _std, _count per
(approach, model) as for any other metric column.

Numeric columns are summarized with mean and std; an n column counts
non-null rows.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from config import EVAL_OUTPUT_ROOT


# Columns we never want to summarize as if they were metrics. Updated to the
# post-DAG-split schema: dag_score / dag_reason / coverage_score /
# coverage_verdicts are gone; dag_hallucination / dag_coverage are numeric
# and DO get summarised; their reason columns are text and must be excluded.
EXCLUDE_FROM_SUMMARY = {
    "doc_index", "note_id", "approach", "model", "summary_format",
    "n_units", "skip_reason",
    # DeepEval text fields:
    "summ_reason",
    "geval_hall_reason", "geval_omit_reason",
    "dag_hall_reason", "dag_cov_reason",
    "assessment_questions", "alignment_verdicts",
    "complex_assessment_questions", "complex_coverage_verdicts",
}


# Row-level F1 helper, NaN- and zero-safe.
def _f1(p: pd.Series, r: pd.Series) -> pd.Series:
    """Harmonic mean of two [0,1] series, computed elementwise.

    - NaN if either side is NaN (we cannot F1 a missing axis).
    - 0.0 when p == r == 0 (well-defined edge case, not div-by-zero).
    """
    p = pd.to_numeric(p, errors="coerce")
    r = pd.to_numeric(r, errors="coerce")
    denom = p + r
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = 2 * p * r / denom
    both_zero = (denom == 0) & p.notna() & r.notna()
    f1 = f1.where(~both_zero, 0.0)
    return f1


def _add_row_level_f1s(df: pd.DataFrame) -> pd.DataFrame:
    """Compute summ_f1 / geval_f1 / dag_f1 per row, BEFORE aggregation.

    Columns are only created when both source columns exist, so a partial
    schema (e.g. a CSV from before the DAG split) does not break the run.
    """
    if df.empty:
        return df
    df = df.copy()
    if {"alignment_score", "complex_coverage_score"}.issubset(df.columns):
        df["summ_f1"] = _f1(df["alignment_score"], df["complex_coverage_score"])
    if {"geval_hallucination", "geval_omission"}.issubset(df.columns):
        df["geval_f1"] = _f1(df["geval_hallucination"], df["geval_omission"])
    if {"dag_hallucination", "dag_coverage"}.issubset(df.columns):
        df["dag_f1"] = _f1(df["dag_hallucination"], df["dag_coverage"])
    return df


def _collect(source: str) -> pd.DataFrame:
    """Concatenate every aggregate/results.csv under eval_outputs/{source}/.
    For DeepEval, append the row-level F1 columns before returning."""
    root = EVAL_OUTPUT_ROOT / source
    if not root.exists():
        return pd.DataFrame()

    filename = {"selfcheck": "aggregate.csv", "deepeval": "results.csv"}[source]
    frames: List[pd.DataFrame] = []
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir():
            continue
        path = run_dir / filename
        if not path.exists():
            continue
        df = pd.read_csv(path)
        # Defensive: in case approach/model didn't propagate, infer from folder
        if "approach" not in df.columns or "model" not in df.columns:
            approach, _, model = run_dir.name.partition("__")
            df["approach"] = approach
            df["model"] = model
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    if source == "deepeval":
        out = _add_row_level_f1s(out)
    return out


def _summary(df: pd.DataFrame) -> pd.DataFrame:
    """Mean / std / n per (approach, model) for every numeric metric column.

    Because the F1 columns were added at row level by _collect, they appear
    here as summ_f1_mean / summ_f1_std / summ_f1_count etc. alongside the
    other metrics, with no special-casing required.
    """
    if df.empty:
        return df
    metric_cols = [
        c for c in df.columns
        if c not in EXCLUDE_FROM_SUMMARY
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not metric_cols:
        return pd.DataFrame()

    grouped = df.groupby(["approach", "model"], dropna=False)
    agg = grouped[metric_cols].agg(["mean", "std", "count"])
    # Flatten MultiIndex columns: ("summ_f1", "mean") -> "summ_f1_mean"
    agg.columns = [f"{metric}_{stat}" for metric, stat in agg.columns]
    return agg.reset_index()


def _save(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  [Saved] {path} ({len(df)} rows)")


def run(source: str, out_root: Path):
    print(f"\n=== Aggregating {source} ===")
    all_rows = _collect(source)
    if all_rows.empty:
        print(f"  [Skip] no per-run files under {EVAL_OUTPUT_ROOT/source}")
        return
    _save(all_rows, out_root / f"{source}_all_rows.csv")

    # Sanity readout: confirm summ_f1 reproduces the stored summ_score on
    # rows where both exist. Disagreement either means a partial row (good
    # - summ_f1 recovers what summ_score lost) or a code bug worth surfacing.
    if source == "deepeval" and {"summ_f1", "summ_score"}.issubset(all_rows.columns):
        f1 = pd.to_numeric(all_rows["summ_f1"], errors="coerce")
        sc = pd.to_numeric(all_rows["summ_score"], errors="coerce")
        both = f1.notna() & sc.notna()
        close = ((f1 - sc).abs() <= 1e-3) & both
        recovered = f1.notna() & sc.isna()
        print(f"  [Check] summ_f1 reproduces summ_score on {int(close.sum())}/"
              f"{int(both.sum())} rows where both exist; "
              f"summ_f1 recovered {int(recovered.sum())} rows where "
              f"summ_score was null but components survived.")

    summary = _summary(all_rows)
    if summary.empty:
        print("  [Warn] no numeric metric columns found; skipping summary")
        return
    _save(summary, out_root / f"{source}_summary.csv")
    print("\n  Per-(approach, model) summary:")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(summary.to_string(index=False))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["selfcheck", "deepeval", "both"], default="both")
    args = p.parse_args()

    out_root = EVAL_OUTPUT_ROOT / "aggregated"
    sources = ["selfcheck", "deepeval"] if args.source == "both" else [args.source]
    for s in sources:
        run(s, out_root)
    print("\nDone.")


if __name__ == "__main__":
    main()