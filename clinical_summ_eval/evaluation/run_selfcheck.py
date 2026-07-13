"""
SelfCheckGPT runner.

  python run_selfcheck.py                              # all runs under results/
  python run_selfcheck.py --approaches cot_abstractive
  python run_selfcheck.py --models gemma-3-27b-it
  python run_selfcheck.py --approaches iterative_schema --models gemma-3-27b-it
  python run_selfcheck.py --limit 5                    # 5 rows per run (debug)
  python run_selfcheck.py --overwrite                  # rerun even if outputs exist

Outputs:
  eval_outputs/selfcheck/{approach}__{model}/unit.csv
  eval_outputs/selfcheck/{approach}__{model}/aggregate.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from config import EVAL_OUTPUT_ROOT
from io_utils import discover_runs, load_run
from metrics_selfcheck import SelfCheckModels, score_run


def _parse_args():
    p = argparse.ArgumentParser(description="Run SelfCheckGPT eval over all discovered runs.")
    p.add_argument("--approaches", nargs="*", default=None,
                   help="Filter to specific approaches (e.g. cot_abstractive iterative_schema)")
    p.add_argument("--models", nargs="*", default=None,
                   help="Filter to specific model names (e.g. gemma-3-27b-it)")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap rows per run (debug). Default: all rows.")
    p.add_argument("--overwrite", action="store_true",
                   help="Rerun even if aggregate.csv already exists for a run.")
    p.add_argument("--device", default=None,
                   help="Force device ('cuda', 'cpu'). Default: auto.")
    return p.parse_args()


def _out_dir(run_slug: str) -> Path:
    d = EVAL_OUTPUT_ROOT / "selfcheck" / run_slug
    d.mkdir(parents=True, exist_ok=True)
    return d


def main():
    args = _parse_args()
    runs = discover_runs(approaches=args.approaches, models=args.models)
    if not runs:
        print("No runs discovered. Check RESULTS_ROOT and filters.")
        sys.exit(1)

    print(f"Discovered {len(runs)} run(s):")
    for r in runs:
        print(f"  - {r.approach}/{r.model}  (fmt={r.fmt})")
    print()

    # Lazy: only build models if at least one run will actually run
    models: Optional[SelfCheckModels] = None

    for run in runs:
        out_dir = _out_dir(run.slug)
        agg_path = out_dir / "aggregate.csv"
        unit_path = out_dir / "unit.csv"
        if agg_path.exists() and not args.overwrite:
            print(f"[Skip] {run.slug}: {agg_path} exists (use --overwrite to redo)")
            continue

        print(f"\n=== {run.slug} ===")
        df = load_run(run, limit=args.limit)
        if df.empty:
            print(f"  [Skip] {run.slug}: no rows after loading")
            continue

        if models is None:
            print("  Initializing SelfCheck models (BERTScore + MQAG)...")
            models = SelfCheckModels(device=args.device)
            print(f"  Device: {models.device}")

        unit_df, agg_df = score_run(df, fmt=run.fmt, models=models)
        unit_df.to_csv(unit_path, index=False)
        agg_df.to_csv(agg_path, index=False)
        print(f"  [Saved] {unit_path} ({len(unit_df)} units)")
        print(f"  [Saved] {agg_path} ({len(agg_df)} docs)")

    print("\nDone.")


if __name__ == "__main__":
    main()
