"""
DeepEval runner.

  python run_deepeval.py                                  # all runs under results/
  python run_deepeval.py --approaches cot_abstractive
  python run_deepeval.py --models gemma-3-27b-it
  python run_deepeval.py --limit 5                        # 5 rows per run (debug)
  python run_deepeval.py --overwrite                      # rerun even if outputs exist
  python run_deepeval.py --judge-model gpt-4o-mini        # override judge
  python run_deepeval.py --chunk-size 5                   # smaller chunks → less timeout exposure

Outputs:
  eval_outputs/deepeval/{approach}__{model}/results.csv

Each chunk is appended to results.csv as it completes, so a crash leaves
partial progress on disk and the next run resumes from where it stopped.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from config import EVAL_OUTPUT_ROOT, JUDGE_MODEL
from io_utils import discover_runs, load_run
from metrics_deepeval import score_run


def _parse_args():
    p = argparse.ArgumentParser(description="Run DeepEval metrics over all discovered runs.")
    p.add_argument("--approaches", nargs="*", default=None)
    p.add_argument("--models", nargs="*", default=None)
    p.add_argument("--limit", type=int, default=None,
                   help="Cap rows per run (debug). Default: all rows.")
    p.add_argument("--overwrite", action="store_true",
                   help="Delete any existing results.csv for the run before starting.")
    p.add_argument("--judge-model", default=JUDGE_MODEL,
                   help=f"Judge model name (default: {JUDGE_MODEL})")
    p.add_argument("--chunk-size", type=int, default=11,
                   help="Test cases per evaluate() call (default 10). Smaller = "
                        "less per-call timeout risk, more overhead.")
    p.add_argument("--max-concurrent", type=int, default=5,
                   help="Max test cases hitting the judge API concurrently (default 5). "
                        "Lower if you see timeouts; raise if your judge has high rate limits.")
    return p.parse_args()


def _out_dir(run_slug: str) -> Path:
    d = EVAL_OUTPUT_ROOT / "deepeval" / run_slug
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
    print(f"Judge model:    {args.judge_model}")
    print(f"Chunk size:     {args.chunk_size}")
    print(f"Max concurrent: {args.max_concurrent}\n")

    for run in runs:
        out_dir = _out_dir(run.slug)
        out_path = out_dir / "results.csv"

        if out_path.exists() and args.overwrite:
            out_path.unlink()
            print(f"[Reset] removed {out_path}")

        print(f"\n=== {run.slug} ===")
        df = load_run(run, limit=args.limit)
        if df.empty:
            print(f"  [Skip] {run.slug}: no rows after loading")
            continue

        # Pass checkpoint path → score_run appends after every chunk and resumes
        # from the existing CSV if present.
        results = score_run(
            df,
            fmt=run.fmt,
            judge_model=args.judge_model,
            chunk_size=args.chunk_size,
            max_concurrent=args.max_concurrent,
            checkpoint_path=str(out_path),
        )
        if results.empty:
            print(f"  [Skip] {run.slug}: no results produced")
            continue
        print(f"  [Done] {out_path} ({len(results)} rows total)")

    print("\nDone.")


if __name__ == "__main__":
    main()