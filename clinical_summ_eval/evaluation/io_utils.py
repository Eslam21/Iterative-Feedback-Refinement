"""
Run discovery + standardized loading.

A "run" is one (approach, model) pair, e.g. ('cot_abstractive', 'gemma-3-27b-it').
Each run has its own results folder under RESULTS_ROOT.

discover_runs() walks the filesystem and returns every (approach, model) where
the expected file exists. load_run() reads the CSV and renames the
approach-specific columns to standard names: 'source_text' and 'model_output'.
That way every downstream metric sees the same column names regardless of
whether it's a summary or a filled schema.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional

import pandas as pd

from config import APPROACHES, RESULTS_ROOT, approach_config


@dataclass(frozen=True)
class Run:
    approach: str
    model: str
    path: Path          # path to the input CSV
    fmt: str            # "text" or "json"

    @property
    def slug(self) -> str:
        """Filesystem-safe identifier, e.g. 'cot_abstractive__gemma-3-27b-it'."""
        return f"{self.approach}__{self.model}"


# ─────────────────────────────────────────────────────────────
# Discovery
# ─────────────────────────────────────────────────────────────
def discover_runs(
    results_root: Path = RESULTS_ROOT,
    approaches: Optional[List[str]] = None,
    models: Optional[List[str]] = None,
) -> List[Run]:
    """Walk results_root and return every (approach, model) with its input file present.

    approaches / models, if provided, act as filters.
    Empty model folders are silently skipped.
    """
    runs: List[Run] = []
    results_root = Path(results_root)
    if not results_root.exists():
        raise FileNotFoundError(f"RESULTS_ROOT does not exist: {results_root}")

    target_approaches = approaches or list(APPROACHES.keys())

    for approach in target_approaches:
        cfg = approach_config(approach)
        approach_dir = results_root / approach
        if not approach_dir.exists():
            continue

        for model_dir in sorted(approach_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            if models and model_dir.name not in models:
                continue
            input_csv = model_dir / cfg["file"]
            if not input_csv.exists():
                # empty folder (e.g. iterative_schema/Qwen3-32B-FP8) — skip quietly
                continue
            runs.append(Run(
                approach=approach,
                model=model_dir.name,
                path=input_csv,
                fmt=cfg["format"],
            ))
    return runs


# ─────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────
def load_run(run: Run, limit: Optional[int] = None) -> pd.DataFrame:
    """Load a run's CSV and standardize column names.

    Adds:
      - source_text   (renamed from approach's source_col)
      - model_output  (renamed from approach's output_col)
      - note_id       (if missing, created from row index)
      - approach, model (constant columns for downstream aggregation)
    Drops rows where either source_text or model_output is empty/NaN.
    """
    cfg = approach_config(run.approach)
    df = pd.read_csv(run.path)

    for col in (cfg["source_col"], cfg["output_col"]):
        if col not in df.columns:
            raise ValueError(
                f"{run.path} is missing required column '{col}'. "
                f"Got columns: {list(df.columns)}"
            )

    df = df.rename(columns={
        cfg["source_col"]: "source_text",
        cfg["output_col"]: "model_output",
    })

    if "note_id" not in df.columns:
        df["note_id"] = df.index

    df["approach"] = run.approach
    df["model"] = run.model

    # Drop blank rows so downstream metrics don't crash
    before = len(df)
    df = df[df["source_text"].notna() & df["model_output"].notna()]
    df = df[df["source_text"].astype(str).str.strip().ne("")]
    df = df[df["model_output"].astype(str).str.strip().ne("")]
    dropped = before - len(df)
    if dropped:
        print(f"  [Load] {run.slug}: dropped {dropped} blank rows")

    if limit is not None:
        df = df.head(limit)

    return df.reset_index(drop=True)


def iter_runs(
    approaches: Optional[List[str]] = None,
    models: Optional[List[str]] = None,
) -> Iterator[Run]:
    """Convenience: yield Run objects, applying filters."""
    yield from discover_runs(approaches=approaches, models=models)
