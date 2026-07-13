# Iterative Schema Refinement for Clinical Note Summarisation

Code accompanying the paper. It covers the full pipeline: generating clinical
note summaries (or filled clinical schemas) with the proposed **iterative schema
refinement** method and several baselines, then evaluating them with two
complementary faithfulness/coverage frameworks (SelfCheckGPT and an LLM-judge
DeepEval suite), and aggregating the results into paper-ready tables.

Every generation output is standardised to two columns — `source_text` (the
original note) and `model_output` (a summary **or** a filled JSON schema) — so
all downstream metrics are approach-agnostic.

<p align="center">
  <img
    src="framework.png"
    alt="Framework overview showing the iterative schema refinement pipeline. Clinical notes are embedded and clustered, then each cluster is evaluated, aggregated, and used to refine a shared schema before generating structured clinical documents."
    width="100%"
  />
</p>


## Repository layout

```
.
├── clinical_summ_eval/
│   ├── generation/                          # produce summaries / filled schemas
│   │   ├── sglang_client.py                    SGLang server mgmt + OpenAI-compatible client
│   │   ├── pipeline.py                         ★ proposed iterative schema refinement method
│   │   ├── baseline_bert_centroid.py           extractive baseline (Bio_ClinicalBERT, no LLM)
│   │   ├── baseline_standard_abstractive.py    single-pass LLM summary
│   │   ├── baseline_cot_abstractive.py         two-stage (analyse → summarise)
│   │   └── baseline_oneshot_icl.py             one-shot ICL schema generation + filling
│   │
│   └── evaluation/                          # score + aggregate the generation outputs
│       ├── config.py                           per-approach column mapping, paths, judge config
│       ├── flatten.py                          JSON schema → declarative statements; sentence split
│       ├── io_utils.py                         discover_runs() / load_run() with column normalisation
│       ├── eval_logging.py                     small shared structured logger
│       ├── metrics_selfcheck.py                SelfCheckGPT (BERTScore + MQAG)
│       ├── metrics_deepeval.py                 DeepEval (GEval hallucination/omission + split DAG)
│       ├── custom_metric.py                    NaN-safe CustomSummarizationMetric (precision/recall F1)
│       ├── run_selfcheck.py                    CLI: SelfCheck over all/some runs
│       ├── run_deepeval.py                     CLI: DeepEval over all/some runs
│       ├── aggregate.py                        combine per-run CSVs into summary tables
│       └── compression_ratios.py               char/word compression ratios per approach
│
├── scripts/
│   └── run_all_selfcheck.sh                 launch SelfCheck across every (approach, model)
│
├── pyproject.toml / uv.lock / .python-version   uv project + locked deps
├── .env.example                            template for the judge API key (copy to .env)
└── .gitignore
```

## Installation

This project uses [uv](https://docs.astral.sh/uv/). The pinned Python version
is in `.python-version` (3.12) and all dependencies are locked in `uv.lock`.

```bash
uv sync                                     # create .venv and install from the lockfile
uv run python -m spacy download en_core_web_sm   # SelfCheck sentence splitter
```

Then run any script with `uv run` (examples below use it), or activate the env
with `source .venv/bin/activate`.

> **Dependency note:** `pyproject.toml` now lists a few packages the code imports
> directly (`openai`, `pandas`, `numpy`, `psutil`, `requests`, `torch`,
> `transformers`) that were previously only pulled in transitively. After
> cloning, run `uv lock` once to refresh `uv.lock` against the updated
> `pyproject.toml`, then `uv sync`. (`sglang[all]` pins the CUDA/torch stack, so
> resolve on the target GPU machine.)


### Data

The generation scripts expect a train/test split of clinical notes as CSVs with
a `clinical_notes` column (the iterative and one-shot methods also use a
`cluster` column for batching). Point the scripts at your data via environment
variables (defaults shown):

```bash
export TRAIN_DATA_PATH=datasets/cluster_training_data.csv
export TEST_DATA_PATH=datasets/cluster_testing_data.csv
export MODEL_NAME=Llama-4-Scout-17B-16E-Instruct-FP8
```

Datasets are git-ignored and not distributed here.

## 1. Generation

Each LLM approach talks to a locally served model through SGLang. Start the
server first, then run an approach:

```bash
# start the model server (separate process)
uv run python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --port 30000

cd clinical_summ_eval/generation

# proposed method
uv run pipeline.py

# baselines
uv run baseline_standard_abstractive.py
uv run baseline_cot_abstractive.py
uv run baseline_oneshot_icl.py          # launch server with a larger context length
uv run baseline_bert_centroid.py        # no server needed — loads Bio_ClinicalBERT directly
```

Outputs are written under `results/{approach}/{model}/`:

| Approach | Folder | Output file | `model_output` column |
|---|---|---|---|
| Iterative schema (proposed) | `iterative_schema/<model>/` | `filled_test.csv` | `filled_schema` (JSON) |
| One-shot ICL schema | `oneshot_icl_schema/<model>/` | `filled_test.csv` | `filled_schema` (JSON) |
| Standard abstractive | `standard_abstractive/<model>/` | `summaries.csv` | `summary` (text) |
| CoT abstractive | `cot_abstractive/<model>/` | `summaries.csv` | `summary` (text) |
| BERT-centroid extractive | `base_extractive/bert_centroid/` | `summaries.csv` | `summary` (text) |

`RESULTS_ROOT` in `evaluation/config.py` must point at this `results/`
directory (defaults to `results`; override with the `RESULTS_ROOT` env var).

## 2. Evaluation

Run from inside the evaluation folder (modules import each other flatly):

```bash
cd clinical_summ_eval/evaluation

uv run run_selfcheck.py          # BERTScore + MQAG
uv run run_deepeval.py           # LLM judge (GEval + DAG + custom summarisation)
uv run aggregate.py              # combine per-run CSVs into summary tables
uv run compression_ratios.py     # optional: compression ratios per approach
```

### Filtering

```bash
uv run run_selfcheck.py --approaches cot_abstractive
uv run run_deepeval.py  --models gemma-3-27b-it
uv run run_deepeval.py  --approaches iterative_schema --models gemma-3-27b-it
uv run run_deepeval.py  --limit 3            # tiny debug slice
uv run run_deepeval.py  --overwrite          # redo even if outputs exist
```

A run is skipped when its output CSV already exists. DeepEval checkpoints after
every chunk, so a crash leaves partial progress on disk and resumes next run.

To launch SelfCheck across the whole grid in the background:

```bash
bash scripts/run_all_selfcheck.sh    # runs from anywhere; edit model/approach lists at the top
```

## 3. Outputs

```
eval_outputs/
├── selfcheck/{approach}__{model}/
│   ├── unit.csv            per sentence/field: bertscore, consistent flag
│   └── aggregate.csv       per document: bertscore_avg, mqag_*, …
├── deepeval/{approach}__{model}/
│   └── results.csv         per document: geval_*, dag_*, summ_*
├── compression/
│   ├── per_document.csv
│   └── summary.csv
└── aggregated/
    ├── selfcheck_all_rows.csv
    ├── selfcheck_summary.csv     mean/std/count per (approach, model)
    ├── deepeval_all_rows.csv     + derived summ_f1 / geval_f1 / dag_f1
    └── deepeval_summary.csv
```

## Metric directions

| Metric | Direction | Range |
|---|---|---|
| `bertscore_avg` | lower = more consistent | ~0–1 |
| `pct_consistent` | higher = more consistent | 0–100 |
| `mqag_kl_div` / `hellinger` / `total_var` / `counting` | lower = more faithful | divergences |
| `geval_hallucination` | higher = better (fewer hallucinations) | 0–1 |
| `geval_omission` | higher = better (fewer omissions) | 0–1 |
| `dag_hallucination` (precision) / `dag_coverage` (recall) | higher = more faithful | 0–1 |
| `summ_score` / `alignment_score` / `complex_coverage_score` | higher = better | 0–1 |

`aggregate.py` also derives row-level **F1** columns before summarising:
`summ_f1` (alignment × complex-coverage), `geval_f1` (hallucination × omission),
and `dag_f1` (dag precision × recall). All are NaN-aware.

## Adding a new approach

Add an entry to `APPROACHES` in `evaluation/config.py`:

```python
"my_new_approach": {
    "file":       "outputs.csv",
    "source_col": "clinical_notes",
    "output_col": "generated_text",
    "format":     "text",   # or "json" for a filled schema
},
```

Write files to `results/my_new_approach/{model}/outputs.csv` and rerun the
evaluation. New models are discovered automatically — just create the folder.
