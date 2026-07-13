#!/bin/bash
# Launch SelfCheckGPT across every (approach, model) in the background.
# Works from anywhere: resolves the evaluation dir relative to this script.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$SCRIPT_DIR/../clinical_summ_eval/evaluation"
cd "$EVAL_DIR"

# Models to evaluate
models=(
  "Llama-3.1-8B-Instruct"
  "Llama-3.3-70B-Instruct-FP8-Dynamic"
  "Llama-4-Scout-17B-16E-Instruct-FP8"
  "Qwen3-4B"
  "Qwen3.5-27B-FP8"
  "Qwen3.5-9B"
  "gemma-3-1b-it"
  "gemma-3-27b-it"
)

# Approaches to evaluate
approaches=(
  "base_extractive"
  "cot_abstractive"
  "iterative_schema"
  "oneshot_icl_schema"
  "standard_abstractive"
)

mkdir -p logs

for approach in "${approaches[@]}"; do
  for model in "${models[@]}"; do

    echo "Starting: $approach / $model"

    uv run run_selfcheck.py \
      --approaches "$approach" \
      --models "$model" \
      > "logs/${approach}_${model}.log" 2>&1 &

  done
done

echo "All jobs launched."
echo "Use 'jobs' to see running processes. Logs in $EVAL_DIR/logs/."
