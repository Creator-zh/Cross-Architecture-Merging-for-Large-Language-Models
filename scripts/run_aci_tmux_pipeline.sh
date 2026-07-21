#!/usr/bin/env bash
# Full ACI test pipeline for tmux: unit tests → download refs → merge → eval → summarize.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY_MERGE="${PY_MERGE:-/home/ziheng_zhao/miniconda3/envs/trans_opt/bin/python}"
PY_EVAL="${PY_EVAL:-/home/ziheng_zhao/miniconda3/envs/lm-eval/bin/python}"
GPUS="${GPUS:-0,1}"
LOG_DIR="${LOG_DIR:-$ROOT/evaluation_results/aci/pipeline_logs}"
mkdir -p "$LOG_DIR"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONUNBUFFERED=1

ts() { date '+%F %T'; }
step() { echo; echo "======== [$(ts)] $* ========"; }

fail() {
  echo "[$(ts)] FAILED: $*" >&2
  exit 1
}

step "0/5 environment"
command -v "$PY_MERGE" >/dev/null || fail "merge python missing: $PY_MERGE"
command -v "$PY_EVAL" >/dev/null || fail "eval python missing: $PY_EVAL"
"$PY_MERGE" -c 'import torch; assert torch.cuda.is_available()' || fail "CUDA unavailable in merge env"
echo "ROOT=$ROOT"
echo "PY_MERGE=$PY_MERGE"
echo "PY_EVAL=$PY_EVAL"
echo "GPUS=$GPUS"
echo "HF_ENDPOINT=$HF_ENDPOINT"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv

step "1/5 unit tests"
"$PY_MERGE" -m unittest discover -s tests -v 2>&1 | tee "$LOG_DIR/01_unittest.log"
echo "[$(ts)] unit tests OK"

step "2/5 download reference 1B models (if missing)"
need_download=0
if [[ ! -d models/Llama-3.2-1B ]]; then need_download=1; fi
if [[ ! -d models/Llama-3.2-1B-Instruct ]]; then need_download=1; fi
if [[ "$need_download" -eq 1 ]]; then
  "$PY_MERGE" scripts/download_models.py \
    --tasks reference_1b_base reference_1b_instruct \
    2>&1 | tee "$LOG_DIR/02_download.log"
else
  echo "reference models already present; skip download" | tee "$LOG_DIR/02_download.log"
fi
[[ -d models/Llama-3.2-1B ]] || fail "missing models/Llama-3.2-1B"
[[ -d models/Llama-3.2-1B-Instruct ]] || fail "missing models/Llama-3.2-1B-Instruct"
echo "[$(ts)] download OK"

step "3/5 ACI merge on GPUs $GPUS"
"$PY_MERGE" scripts/run_aci_tasks.py \
  --tasks medical,thai,malay \
  --gpus "$GPUS" \
  --models-root ./models \
  --results-root ./merge_results/aci \
  2>&1 | tee "$LOG_DIR/03_merge.log"
for task_dir in \
  merge_results/aci/medical_aci_beta0.03/fused_model \
  merge_results/aci/thai_aci_beta0.01/fused_model \
  merge_results/aci/malay_aci_beta0.1/fused_model
do
  [[ -d "$task_dir" ]] || fail "missing fused model: $task_dir"
done
echo "[$(ts)] merge OK"

step "4/5 evaluate target vs aci"
"$PY_EVAL" scripts/evaluate_aci_tasks.py \
  --tasks medical,thai,malay \
  --variants target,aci \
  --gpus "$GPUS" \
  --models-root ./models \
  --results-root ./merge_results/aci \
  --eval-root ./evaluation_results/aci \
  --lm-eval-repo ./lm-evaluation-harness \
  --malay-repo ./evaluation/malay/MalayMMLU \
  2>&1 | tee "$LOG_DIR/04_evaluate.log"
echo "[$(ts)] evaluate OK"

step "5/5 summarize"
"$PY_MERGE" scripts/summarize_merge_results.py \
  --eval-root ./evaluation_results/aci \
  --tasks medical,thai,malay \
  --variants target,aci \
  --output-prefix aci_merge_only \
  2>&1 | tee "$LOG_DIR/05_summarize.log"
echo "[$(ts)] summarize OK"

step "DONE — full ACI pipeline finished"
echo "Logs: $LOG_DIR"
echo "Merged models: ./merge_results/aci/"
echo "Eval results: ./evaluation_results/aci/"
ls -la evaluation_results/aci/aci_merge_only* 2>/dev/null || true
