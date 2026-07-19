#!/usr/bin/env bash
# Attn-only DFOP: for each top-k in {1,2}, run fusion -> pre-SFT eval -> SFT -> post-SFT eval.
# Designed for a single free GPU (default GPU=6), sequential across tasks.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

GPU="${GPU:-6}"
TOPS="${TOPS:-1,2}"
TASKS="${TASKS:-medical,thai,malay}"
RANK="${RANK:-128}"
TRACK="${TRACK:-universal}"
MODE="attn"
FUSION_ROOT="${FUSION_ROOT:-$REPO/transport_results/dfop}"
SFT_ROOT="${SFT_ROOT:-$REPO/sft_results/dfop}"
STAGE_ROOT="${STAGE_ROOT:-$REPO/evaluation_results/dfop_attn_pipeline}"

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_XET=1
export MALAYSIAN_SFT_LOCAL_DATASET="${MALAYSIAN_SFT_LOCAL_DATASET:-$REPO/data/Malaysian-SFT}"

# shellcheck disable=SC1091
source "$HOME/miniconda3/etc/profile.d/conda.sh"

mkdir -p "$STAGE_ROOT"

ts() { date '+%F %T'; }
log() {
  local top="$1"; shift
  local file="$STAGE_ROOT/top${top}/pipeline.log"
  mkdir -p "$(dirname "$file")"
  echo "[$(ts)] $*" | tee -a "$file"
}
mark_done() { echo "done $(ts)" >"$STAGE_ROOT/top$1/$2.done"; }
is_done() { [[ -f "$STAGE_ROOT/top$1/$2.done" ]]; }

has_lm_eval_results() {
  local dir="$1"
  [[ -d "$dir" ]] && find "$dir" -name 'results_*.json' -print -quit | grep -q .
}

has_malay_metrics() {
  local dir="$1"
  [[ -f "$dir/metrics.json" ]] || find "$dir" -name 'MalayMMLU_result_*_True_0shot.csv' -print -quit | grep -q .
}

run_fusion_task() {
  local top="$1" task="$2"
  local stage="fusion_${task}"
  local out="$FUSION_ROOT/${task}_${MODE}_${TRACK}_r${RANK}_top${top}_beta0.05/fused_model"
  if is_done "$top" "$stage"; then
    log "$top" "[skip] $stage"
    return 0
  fi
  if [[ -f "$out/model.safetensors" || -f "$out/pytorch_model.bin" ]]; then
    log "$top" "[skip] $stage fused model present"
    mark_done "$top" "$stage"
    return 0
  fi
  log "$top" "[start] fusion task=$task top=$top gpu=$GPU"
  conda activate trans_opt
  set +e
  python scripts/run_dfop_tasks.py \
    --tasks "$task" \
    --gpus "$GPU" \
    --models-root "$REPO/models" \
    --results-root "$FUSION_ROOT" \
    --mode "$MODE" \
    --track "$TRACK" \
    --rank "$RANK" \
    --top-source-layers "$top" \
    --resume \
    2>&1 | tee -a "$STAGE_ROOT/top${top}/pipeline.log"
  local rc=${PIPESTATUS[0]}
  set -e
  if [[ $rc -ne 0 ]]; then
    log "$top" "[fail] fusion $task exit=$rc"
    return "$rc"
  fi
  mark_done "$top" "$stage"
  log "$top" "[ok] fusion $task"
}

run_eval_job() {
  local top="$1" task="$2" variant="$3" eval_root="$4" stage="$5"
  local out="$eval_root/$task/$variant/primary"
  if is_done "$top" "$stage"; then
    log "$top" "[skip] $stage"
    return 0
  fi
  if [[ "$task" == "malay" ]]; then
    if has_malay_metrics "$out"; then
      log "$top" "[skip] $task:$variant metrics present"
      mark_done "$top" "$stage"
      return 0
    fi
  else
    if has_lm_eval_results "$out"; then
      log "$top" "[skip] $task:$variant results present"
      mark_done "$top" "$stage"
      return 0
    fi
  fi
  if [[ -d "$out" ]] && ! has_lm_eval_results "$out" && ! has_malay_metrics "$out"; then
    rm -rf "$out"
  fi
  log "$top" "[start] eval $task:$variant -> $eval_root"
  conda activate lm-eval
  set +e
  python scripts/evaluate_dfop_tasks.py \
    --tasks "$task" \
    --variants "$variant" \
    --gpus "$GPU" \
    --results-root "$FUSION_ROOT" \
    --sft-results-root "$SFT_ROOT" \
    --eval-root "$eval_root" \
    --mode "$MODE" \
    --track "$TRACK" \
    --rank "$RANK" \
    --top-source-layers "$top" \
    --sft-train-mode full \
    --sft-profile declared \
    --scope primary \
    --malay-repo "$REPO/evaluation/malay/MalayMMLU" \
    2>&1 | tee -a "$STAGE_ROOT/top${top}/pipeline.log"
  local rc=${PIPESTATUS[0]}
  set -e
  if [[ $rc -ne 0 ]]; then
    log "$top" "[fail] eval $task:$variant exit=$rc"
    return "$rc"
  fi
  mark_done "$top" "$stage"
  log "$top" "[ok] eval $task:$variant"
}

run_sft_task() {
  local top="$1" task="$2"
  local stage="sft_${task}"
  local out="$SFT_ROOT/${task}_${MODE}_${TRACK}_r${RANK}_top${top}/full_declared"
  if is_done "$top" "$stage"; then
    log "$top" "[skip] $stage"
    return 0
  fi
  if [[ -f "$out/_DFOP_SFT_DONE" ]]; then
    log "$top" "[skip] $stage marker present"
    mark_done "$top" "$stage"
    return 0
  fi
  log "$top" "[start] SFT task=$task top=$top"
  conda activate trans_opt
  set +e
  local extra=()
  if [[ "$task" == "thai" ]]; then
    extra+=(--thai-dataset-path "$REPO/data/fineweb_thai")
  fi
  if [[ "$task" == "malay" ]]; then
    extra+=(--malay-dataset-path "$REPO/data/Malaysian-SFT")
  fi
  python scripts/run_dfop_sft_tasks.py \
    --tasks "$task" \
    --gpus "$GPU" \
    --fusion-results-root "$FUSION_ROOT" \
    --sft-results-root "$SFT_ROOT" \
    --mode "$MODE" \
    --track "$TRACK" \
    --rank "$RANK" \
    --top-source-layers "$top" \
    --train-mode full \
    --profile declared \
    --resume \
    "${extra[@]}" \
    2>&1 | tee -a "$STAGE_ROOT/top${top}/pipeline.log"
  local rc=${PIPESTATUS[0]}
  set -e
  if [[ $rc -ne 0 ]]; then
    log "$top" "[fail] SFT $task exit=$rc"
    return "$rc"
  fi
  mark_done "$top" "$stage"
  log "$top" "[ok] SFT $task"
}

write_summary() {
  local top="$1"
  local pre="$STAGE_ROOT/top${top}/pre_sft"
  local post="$STAGE_ROOT/top${top}/post_sft"
  local summary="$STAGE_ROOT/top${top}/summary.txt"
  conda activate lm-eval
  TOP="$top" PRE="$pre" POST="$post" SUMMARY="$summary" python - <<'PY' | tee "$SUMMARY" | tee -a "$STAGE_ROOT/top${TOP}/pipeline.log"
import csv, json, os
from pathlib import Path

top = os.environ["TOP"]
pre = Path(os.environ["PRE"])
post = Path(os.environ["POST"])

def pick(blob):
    for key in ("acc,none", "exact_match,none", "f1,none"):
        if key in blob and isinstance(blob[key], (int, float)):
            return key, float(blob[key])
    return None, None

def load_lm(path: Path):
    files = sorted(path.rglob("results_*.json"))
    if not files:
        return None
    data = json.loads(files[-1].read_text())
    out = {}
    for name, blob in data.get("results", {}).items():
        mk, mv = pick(blob)
        if mk is not None:
            out[name] = mv
    return out

def malay_acc(path: Path):
    metrics = path / "metrics.json"
    if metrics.is_file():
        data = json.loads(metrics.read_text())
        return {"MalayMMLU": float(data["results"]["MalayMMLU"]["acc,none"])}
    csvs = sorted(path.glob("MalayMMLU_result_*_True_0shot.csv"))
    if len(csvs) != 1:
        return None
    correct = total = 0
    with csvs[0].open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            total += 1
            g = str(row["golds"]).strip()
            p = str(row["preds"]).strip().upper()[:1]
            g2 = chr(65 + int(float(g))) if g.replace(".", "", 1).isdigit() else g.upper()[:1]
            correct += g2 == p
    return {"MalayMMLU": correct / total if total else 0.0}

print(f"=== DFOP attn-only top-{top} summary ===")
for task in ("medical", "thai", "malay"):
    print(f"\n[{task}]")
    for variant, base in (("target", pre), ("dfop_attn", pre), ("dfop_sft", post)):
        d = base / task / variant / "primary"
        scores = malay_acc(d) if task == "malay" else load_lm(d)
        if scores is None:
            print(f"  {variant}: MISSING")
            continue
        vals = list(scores.values())
        mean = sum(vals) / len(vals)
        detail = ", ".join(f"{k}={v:.4f}" for k, v in sorted(scores.items()))
        print(f"  {variant}: mean={mean:.4f} | {detail}")
print("\nDone.")
PY
}

IFS=',' read -r -a TOP_LIST <<<"$TOPS"
IFS=',' read -r -a TASK_LIST <<<"$TASKS"

log_root() { mkdir -p "$STAGE_ROOT/top$1"; }

set -e
for top in "${TOP_LIST[@]}"; do
  log_root "$top"
  PRE_EVAL_ROOT="$STAGE_ROOT/top${top}/pre_sft"
  POST_EVAL_ROOT="$STAGE_ROOT/top${top}/post_sft"
  mkdir -p "$PRE_EVAL_ROOT" "$POST_EVAL_ROOT"
  log "$top" "===== attn pipeline start gpu=$GPU top=$top ====="

  for task in "${TASK_LIST[@]}"; do
    run_fusion_task "$top" "$task"
  done

  # Reuse shared target baselines when available; otherwise evaluate target once per top tree.
  for task in "${TASK_LIST[@]}"; do
    run_eval_job "$top" "$task" "target" "$PRE_EVAL_ROOT" "pre_${task}_target"
    run_eval_job "$top" "$task" "dfop_attn" "$PRE_EVAL_ROOT" "pre_${task}_dfop_attn"
  done

  for task in "${TASK_LIST[@]}"; do
    run_sft_task "$top" "$task"
  done

  for task in "${TASK_LIST[@]}"; do
    run_eval_job "$top" "$task" "dfop_sft" "$POST_EVAL_ROOT" "post_${task}_dfop_sft"
  done

  write_summary "$top"
  mark_done "$top" "pipeline_all"
  log "$top" "===== attn pipeline COMPLETE top=$top ====="
done

echo "[$(ts)] all tops finished: $TOPS"
