#!/usr/bin/env bash
# Thai/Malay overnight: pre-SFT eval -> SFT -> post-SFT eval (GPU 6 by default).
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

GPU="${GPU:-6}"
PRE_EVAL_ROOT="${PRE_EVAL_ROOT:-$REPO/evaluation_results/dfop_pre_sft}"
POST_EVAL_ROOT="${POST_EVAL_ROOT:-$REPO/evaluation_results/dfop_post_sft}"
FUSION_ROOT="${FUSION_ROOT:-$REPO/transport_results/dfop}"
SFT_ROOT="${SFT_ROOT:-$REPO/sft_results/dfop}"
STAGE_DIR="$REPO/evaluation_results/dfop_thai_malay_pipeline"
LOG="$STAGE_DIR/overnight.log"
SUMMARY="$STAGE_DIR/summary.txt"

mkdir -p "$STAGE_DIR" "$PRE_EVAL_ROOT" "$POST_EVAL_ROOT" "$SFT_ROOT"

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_XET=1

# shellcheck disable=SC1091
source "$HOME/miniconda3/etc/profile.d/conda.sh"

ts() { date '+%F %T'; }
log() { echo "[$(ts)] $*" | tee -a "$LOG"; }
mark_done() { echo "done $(ts)" >"$STAGE_DIR/$1.done"; }
is_done() { [[ -f "$STAGE_DIR/$1.done" ]]; }

has_lm_eval_results() {
  local dir="$1"
  [[ -d "$dir" ]] && find "$dir" -name 'results_*.json' -print -quit | grep -q .
}

has_malay_metrics() {
  local dir="$1"
  [[ -f "$dir/metrics.json" ]] || find "$dir" -name 'MalayMMLU_result_*_True_0shot.csv' -print -quit | grep -q .
}

run_eval_job() {
  local task="$1" variant="$2" eval_root="$3" stage="$4"
  local out="$eval_root/$task/$variant/primary"

  if is_done "$stage"; then
    log "[skip] $stage already marked done"
    return 0
  fi
  if [[ "$task" == "malay" ]]; then
    if has_malay_metrics "$out"; then
      log "[skip] $task:$variant metrics already present"
      mark_done "$stage"
      return 0
    fi
  else
    if has_lm_eval_results "$out"; then
      log "[skip] $task:$variant results already present"
      mark_done "$stage"
      return 0
    fi
  fi

  # Clear incomplete prior outputs so lm-eval does not confuse resume.
  if [[ -d "$out" ]] && ! has_lm_eval_results "$out" && ! has_malay_metrics "$out"; then
    rm -rf "$out"
  fi

  log "[start] eval $task:$variant -> $eval_root (gpu=$GPU)"
  conda activate lm-eval
  set +e
  python scripts/evaluate_dfop_tasks.py \
    --tasks "$task" \
    --variants "$variant" \
    --gpus "$GPU" \
    --results-root "$FUSION_ROOT" \
    --sft-results-root "$SFT_ROOT" \
    --eval-root "$eval_root" \
    --mode full \
    --track universal \
    --rank 128 \
    --sft-train-mode full \
    --sft-profile declared \
    --scope primary \
    --malay-repo "$REPO/evaluation/malay/MalayMMLU" \
    2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  set -e
  if [[ $rc -ne 0 ]]; then
    log "[fail] eval $task:$variant exit=$rc"
    return "$rc"
  fi
  mark_done "$stage"
  log "[ok] eval $task:$variant"
  return 0
}

run_sft_tasks() {
  local stage="sft_thai_malay"
  if is_done "$stage"; then
    log "[skip] $stage already marked done"
    return 0
  fi
  # Individual markers from launcher.
  local thai_done="$SFT_ROOT/thai_full_universal_r128/full_declared/_DFOP_SFT_DONE"
  local malay_done="$SFT_ROOT/malay_full_universal_r128/full_declared/_DFOP_SFT_DONE"
  if [[ -f "$thai_done" && -f "$malay_done" ]]; then
    log "[skip] thai+malay SFT markers present"
    mark_done "$stage"
    return 0
  fi

  log "[start] SFT thai,malay (gpu=$GPU)"
  conda activate trans_opt
  set +e
  # One GPU: jobs run sequentially inside the launcher (one process at a time
  # only if we pass one task; with two tasks it launches both concurrently).
  # Launch sequentially to stay on a single free GPU.
  python scripts/run_dfop_sft_tasks.py \
    --tasks thai \
    --gpus "$GPU" \
    --fusion-results-root "$FUSION_ROOT" \
    --sft-results-root "$SFT_ROOT" \
    --mode full \
    --track universal \
    --rank 128 \
    --train-mode full \
    --profile declared \
    --thai-dataset-path "$REPO/data/fineweb_thai" \
    --resume \
    2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  if [[ $rc -ne 0 ]]; then
    log "[fail] thai SFT exit=$rc"
    return "$rc"
  fi

  python scripts/run_dfop_sft_tasks.py \
    --tasks malay \
    --gpus "$GPU" \
    --fusion-results-root "$FUSION_ROOT" \
    --sft-results-root "$SFT_ROOT" \
    --mode full \
    --track universal \
    --rank 128 \
    --train-mode full \
    --profile declared \
    --malay-dataset-path "$REPO/data/Malaysian-SFT" \
    --resume \
    2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  set -e
  if [[ $rc -ne 0 ]]; then
    log "[fail] malay SFT exit=$rc"
    return "$rc"
  fi
  mark_done "$stage"
  log "[ok] SFT thai+malay"
  return 0
}

write_summary() {
  conda activate lm-eval
  python - <<'PY' | tee "$SUMMARY" | tee -a "$LOG"
import json
from pathlib import Path

root = Path(".")
pre = root / "evaluation_results" / "dfop_pre_sft"
post = root / "evaluation_results" / "dfop_post_sft"

def pick_metric(task_blob):
    for key in ("acc,none", "exact_match,none", "exact_match,flexible-extract", "f1,none"):
        if key in task_blob and isinstance(task_blob[key], (int, float)):
            return key, float(task_blob[key])
    for mk, mv in task_blob.items():
        if isinstance(mv, (int, float)) and "stderr" not in mk and mk.endswith(",none"):
            return mk, float(mv)
    return None, None

def load_lm(path):
    files = sorted(path.rglob("results_*.json"))
    if not files:
        return None
    data = json.loads(files[-1].read_text())
    out = {}
    for name, blob in data.get("results", {}).items():
        mk, mv = pick_metric(blob)
        if mk is not None:
            out[name] = mv
    return out

def load_malay(path):
    metrics = path / "metrics.json"
    if metrics.is_file():
        data = json.loads(metrics.read_text())
        return {"MalayMMLU": float(data["results"]["MalayMMLU"]["acc,none"])}
    return None

print("=== DFOP Thai/Malay overnight summary ===")
for task in ("thai", "malay"):
    print(f"\n## {task}")
    rows = {}
    for variant, base in (
        ("target", pre),
        ("dfop_full", pre),
        ("dfop_sft", post),
    ):
        d = base / task / variant / "primary"
        scores = load_malay(d) if task == "malay" else load_lm(d)
        rows[variant] = scores
        if scores is None:
            print(f"  {variant}: MISSING")
        else:
            vals = list(scores.values())
            mean = sum(vals) / len(vals)
            detail = ", ".join(f"{k}={v:.4f}" for k, v in sorted(scores.items()))
            print(f"  {variant}: mean={mean:.4f} | {detail}")
print("\nDone.")
PY
}

log "===== overnight pipeline start gpu=$GPU ====="
log "REPO=$REPO"

set -e
# 1) Pre-SFT eval
run_eval_job thai target "$PRE_EVAL_ROOT" pre_thai_target
run_eval_job thai dfop_full "$PRE_EVAL_ROOT" pre_thai_dfop_full
run_eval_job malay target "$PRE_EVAL_ROOT" pre_malay_target
run_eval_job malay dfop_full "$PRE_EVAL_ROOT" pre_malay_dfop_full

# 2) SFT
run_sft_tasks

# 3) Post-SFT eval
run_eval_job thai dfop_sft "$POST_EVAL_ROOT" post_thai_dfop_sft
run_eval_job malay dfop_sft "$POST_EVAL_ROOT" post_malay_dfop_sft

# 4) Summary
write_summary
mark_done "pipeline_all"
log "===== overnight pipeline COMPLETE ====="
log "Summary: $SUMMARY"
log "Log: $LOG"
