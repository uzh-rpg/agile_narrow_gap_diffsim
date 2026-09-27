#!/bin/bash
# Evaluate BPTT checkpoints across the 4 task configurations and generate plots.
#
# === EDIT HERE: paste the checkpoint paths to evaluate ======================
# Leave a path empty ("") to skip that task.
# Accepts either a checkpoints/ directory or a specific checkpoint_<N> dir.

STATIC_STATE="runs/bptt_static_state_envs100/checkpoints/checkpoint_20000"
MOVING_STATE="runs/bptt_moving_state_envs100/checkpoints/checkpoint_20000"
STATIC_VISION="runs/bptt_static_vision_envs100/checkpoints/checkpoint_20000"
MOVING_VISION="runs/bptt_moving_vision_envs100/checkpoints/checkpoint_19000"

ROLLOUTS=1000
# ===========================================================================
#
# Defaults above can also be overridden via env vars or CLI flags, e.g.:
#   ./scripts/evaluate_all.sh --static-state runs/.../checkpoints --rollouts 500
#   STATIC_STATE=runs/.../checkpoints ./scripts/evaluate_all.sh
#
# Stdout/stderr of every evaluation are also appended to evaluate_all.out
# in the repo root.

set -u
cd "$(dirname "$0")/.."

LOG_FILE="evaluate_all.out"
# Tee everything (this script + child processes) to the log file.
exec > >(tee -a "$LOG_FILE") 2>&1

echo "============================================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] evaluate_all.sh started"
echo "  log: $LOG_FILE"
echo "============================================================"

# Env-var overrides (only if explicitly exported)
STATIC_STATE="${STATIC_STATE_OVERRIDE:-$STATIC_STATE}"
MOVING_STATE="${MOVING_STATE_OVERRIDE:-$MOVING_STATE}"
STATIC_VISION="${STATIC_VISION_OVERRIDE:-$STATIC_VISION}"
MOVING_VISION="${MOVING_VISION_OVERRIDE:-$MOVING_VISION}"
ROLLOUTS="${ROLLOUTS_OVERRIDE:-$ROLLOUTS}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --static-state)  STATIC_STATE="$2";  shift 2 ;;
        --moving-state)  MOVING_STATE="$2";  shift 2 ;;
        --static-vision) STATIC_VISION="$2"; shift 2 ;;
        --moving-vision) MOVING_VISION="$2"; shift 2 ;;
        --rollouts)      ROLLOUTS="$2";      shift 2 ;;
        -h|--help)
            sed -n '2,21p' "$0"; exit 0 ;;
        *)
            echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

evaluate() {
    local name="$1"
    local ckpt="$2"
    shift 2
    local extra_flags=("$@")

    if [[ -z "$ckpt" || "$ckpt" == runs/\<* ]]; then
        echo "[skip] $name (no checkpoint configured)"
        return
    fi

    echo "============================================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Evaluating: bptt $name"
    echo "  checkpoint: $ckpt"
    echo "  rollouts:   $ROLLOUTS"
    echo "============================================================"
    python experiments/evaluate.py \
        --algo bptt \
        --checkpoint "$ckpt" \
        --rollouts "$ROLLOUTS" \
        "${extra_flags[@]}"
    echo
}

evaluate "static_state"  "$STATIC_STATE"                    --no-heatmap
evaluate "moving_state"  "$MOVING_STATE"  --moving          --no-heatmap
evaluate "static_vision" "$STATIC_VISION"          --vision
evaluate "moving_vision" "$MOVING_VISION" --moving --vision

echo "All evaluations completed."
