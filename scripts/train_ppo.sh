#!/bin/bash
cd "$(dirname "$0")/.."

# Usage: train_ppo.sh [moving|static] [num_envs]
# Defaults: static gap, 100 environments.
# Note: PPO only supports state-based observations (vision is not supported).
GAP="static"
NUM_ENVS=100

for arg in "$@"; do
    case "$arg" in
        moving|static) GAP="$arg" ;;
        vision)
            echo "Error: PPO does not support vision observations (use train_bptt.sh for vision)."
            exit 1
            ;;
        state)         ;;
        [0-9]*)        NUM_ENVS="$arg" ;;
        *)
            echo "Unknown argument: $arg"
            echo "Usage: $0 [moving|static] [num_envs]"
            exit 1
            ;;
    esac
done

EXTRA=""
[ "$GAP" = "moving" ] && EXTRA="$EXTRA --moving"

nohup python experiments/train_ppo_gap_crossing.py -t --num-envs="$NUM_ENVS"$EXTRA > /dev/null 2>&1 &
echo "PPO training started (PID: $!) [gap=$GAP obs=state num_envs=$NUM_ENVS]"
