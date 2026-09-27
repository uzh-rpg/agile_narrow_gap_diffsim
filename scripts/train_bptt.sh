#!/bin/bash
cd "$(dirname "$0")/.."

# Usage: train_bptt.sh [moving|static] [vision|state] [num_envs]
# Defaults: static gap, state observations, 100 environments.
GAP="static"
OBS="state"
NUM_ENVS=100

for arg in "$@"; do
    case "$arg" in
        moving|static) GAP="$arg" ;;
        vision|state)  OBS="$arg" ;;
        [0-9]*)        NUM_ENVS="$arg" ;;
        *)
            echo "Unknown argument: $arg"
            echo "Usage: $0 [moving|static] [vision|state] [num_envs]"
            exit 1
            ;;
    esac
done

EXTRA=""
[ "$GAP" = "moving" ] && EXTRA="$EXTRA --moving"
[ "$OBS" = "vision" ] && EXTRA="$EXTRA --vision"

nohup python experiments/train_bptt_gap_crossing.py -t --num-envs="$NUM_ENVS"$EXTRA > /dev/null 2>&1 &
echo "BPTT training started (PID: $!) [gap=$GAP obs=$OBS num_envs=$NUM_ENVS]"
