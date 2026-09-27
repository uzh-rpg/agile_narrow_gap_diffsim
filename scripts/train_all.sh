#!/bin/bash
# Run all trainings sequentially (one at a time, waiting for each to finish).
set -u
cd "$(dirname "$0")/.."


run() {
    local name="$1"
    shift
    echo "============================================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting: $name"
    echo "  cmd: $*"
    "$@" > /dev/null 2>&1 &
    local pid=$!
    echo "  pid: $pid"
    echo "============================================================"
    if wait "$pid"; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished: $name (pid=$pid OK)"
    else
        local rc=$?
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished: $name (pid=$pid FAILED rc=$rc)"
    fi
}

BPTT="python experiments/train_bptt_gap_crossing.py -t"
PPO="python experiments/train_ppo_gap_crossing.py -t"

for N in 10 100 1000; do
    run "bptt_static_state_envs${N}"  $BPTT --num-envs=$N
    run "bptt_moving_state_envs${N}"  $BPTT --num-envs=$N --moving
done

for N in 10 100 1000; do
    run "ppo_static_state_envs${N}"   $PPO  --num-envs=$N
    run "ppo_moving_state_envs${N}"   $PPO  --num-envs=$N --moving
done

for N in 10 100 1000; do
    run "bptt_static_vision_envs${N}" $BPTT --num-envs=$N --vision
    run "bptt_moving_vision_envs${N}" $BPTT --num-envs=$N --moving --vision
done

echo "All trainings completed."
