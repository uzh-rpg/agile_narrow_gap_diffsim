#!/usr/bin/env python3
"""
Evaluate a trained checkpoint on N rollouts and generate the paper plots.

Same CLI as `scripts/generate_plots.py`, but in addition prints the same
gap-crossing statistics that are logged during training (via
`compute_and_print_gap_crossing_statistics`).

Examples
--------
    python experiments/evaluate.py --algo bptt --checkpoint runs/.../checkpoints
    python experiments/evaluate.py --algo ppo --moving --vision \\
        --checkpoint runs/.../checkpoints --rollouts 200
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import jax

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _import_train_module(algo: str, moving: bool, vision: bool):
    if algo == "bptt":
        import experiments.train_bptt_gap_crossing as tm
    elif algo == "ppo":
        import experiments.train_ppo_gap_crossing as tm
    else:
        raise ValueError(f"Unknown algo: {algo}")
    tm.USE_MOVING_GAP = moving
    tm.USE_VISION = vision
    tm._import_gap_env()
    return tm


def _build_policy(algo: str, train_state):
    if algo == "bptt":
        def policy(obs, key=None):
            return train_state.apply_fn(train_state.params, obs)
        return policy

    def policy(obs, key):
        pi = train_state.apply_fn(train_state.params, obs).pi
        return pi.sample(seed=key)
    return policy


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, type=str,
                        help="Path to a checkpoint or to a checkpoints/ directory.")
    parser.add_argument("--algo", choices=["bptt", "ppo"], required=True)
    parser.add_argument("--moving", action="store_true")
    parser.add_argument("--vision", action="store_true")
    parser.add_argument("--rollouts", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Where to save the plots. Defaults to <run_dir>/plots/.")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--no-plots", action="store_true",
                        help="Only compute and print statistics, skip plot generation.")
    parser.add_argument("--no-heatmap", action="store_true",
                        help="Skip the bbox heatmap plot (still generates the 2D gap-crossing plot).")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint).resolve()
    if ckpt_path.name.startswith("checkpoint_"):
        run_dir = ckpt_path.parent.parent
    elif ckpt_path.name == "checkpoints":
        run_dir = ckpt_path.parent
    else:
        run_dir = ckpt_path
    out_dir = Path(args.output_dir) if args.output_dir else (run_dir / "plots")
    if not args.no_plots:
        out_dir.mkdir(parents=True, exist_ok=True)

    tm = _import_train_module(args.algo, args.moving, args.vision)

    drone_path = tm.FLIGHTNING_PATH + "/objects/quadrotor_files/example_quad.yaml"
    dyn_config = {
        "high_fidelity_forward": False,
        "forward_residual": False,
        "backprop_residual": False,
    }

    key = jax.random.PRNGKey(args.seed)
    key_init, key_eval = jax.random.split(key, 2)

    print("=" * 60)
    print(f"EVALUATION  algo={args.algo}  moving={args.moving}  vision={args.vision}")
    print(f"checkpoint:  {ckpt_path}")
    print(f"rollouts:    {args.rollouts}   seed: {args.seed}")
    print(f"env class:   {tm.GapEnv.__name__}")
    print("=" * 60)

    _env_template, train_state = tm.create_train_state(
        drone_path, key_init, num_epochs=1, dyn_config=dyn_config
    )
    train_state = tm.load_model(train_state, str(ckpt_path))

    env_eval = tm.create_env(drone_path, training=False, dyn_config=dyn_config)
    policy = _build_policy(args.algo, train_state)

    print(f"\n[eval] running {args.rollouts} rollouts...")
    t0 = time.time()
    transitions = tm.get_rollouts(env_eval, policy,
                                  num_rollouts=args.rollouts, key=key_eval)

    # Reuse the exact same statistics function used during training/load.
    stats = tm.compute_and_print_gap_crossing_statistics(transitions, env_eval, t0)

    if args.no_plots:
        print(f"\n[done] (plots skipped)")
        return

    from flightning.envs.visualization import (
        MultiRolloutVisualizer,
        SingleRolloutVisualizer,
    )

    suffix = f"{args.algo}_{tm.get_config_suffix()}"
    multi = MultiRolloutVisualizer(transitions, env_eval)
    multi.plot_gap_crossing_2d(
        save_path=str(out_dir / f"{suffix}_gap_crossing_2d.png"), show=args.show
    )

    if not args.no_heatmap:
        single = SingleRolloutVisualizer(env_eval, transitions.state, traj_idx=0)
        single.plot_bbox_heatmap(
            save_path=str(out_dir / f"{suffix}_bbox_heatmap.png"), show=args.show
        )
    else:
        print("[skip] bbox heatmap (--no-heatmap)")

    print(f"\n[done] plots saved under: {out_dir}")


if __name__ == "__main__":
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.60")
    main()
