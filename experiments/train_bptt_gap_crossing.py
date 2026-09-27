##########
# UNIFIED TRAINING SCRIPT FOR GAP CROSSING
# Configure using global variables:
# - USE_VISION: True for vision-based, False for state-based
# - USE_MOVING_GAP: True for moving gap, False for static gap
#
# RUN Detached Examples:
# nohup python train/train_bptt_gap_crossing.py -t -e=60000 -r=100 > /dev/null 2>&1 &
# 
# RESUME Detached:
# PYTHONUNBUFFERED=1 nohup python -u train/train_bptt_gap_crossing.py --resume=14 --resume-checkpoint=45600 --continue-epochs=20000 -r=100 > /dev/null 2>&1 &
##########

import argparse
import csv
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Optional

import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import optax
import yaml
import flax.training.checkpoints as checkpoints
matplotlib.use('Agg')  # Use non-interactive backend to avoid threading issues

from flax.training.train_state import TrainState

# Configure JAX for deterministic behavior
jax.config.update("jax_default_prng_impl", "rbg")  # Use deterministic PRNG
jax.config.update("jax_enable_x64", False)  # Consistent precision

from flightning import FLIGHTNING_PATH, REPO_PATH
from flightning.algos import bptt
from flightning.envs import rollout
from flightning.envs.wrappers import MinMaxObservationWrapper
from flightning.modules import MLP

#############################
# GLOBAL CONFIGURATION FLAGS
#############################
USE_VISION = False  # Set to True for vision-based, False for state-based
USE_MOVING_GAP = False  # Set to True for moving gap, False for static gap
#############################
# QUAD_NAME = "example_quad"  # Set quad name (only "example_quad" is supported)
QUAD_NAME = "example_quad"
FREQUENCY = 60.0  # Control frequency in Hz (must match environment dt)
DELAY = 0.035  # Command delay in seconds (must match environment delay)

GapEnv = None  # Set by _import_gap_env() after argument parsing


def _import_gap_env():
    """Import the appropriate environment class based on global config flags."""
    global GapEnv
    if USE_MOVING_GAP:
        if USE_VISION:
            from flightning.envs import MovingGapVisionEnv as _Env
        else:
            from flightning.envs import MovingGapStateEnv as _Env
    else:
        if USE_VISION:
            from flightning.envs import StaticGapVisionEnv as _Env
        else:
            from flightning.envs import StaticGapStateEnv as _Env
    GapEnv = _Env


def load_training_config():
    """Load training hyperparameters from training_configs.yaml."""
    config_path = os.path.join(REPO_PATH, "configs/training_configs.yaml")
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)
    gap_type = "moving" if USE_MOVING_GAP else "static"
    obs_type = "vision" if USE_VISION else "state"
    config_key = f"{gap_type}_{obs_type}"
    train_cfg = data["training"]["bptt"][config_key]
    print(f"[BPTT] Loaded training config: bptt/{config_key}")
    return train_cfg


def _rotation_matrix_to_euler_zyx(R):
    """Extract roll, pitch, yaw from rotation matrix using ZYX convention.
    
    Args:
        R: 3x3 rotation matrix
        
    Returns:
        (roll, pitch, yaw) in radians
    """
    # ZYX Euler: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    pitch = np.arcsin(-R[2, 0])
    
    # Check for gimbal lock
    if np.abs(np.cos(pitch)) > 1e-6:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        # Gimbal lock case
        roll = 0.0
        yaw = np.arctan2(-R[0, 1], R[1, 1])
    
    return roll, pitch, yaw


def _check_collision_at_crossing(p, R, gap_center, gap_u_axis, gap_v_axis, 
                                 gap_half_size, gap_normal, drone_Q_np):
    """Check if drone ellipsoid collides with gap boundaries at crossing point.
    
    Returns True if there is collision, False if ellipsoid is fully contained in gap.
    """
    # Compute collision vector for the 4 gap boundary planes
    # Using the same logic as in static_gap_state_env.py
    
    # Gap frame basis: [gap_normal, gap_u_axis, gap_v_axis]
    # Constraint normals (pointing inward): [+u, -u, +v, -v]
    A = np.stack([gap_u_axis, -gap_u_axis, gap_v_axis, -gap_v_axis])
    
    # Gap boundaries in constraint form: A @ p <= b
    w_half = gap_half_size[0]
    h_half = gap_half_size[1]
    
    b = np.array([
        w_half + np.dot(gap_u_axis, gap_center),
        w_half - np.dot(gap_u_axis, gap_center),
        h_half + np.dot(gap_v_axis, gap_center),
        h_half - np.dot(gap_v_axis, gap_center),
    ])
    
    # Ellipsoid equation: (p - c)^T Q^T R^T R Q (p - c) <= 1
    Q = drone_Q_np
    RQ = R @ Q
    
    # Collision vector for each constraint
    term1 = A @ p  # shape (4,)
    norms = np.linalg.norm(A @ RQ, axis=1)  # shape (4,)
    collision_vector = term1 + norms - b
    
    # Clip negative values (no penetration)
    collision_vector = np.maximum(collision_vector, 0.0)
    
    # If any component is positive, there is collision
    return np.any(collision_vector > 0.0)


def _to_numpy(x: Any):
    """Convert JAX/NumPy/pytrees leaves to NumPy arrays on host."""
    return np.asarray(jax.device_get(x))


def append_csv_row(csv_path: Path, row: dict):
    """Append a single row to CSV, creating header on first write."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def init_paper_logging(run_dir: Path, file_prefix: str = "") -> dict:
    """Create paper-data directories and return canonical export paths."""
    paper_dir = run_dir / "paper_data"
    rollouts_dir = paper_dir / "rollouts"
    paper_dir.mkdir(parents=True, exist_ok=True)
    rollouts_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{file_prefix}_" if file_prefix else ""
    return {
        "paper_dir": paper_dir,
        "rollouts_dir": rollouts_dir,
        "train_component_losses_csv": paper_dir / f"{prefix}train_component_losses.csv",
        "eval_metrics_csv": paper_dir / f"{prefix}eval_metrics.csv",
        "timing_blocks_csv": paper_dir / f"{prefix}timing_blocks.csv",
    }


def save_eval_rollouts_npz(traj, save_path: Path, metadata: Optional[dict] = None):
    """Serialize rollout tensors for post-processing in a compact NPZ file."""
    save_path.parent.mkdir(parents=True, exist_ok=True)

    state = traj.state
    payload = {
        "obs": _to_numpy(traj.obs),
        "reward": _to_numpy(traj.reward),
        "terminated": _to_numpy(traj.terminated),
        "truncated": _to_numpy(traj.truncated),
    }

    if hasattr(state, "time"):
        payload["time"] = _to_numpy(state.time)
    if hasattr(state, "step_idx"):
        payload["step_idx"] = _to_numpy(state.step_idx)

    q = getattr(state, "quadrotor_state", None)
    if q is not None:
        for field in ["p", "v", "R", "omega", "domega", "motor_omega", "acc"]:
            if hasattr(q, field):
                payload[f"drone_{field}"] = _to_numpy(getattr(q, field))

    for field in [
        "goal",
        "gap_center",
        "gap_normal",
        "gap_dims",
        "gap_u_axis",
        "gap_v_axis",
        "gap_R",
        "pivot",
        "pend_theta",
        "pend_theta_dot",
        "pend_L",
        "gap_center_prev",
    ]:
        if hasattr(state, field):
            payload[field] = _to_numpy(getattr(state, field))

    info = getattr(traj, "info", None)
    if isinstance(info, dict):
        reward_components = info.get("reward_components", None)
        if isinstance(reward_components, dict):
            for comp_name, comp_values in reward_components.items():
                payload[f"reward_component_{comp_name}"] = _to_numpy(comp_values)

    if metadata is not None:
        payload["metadata_json"] = np.array(json.dumps(metadata), dtype="U")

    np.savez_compressed(save_path, **payload)


def compute_and_print_gap_crossing_statistics(traj, env, time_start=None):
    """
    Compute and print gap crossing statistics from trajectory rollouts without generating plots.
    
    This function extracts the same statistics as MultiRolloutVisualizationDashboard
    but only prints them without creating any visualizations.
    
    Args:
        traj: EnvTransition object containing trajectory data
        env: Environment instance
        time_start: Time when training started (from time.time()). If provided, elapsed time will be printed.
    """
    
    # Extract basic data
    state = traj.state
    done = np.logical_or(traj.terminated, traj.truncated)
    num_trajs = traj.reward.shape[0] if traj.reward.ndim == 2 else 1
    drone_Q_np = np.asarray(env.drone_Q)
    
    # Initialize statistics accumulators
    cross_gap_count = 0
    cross_plane_count = 0
    cross_no_collision_count = 0
    avg_final_dist_to_goal = 0.0
    avg_crossing_dist_from_gap_center = 0.0
    avg_crossing_dist_u = 0.0
    avg_crossing_dist_v = 0.0
    avg_crossing_roll_error = 0.0
    avg_crossing_pitch_error = 0.0
    avg_crossing_yaw_error = 0.0
    
    def _sel(x, i):
        """Select data for trajectory i."""
        x = np.asarray(x)
        return x[i] if (x.ndim >= 2 and x.shape[0] == num_trajs) else x
    
    # Process each trajectory
    for i in range(num_trajs):
        # Extract trajectory indices
        if num_trajs > 1:
            idx = np.where(done[i])[0][0].item() + 1
            p = np.asarray(state.quadrotor_state.p[i, :idx])
            R_all = np.asarray(state.quadrotor_state.R[i, :idx])
        else:
            idx = np.where(done)[0][0].item() + 1
            p = np.asarray(state.quadrotor_state.p[:idx])
            R_all = np.asarray(state.quadrotor_state.R[:idx])
        
        # Get goal and final position
        goal = _sel(state.goal, i).astype(float)[0, :]
        final_p = p[-1, :]
        final_dist_to_goal = np.linalg.norm(final_p - goal)
        
        # Get gap geometry at final state
        gap_center_final = _sel(state.gap_center, i).astype(float)[0, :]
        gap_normal_final = _sel(state.gap_normal, i).astype(float)[0, :]
        gap_normal_final /= np.linalg.norm(gap_normal_final) + 1e-9
        
        # Detect crossings using final gap
        rel = p - gap_center_final
        d_signed = rel @ gap_normal_final
        sign_change = (d_signed[:-1] * d_signed[1:]) < -1e-9
        cross_ids = np.where(sign_change)[0]
        
        # Get gap geometry at crossing timestep if crossing occurred
        if cross_ids.size > 0:
            k_cross = cross_ids[0]  # Use first crossing
            if num_trajs > 1:
                gap_center = np.asarray(state.gap_center[i, k_cross]).astype(float)
                gap_normal = np.asarray(state.gap_normal[i, k_cross]).astype(float)
                gap_dims = np.asarray(state.gap_dims[i, k_cross]).astype(float)
                gap_u_axis = np.asarray(state.gap_u_axis[i, k_cross]).astype(float)
                gap_v_axis = np.asarray(state.gap_v_axis[i, k_cross]).astype(float)
            else:
                gap_center = np.asarray(state.gap_center[k_cross]).astype(float)
                gap_normal = np.asarray(state.gap_normal[k_cross]).astype(float)
                gap_dims = np.asarray(state.gap_dims[k_cross]).astype(float)
                gap_u_axis = np.asarray(state.gap_u_axis[k_cross]).astype(float)
                gap_v_axis = np.asarray(state.gap_v_axis[k_cross]).astype(float)
            gap_normal /= np.linalg.norm(gap_normal) + 1e-9
        else:
            # No crossing, use final state
            gap_center = gap_center_final
            gap_normal = gap_normal_final
            gap_dims = _sel(state.gap_dims, i).astype(float)[0, :]
            gap_u_axis = _sel(state.gap_u_axis, i).astype(float)[0, :]
            gap_v_axis = _sel(state.gap_v_axis, i).astype(float)[0, :]
        
        gap_half_size = gap_dims / 2.0
        
        # Detect crossings again with actual gap geometry
        rel = p - gap_center
        d_signed = rel @ gap_normal
        sign_change = (d_signed[:-1] * d_signed[1:]) < -1e-9
        cross_ids = np.where(sign_change)[0]
        
        if cross_ids.size > 0:
            hits = []
            for k in cross_ids:
                denom = d_signed[k + 1] - d_signed[k]
                if abs(denom) < 1e-12:
                    continue
                s = np.clip(-d_signed[k] / denom, 0.0, 1.0)
                p_cross = p[k] + s * (p[k + 1] - p[k])
                hits.append(p_cross)
            
            if hits:
                hits = np.array(hits)
                rel_hits = hits - gap_center
                d_hits = rel_hits @ gap_normal
                hits_proj = hits - d_hits[:, None] * gap_normal[None, :]
                uv = np.stack([
                    (hits_proj - gap_center) @ gap_u_axis,
                    (hits_proj - gap_center) @ gap_v_axis,
                ], axis=1)
                best_idx = np.argmin((uv**2).sum(axis=1))
                u, v = uv[best_idx]
                
                k_cross = cross_ids[best_idx]
                R_cross = R_all[k_cross + 1] if k_cross + 1 < len(R_all) else R_all[-1]
                
                # Calculate distance from gap center (2D in gap plane)
                crossing_dist_from_center = np.sqrt(u**2 + v**2)
                
                # Calculate orientation errors at crossing
                gap_R = np.stack([gap_normal, gap_u_axis, gap_v_axis], axis=1)
                R_rel = gap_R.T @ R_cross
                roll_drone, pitch_drone, yaw_drone = _rotation_matrix_to_euler_zyx(R_rel)
                
                # Update plane crossing statistics (provisional)
                cross_plane_count += 1
                avg_final_dist_to_goal += final_dist_to_goal
                avg_crossing_dist_from_gap_center += crossing_dist_from_center
                avg_crossing_dist_u += np.abs(u)
                avg_crossing_dist_v += np.abs(v)
                avg_crossing_roll_error += np.abs(roll_drone)
                avg_crossing_pitch_error += np.abs(pitch_drone)
                avg_crossing_yaw_error += np.abs(yaw_drone)
                
                # Check if crossing is within gap boundaries
                if -gap_half_size[0] < u < gap_half_size[0] and -gap_half_size[1] < v < gap_half_size[1]:
                    cross_gap_count += 1
                    
                    # Check collision at crossing point
                    p_cross_3d = hits[best_idx]
                    has_collision = _check_collision_at_crossing(
                        p_cross_3d, R_cross, gap_center, gap_u_axis, gap_v_axis, 
                        gap_half_size, gap_normal, drone_Q_np
                    )
                    
                    if not has_collision:
                        cross_no_collision_count += 1
    
    # Compute final statistics
    cross_gap_rate = (cross_gap_count / max(1, num_trajs)) * 100.0
    cross_plane_rate = (cross_plane_count / max(1, num_trajs)) * 100.0
    cross_no_collision_rate = (cross_no_collision_count / max(1, num_trajs)) * 100.0
    avg_dist = avg_final_dist_to_goal / max(1, cross_plane_count)
    avg_crossing_dist = avg_crossing_dist_from_gap_center / max(1, cross_plane_count)
    avg_crossing_u = avg_crossing_dist_u / max(1, cross_plane_count)
    avg_crossing_v = avg_crossing_dist_v / max(1, cross_plane_count)
    avg_roll_error = np.rad2deg(avg_crossing_roll_error / max(1, cross_plane_count))
    avg_pitch_error = np.rad2deg(avg_crossing_pitch_error / max(1, cross_plane_count))
    avg_yaw_error = np.rad2deg(avg_crossing_yaw_error / max(1, cross_plane_count))
    
    # Print statistics
    elapsed_time = None
    if time_start is not None:
        elapsed_time = time.time() - time_start
    
    print("\n" + "="*80)
    if elapsed_time is not None:
        print(f"GAP CROSSING STATISTICS [Elapsed: {elapsed_time:.1f}s]")
    else:
        print("GAP CROSSING STATISTICS")
    print("="*80)
    print(f"[SUCCESS RATE]: {cross_no_collision_rate:.2f}% (no collision)")
    print(f"Gap crossing rate: {cross_gap_rate:.2f}% - num trajs: {num_trajs}")
    print(f"Gap plane crossing rate: {cross_plane_rate:.2f}% - num trajs: {num_trajs}")
    print(f"Average final distance to goal: {100*avg_dist:.1f} cm")
    print(f"Average crossing distance from gap center: {100*avg_crossing_dist:.1f} cm")
    print(f"Average crossing distance u-axis: {100*avg_crossing_u:.1f} cm")
    print(f"Average crossing distance v-axis: {100*avg_crossing_v:.1f} cm")
    print(f"Average crossing orientation errors - Roll: {avg_roll_error:.2f}°, Pitch: {avg_pitch_error:.2f}°, Yaw: {avg_yaw_error:.2f}°")
    print("="*80 + "\n")
    
    stats = {
        "num_trajectories": int(num_trajs),
        "cross_no_collision_count": int(cross_no_collision_count),
        "cross_gap_count": int(cross_gap_count),
        "cross_plane_count": int(cross_plane_count),
        "success_rate_no_collision_pct": float(cross_no_collision_rate),
        "gap_crossing_rate_pct": float(cross_gap_rate),
        "gap_plane_crossing_rate_pct": float(cross_plane_rate),
        "avg_final_dist_to_goal_m": float(avg_dist),
        "avg_crossing_dist_from_gap_center_m": float(avg_crossing_dist),
        "avg_crossing_dist_u_m": float(avg_crossing_u),
        "avg_crossing_dist_v_m": float(avg_crossing_v),
        "avg_crossing_roll_error_deg": float(avg_roll_error),
        "avg_crossing_pitch_error_deg": float(avg_pitch_error),
        "avg_crossing_yaw_error_deg": float(avg_yaw_error),
        "elapsed_seconds": float(elapsed_time) if elapsed_time is not None else np.nan,
    }
    return stats


def get_config_suffix() -> str:
    """Return a suffix string describing the current configuration."""
    gap_type = "moving" if USE_MOVING_GAP else "static"
    obs_type = "vision" if USE_VISION else "state"
    return f"{gap_type}_{obs_type}"


def setup_run_dir(num_envs: int) -> Path:
    base_run_dir = Path(REPO_PATH) / "runs"
    base_run_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S") + f"_{int((time.time() % 1) * 1000):03d}"
    suffix = get_config_suffix()
    run_dir = base_run_dir / f"{timestamp}_bptt_{suffix}_envs{num_envs}"
    run_dir.mkdir()

    # Save a copy of training_configs.yaml
    config_src = os.path.join(REPO_PATH, "configs/training_configs.yaml")
    shutil.copy2(config_src, run_dir / "training_configs.yaml")

    print(f"Saving run to: {run_dir.resolve()}")
    print(f"Configuration: {suffix}")
    return run_dir


def create_policy(obs_dim: int, action_dim: int, env) -> MLP:
    """Create policy network with appropriate architecture based on configuration."""
    train_cfg = load_training_config()
    hidden_dims = [obs_dim] + train_cfg["hidden_dims"] + [action_dim]
    initial_scale = train_cfg.get("initial_scale", 0.01)
    
    return MLP(
        hidden_dims,
        initial_scale=initial_scale,
        action_bias=env.hovering_action,
    )


def create_env(drone_path: str, num_epochs=1000, training=True, dt=1/FREQUENCY, #! CHANGE at 60 Hz
               delay=DELAY, seconds_per_episode=2.5, dyn_config=None, residual_params=None, 
               initial_epoch=0):
    """Create environment based on global configuration flags."""
    env = GapEnv(
        max_steps_in_episode=int(seconds_per_episode / dt),
        dt=dt,
        delay=delay,
        drone_path=drone_path,
        dyn_config=dyn_config,
        res_model_params=residual_params,
        training=training,
        num_epochs=num_epochs,
        initial_epoch=initial_epoch,
        algo="bptt",
    )
    return MinMaxObservationWrapper(env)


def create_train_state(drone_path: str, key_init, num_epochs=1000, dyn_config=None, initial_epoch=0) -> TrainState:
    """Create training state with appropriate configuration."""
    env = create_env(drone_path, num_epochs=num_epochs, dyn_config=dyn_config, 
                     initial_epoch=initial_epoch)
    action_dim = env.action_space.shape[0]
    obs_dim = env.observation_space.shape[0]

    policy_net = create_policy(obs_dim, action_dim, env)
    policy_params = policy_net.initialize(key_init)

    # Configure learning rate scheduler from config
    train_cfg = load_training_config()
    scheduler = optax.warmup_cosine_decay_schedule(
        init_value=train_cfg["lr_init"],
        peak_value=train_cfg["lr_peak"],
        warmup_steps=train_cfg["lr_warmup_steps"],
        decay_steps=train_cfg["lr_decay_steps"],
        end_value=train_cfg["lr_end"]
    )
    
    tx = optax.adam(scheduler)
    train_state = TrainState.create(apply_fn=policy_net.apply, params=policy_params, tx=tx)
    return env, train_state


def get_rollouts(env, policy, num_rollouts, key):
    """Execute parallel rollouts."""
    parallel_rollout = jax.vmap(rollout, in_axes=(None, 0, None))
    rollout_keys = jax.random.split(key, num_rollouts)
    transitions = parallel_rollout(env, rollout_keys, policy)
    return transitions


def plot_and_save(save_path: Path, title: str = None):
    """Save current plot to file."""
    if title:
        plt.title(title)
    plt.savefig(save_path.resolve())
    plt.close()


def train_model(env, train_state, key_bptt, run_dir: Path, num_epochs: int, 
                num_rollouts: int, key_eval, dyn_config=None, drone_path=None, num_envs=100) -> TrainState:
    """Train the policy using BPTT."""
    print("Starting BPTT training...")
    time_start = time.time()
    
    # Create checkpoint directory
    model_dir = run_dir / "checkpoints"
    model_dir.mkdir(exist_ok=True)
    artifact_prefix = f"bptt_{get_config_suffix()}_envs{num_envs}"
    paper_paths = init_paper_logging(run_dir, file_prefix=artifact_prefix)
    print(f"🧾 Paper data export enabled: {paper_paths['paper_dir'].resolve()}")
    
    # Training in blocks to save checkpoints periodically
    train_cfg = load_training_config()
    checkpoint_interval = train_cfg.get("checkpoint_interval", 500)
    num_blocks = (num_epochs + checkpoint_interval - 1) // checkpoint_interval
    
    all_reward_components = {}
    current_train_state = train_state
    initial_step = int(getattr(train_state, 'step', 0)) if train_state is not None else 0
    
    # Create evaluation environment once (reused across all checkpoints to avoid JIT recompilation)
    env_eval = create_env(drone_path, training=False, dyn_config=dyn_config)
    
    # Pre-compile JIT training function once (reused across all blocks)
    train_fn, wrapped_env = bptt.make_train(
        env=env,
        num_epochs=checkpoint_interval,
        num_steps_per_epoch=env.max_steps_in_episode,
        num_envs=num_envs,
    )
    
    for block_idx in range(num_blocks):
        block_start_epoch = block_idx * checkpoint_interval
        block_epochs = min(checkpoint_interval, num_epochs - block_start_epoch)
        block_time_start = time.time()
        
        print(f"\n{'='*60}")
        print(f"Training block {block_idx + 1}/{num_blocks}")
        print(f"Epochs: {block_start_epoch} to {block_start_epoch + block_epochs - 1}")
        print(f"{'='*60}")

        key_bptt, key_block = jax.random.split(key_bptt)
        
        # Recompile only if last block has fewer epochs
        if block_epochs != checkpoint_interval:
            train_fn, wrapped_env = bptt.make_train(
                env=env,
                num_epochs=block_epochs,
                num_steps_per_epoch=env.max_steps_in_episode,
                num_envs=num_envs,
            )
        
        # Initialize environments and run compiled training function
        key_block, key_reset = jax.random.split(key_block)
        key_reset_envs = jax.random.split(key_reset, num_envs)
        env_state, obs = wrapped_env.reset(key_reset_envs, None)
        runner_state = bptt.RunnerState(current_train_state, env_state, obs, key_block, epoch_idx=0)

        # Separately time JIT compile vs execution (instrumentation only —
        # does not change numerics). After the first block (or on the
        # last block if it triggered a recompile) the compile is cached
        # and `compile_seconds` will be near-zero.
        _t_compile = time.time()
        compiled_train_fn = train_fn.lower(runner_state).compile()
        block_compile_seconds = time.time() - _t_compile

        _t_exec = time.time()
        res_dict = compiled_train_fn(runner_state)
        jax.block_until_ready(res_dict["runner_state"].train_state.params)
        block_exec_seconds = time.time() - _t_exec

        block_train_seconds = time.time() - block_time_start

        current_train_state = res_dict['runner_state'].train_state
        reward_components = res_dict['reward_components']

        # Export per-epoch train losses (wide format: epoch, comp1_loss, ..., total_loss)
        reward_components_np = {
            comp_name: _to_numpy(comp_values).reshape(-1)
            for comp_name, comp_values in reward_components.items()
        }
        if reward_components_np:
            local_num_epochs = min(len(v) for v in reward_components_np.values())
            for local_epoch_idx in range(local_num_epochs):
                epoch = initial_step + block_start_epoch + local_epoch_idx + 1
                total_loss = float(sum(v[local_epoch_idx] for v in reward_components_np.values()))
                row = {"epoch": int(epoch)}
                for comp_name, comp_values in reward_components_np.items():
                    row[f"{comp_name}_loss"] = float(comp_values[local_epoch_idx])
                row["total_loss"] = total_loss
                append_csv_row(paper_paths["train_component_losses_csv"], row)
        
        # Accumulate reward components
        for comp_name, comp_values in reward_components.items():
            if comp_name not in all_reward_components:
                all_reward_components[comp_name] = []
            all_reward_components[comp_name].extend(comp_values)
        
        # Save checkpoint
        checkpoint_step = initial_step + block_start_epoch + block_epochs
        checkpoint_time_start = time.time()
        checkpoints.save_checkpoint(
            ckpt_dir=model_dir.resolve(),
            target=current_train_state,
            step=checkpoint_step,
            overwrite=True,
            keep=100,
        )
        checkpoint_seconds = time.time() - checkpoint_time_start
        print(f"💾 Checkpoint saved at epoch {checkpoint_step}")
        
        # Evaluate checkpoint with multi-rollout dashboard
        print(f"🔍 Evaluating checkpoint at epoch {checkpoint_step}...")
        
        # Create policy function for evaluation
        def policy_current(obs, key=None):
            return current_train_state.apply_fn(current_train_state.params, obs)
        
        # Split key_eval to ensure different rollouts at each checkpoint
        eval_time_start = time.time()
        key_eval, key_eval_checkpoint = jax.random.split(key_eval)
        transitions_eval = get_rollouts(env_eval, policy_current, num_rollouts=num_rollouts, key=key_eval_checkpoint)
        
        # # Generate complete dashboard with all multi-rollout visualizations
        # print(f"📊 Generating complete multi-rollout dashboard...")
        # dashboard_multi = MultiRolloutVisualizationDashboard(transitions_eval, env_eval)
        # dashboard_multi.generate_complete_dashboard(show=False, save_path=str(checkpoint_eval_dir / "multi_rollout_dashboard.png"))
        
        eval_stats = compute_and_print_gap_crossing_statistics(transitions_eval, env_eval, time_start)
        eval_seconds = time.time() - eval_time_start

        # Save raw rollout tensors for later post-processing
        rollout_npz_path = paper_paths["rollouts_dir"] / f"{artifact_prefix}_eval_epoch_{checkpoint_step}.npz"
        save_eval_rollouts_npz(
            transitions_eval,
            rollout_npz_path,
            metadata={
                "checkpoint_epoch": int(checkpoint_step),
                "num_rollouts": int(num_rollouts),
                "algo": "bptt",
                "use_moving_gap": bool(USE_MOVING_GAP),
                "use_vision": bool(USE_VISION),
            },
        )

        append_csv_row(
            paper_paths["eval_metrics_csv"],
            {
                "checkpoint_epoch": int(checkpoint_step),
                "num_rollouts": int(num_rollouts),
                "eval_seconds": float(eval_seconds),
                "rollout_npz": rollout_npz_path.name,
                "global_elapsed_seconds": float(time.time() - time_start),
                **eval_stats,
            },
        )

        append_csv_row(
            paper_paths["timing_blocks_csv"],
            {
                "block_idx": int(block_idx + 1),
                "start_epoch": int(initial_step + block_start_epoch + 1),
                "end_epoch": int(checkpoint_step),
                "block_epochs": int(block_epochs),
                "train_seconds": float(block_train_seconds),
                "compile_seconds": float(block_compile_seconds),
                "exec_seconds": float(block_exec_seconds),
                "checkpoint_seconds": float(checkpoint_seconds),
                "eval_seconds": float(eval_seconds),
                "total_block_seconds": float(time.time() - block_time_start),
                "global_elapsed_seconds": float(time.time() - time_start),
            },
        )
    
    time_train = time.time() - time_start
    print(f"\n⏱️  Total training time: {time_train:.2f} seconds")

    # Plot reward components
    if all_reward_components:
        reward_components_keys = list(all_reward_components.keys())
        plt.figure()
        total_reward = jnp.zeros(len(all_reward_components[reward_components_keys[0]]))
        for comp in reward_components_keys:
            plt.plot(all_reward_components[comp], label=comp, alpha=0.7)
            total_reward += jnp.array(all_reward_components[comp])
        plt.plot(total_reward, label="total", color='black')
        plt.xlabel("Epoch")
        plt.ylabel("Component sum per epoch")
        plt.legend()
        plot_and_save(run_dir / "reward_components_plot.png", title="Reward Components per Epoch")

    print(f"✅ All checkpoints saved to {model_dir.resolve()}")
    print(f"✅ Paper data saved to {paper_paths['paper_dir'].resolve()}")
    return current_train_state


def load_model(train_state, checkpoint_path: str, resume: bool = False):
    """
    Load model from a checkpoint path.
    
    Args:
        train_state: Template train state for structure
        checkpoint_path: Path to checkpoint directory (e.g. 'runs/20260423_.../checkpoints/checkpoint_295')
                         or to a checkpoints/ directory (loads latest).
        resume: If True, always load the latest checkpoint in the directory.
    
    Returns:
        Updated train_state with loaded parameters
    """
    ckpt_path = Path(checkpoint_path).resolve()
    
    # If pointing to a checkpoints/ directory, find the latest checkpoint inside
    if ckpt_path.is_dir() and not ckpt_path.name.startswith("checkpoint_"):
        checkpoint_files = sorted(ckpt_path.glob("checkpoint_*"))
        if not checkpoint_files:
            raise FileNotFoundError(f"No checkpoints found in {ckpt_path}")
        steps = []
        for cf in checkpoint_files:
            try:
                steps.append(int(cf.name.split("_")[1]))
            except (IndexError, ValueError):
                continue
        if not steps:
            raise FileNotFoundError(f"No valid checkpoints found in {ckpt_path}")
        latest_step = max(steps)
        print(f"No specific checkpoint given, loading latest: {latest_step}")
        model_dir = ckpt_path
        checkpoint_step = latest_step
    else:
        # Pointing to a specific checkpoint_N directory
        model_dir = ckpt_path.parent
        try:
            checkpoint_step = int(ckpt_path.name.split("_")[1])
        except (IndexError, ValueError):
            raise ValueError(f"Cannot parse checkpoint step from: {ckpt_path.name}")
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint path does not exist: {ckpt_path}")
        
        if resume:
            checkpoint_files = sorted(model_dir.glob("checkpoint_*"))
            steps = []
            for cf in checkpoint_files:
                try:
                    steps.append(int(cf.name.split("_")[1]))
                except (IndexError, ValueError):
                    continue
            if steps and max(steps) > checkpoint_step:
                checkpoint_step = max(steps)
                print(f"Resume mode: loading latest checkpoint {checkpoint_step}")
    
    print(f"Loading checkpoint {checkpoint_step} from {model_dir}")
    
    # Restore full TrainState (optimizer state + step)
    restored = checkpoints.restore_checkpoint(
        ckpt_dir=str(model_dir),
        target=train_state,
        step=checkpoint_step,
    )
    
    if hasattr(restored, 'params'):
        return restored
    else:
        raise ValueError("Restored checkpoint does not have expected TrainState structure")


def main(args):
    # Determine mode
    if args.train:
        mode = "train"
    elif args.load:
        mode = "load"
    elif getattr(args, 'resume', None) is not None:
        mode = "resume"
    else:
        mode = "other"
    
    # Only create a new run dir for new runs; for resume reuse existing run dir
    if mode != "resume":
        run_dir = setup_run_dir(args.num_envs)
    else:
        # Derive run_dir from the checkpoint path (go up from checkpoints/ dir)
        resume_path = Path(args.resume).resolve()
        if resume_path.parent.name == "checkpoints":
            run_dir = resume_path.parent.parent
        elif resume_path.name == "checkpoints":
            run_dir = resume_path.parent
        else:
            run_dir = resume_path.parent
        if not run_dir.exists():
            raise FileNotFoundError(f"Cannot resume - run directory not found from: {args.resume}")
        print(f"Resuming training in existing directory: {run_dir.resolve()}")

    # --- Logging: redirect stdout/stderr to a run-specific log file ---
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        log_file = open(log_path, 'a', buffering=1)
        
        class TeeStream:
            def __init__(self, *streams):
                self.streams = streams
            def write(self, data):
                for stream in self.streams:
                    stream.write(data)
                    stream.flush()
            def flush(self):
                for stream in self.streams:
                    stream.flush()
        
        sys.stdout = TeeStream(sys.stdout, log_file)
        sys.stderr = TeeStream(sys.stderr, log_file)
        
        print(f"\n{'='*60}")
        print(f"Logging to: {log_path.resolve()}")
        print(f"Configuration: {get_config_suffix()}")
        print(f"{'='*60}\n")
    except Exception as e:
        print(f"⚠️  Warning: Could not set up logging: {e}")

    # Set random seeds
    seed = args.seed if hasattr(args, 'seed') and args.seed is not None else 0
    print(f"Using seed: {seed}")

    np.random.seed(seed)
    random.seed(seed)
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

    key = jax.random.PRNGKey(seed)
    key_init, key_bptt, key_eval = jax.random.split(key, 3)

    
    drone_path = FLIGHTNING_PATH + "/objects/quadrotor_files/example_quad.yaml"

    dyn_config = {
        "high_fidelity_forward": False,
        "forward_residual": False,
        "backprop_residual": False,
    }
    
    env, train_state = create_train_state(drone_path, key_init, num_epochs=args.epochs, dyn_config=dyn_config)
    
    
    # --- Resume training from an existing run/checkpoint (exact continuation) ---
    if args.resume is not None:
        train_state = load_model(train_state, args.resume, resume=True)
        
        current_epoch = int(train_state.step)
        total_epochs = current_epoch + args.continue_epochs
        print(f"Resuming from epoch {current_epoch}, training until epoch {total_epochs}")
        
        # Create new env with updated epoch range
        env, _ = create_train_state(drone_path, key_init, num_epochs=total_epochs, 
                                     dyn_config=dyn_config, initial_epoch=current_epoch)
        
        train_state = train_model(env, train_state, key_bptt, run_dir, args.continue_epochs, 
                                  args.rollouts, key_eval, dyn_config=dyn_config, drone_path=drone_path, num_envs=args.num_envs)
    
    # --- Train a new model ---
    elif args.train:
        train_state = train_model(env, train_state, key_bptt, run_dir, args.epochs, 
                                  args.rollouts, key_eval, dyn_config=dyn_config, drone_path=drone_path, num_envs=args.num_envs)
    
    # --- Load an existing model ---
    elif args.load is not None:
        train_state = load_model(train_state, args.load)
        print(f"✅ Model loaded successfully from {args.load}")
        
        # Generate evaluation dashboards for loaded model
        print(f"\n🔍 Evaluating loaded model...")
        load_eval_dir = run_dir / "evaluation"
        load_eval_dir.mkdir(exist_ok=True)
        
        # Create evaluation environment
        env_eval_load = create_env(drone_path, training=False, dyn_config=dyn_config)
        
        # Create policy function
        def policy_load(obs, key=None):
            return train_state.apply_fn(train_state.params, obs)
        
        # Run evaluation rollouts
        # Advance key_eval by the same number of splits that happened during training
        # so the evaluation uses the exact same key as at this checkpoint
        checkpoint_epoch = int(train_state.step)
        train_cfg_load = load_training_config()
        checkpoint_interval_load = train_cfg_load.get("checkpoint_interval", 500)
        num_eval_splits = checkpoint_epoch // checkpoint_interval_load
        key_eval_load = key_eval
        key_eval_checkpoint = key_eval  # fallback if num_eval_splits == 0
        for _ in range(num_eval_splits):
            key_eval_load, key_eval_checkpoint = jax.random.split(key_eval_load)
        transitions_eval_load = get_rollouts(env_eval_load, policy_load, num_rollouts=args.rollouts, key=key_eval_checkpoint)
        
        time_start_load = time.time()
        eval_stats = compute_and_print_gap_crossing_statistics(transitions_eval_load, env_eval_load, time_start_load)

        artifact_prefix = f"bptt_{get_config_suffix()}_envs{args.num_envs}"
        paper_paths = init_paper_logging(run_dir, file_prefix=artifact_prefix)
        rollout_npz_path = paper_paths["rollouts_dir"] / f"{artifact_prefix}_evaluation_loaded_model.npz"
        save_eval_rollouts_npz(
            transitions_eval_load,
            rollout_npz_path,
            metadata={
                "mode": "load",
                "checkpoint_path": args.load,
                "num_rollouts": int(args.rollouts),
                "algo": "bptt",
                "use_moving_gap": bool(USE_MOVING_GAP),
                "use_vision": bool(USE_VISION),
            },
        )
        append_csv_row(
            paper_paths["eval_metrics_csv"],
            {
                "checkpoint_path": args.load,
                "num_rollouts": int(args.rollouts),
                "eval_seconds": float(time.time() - time_start_load),
                "rollout_npz": rollout_npz_path.name,
                "global_elapsed_seconds": float(time.time() - time_start_load),
                **eval_stats,
            },
        )
    
    else:
        raise ValueError("Must specify one of: --train, --load, or --resume")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified BPTT training script for gap crossing.")
    parser.add_argument("-t", "--train", action="store_true", help="Train a new model.")
    parser.add_argument("-l", "--load", type=str, default=None, help="Path to checkpoint to load (e.g. 'runs/20260423_.../checkpoints/checkpoint_295' or 'runs/.../checkpoints/' for latest).")
    parser.add_argument("-r", "--rollouts", type=int, default=100, help="Number of rollouts for evaluation.")
    parser.add_argument("-e", "--epochs", type=int, default=20000, help="Number of training epochs.")
    parser.add_argument("-s", "--seed", type=int, default=0, help="Random seed for reproducibility (default: 0).")
    
    parser.add_argument("--num-envs", type=int, default=100, help="Number of parallel environments for training (default: 100).")

    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from (e.g. 'runs/.../checkpoints/checkpoint_295' or 'runs/.../checkpoints/' for latest).")
    parser.add_argument("--continue-epochs", type=int, default=1000, help="Number of additional epochs to train when resuming.")
    parser.add_argument("--moving", action="store_true", help="Use moving gap environment (default: static gap).")
    parser.add_argument("--vision", action="store_true", help="Use vision-based observations (default: state-based).")
    
    args = parser.parse_args()
    
    # Apply CLI flags to global config
    USE_MOVING_GAP = args.moving
    USE_VISION = args.vision
    _import_gap_env()
    
    # Print configuration
    print("\n" + "="*60)
    print("UNIFIED GAP CROSSING TRAINING")
    print("="*60)
    print(f"Configuration:")
    print(f"  USE_VISION: {USE_VISION}")
    print(f"  USE_MOVING_GAP: {USE_MOVING_GAP}")
    print(f"  Environment: {GapEnv.__name__}")
    print(f"  Network size: {'1024x1024' if USE_VISION else '512x512'}")
    print("="*60 + "\n")
    
    main(args)
