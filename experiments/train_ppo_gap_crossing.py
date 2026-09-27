##########
# UNIFIED TRAINING SCRIPT FOR GAP CROSSING
# Configure using global variables:
# - USE_VISION: True for vision-based, False for state-based
# - USE_MOVING_GAP: True for moving gap, False for static gap
#
# RUN Detached Examples:
# nohup python train/train_ppo_gap_crossing.py -t -e=60000 -r=100 > /dev/null 2>&1 &
# 
# RESUME Detached:
# PYTHONUNBUFFERED=1 nohup python -u train/train_ppo_gap_crossing.py --resume=14 --resume-checkpoint=45600 --continue-epochs=20000 -r=100 > /dev/null 2>&1 &
##########

import argparse
import os
import random
import shutil
import sys
import time
from pathlib import Path

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

# Import shared evaluation utilities from BPTT script
sys.path.insert(0, str(Path(__file__).parent))
from train_bptt_gap_crossing import (
    append_csv_row,
    compute_and_print_gap_crossing_statistics,
    init_paper_logging,
    save_eval_rollouts_npz,
)

# Configure JAX for deterministic behavior
jax.config.update("jax_default_prng_impl", "rbg")  # Use deterministic PRNG
jax.config.update("jax_enable_x64", False)  # Consistent precision

from flightning import FLIGHTNING_PATH, REPO_PATH
from flightning.algos import ppo
from flightning.envs import rollout
from flightning.envs.wrappers import MinMaxObservationWrapper
from flightning.modules import ActorCriticPPO

#############################
# GLOBAL CONFIGURATION FLAGS
#############################
USE_VISION = False  # Set to True for vision-based, False for state-based # NOTE: not supported for PPO
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
    train_cfg = data["training"]["ppo"][config_key]
    print(f"[PPO] Loaded training config: ppo/{config_key}")
    return train_cfg


def load_curriculum_config():
    """Load curriculum learning config from training_configs.yaml."""
    train_cfg = load_training_config()
    curriculum_cfg = {
        "enabled": train_cfg.get("curriculum_enabled", False),
        "start_epoch": train_cfg.get("curriculum_start_epoch", 0),
        "final_epoch": train_cfg.get("curriculum_final_epoch", 1),
        "weights": train_cfg.get("curriculum_weights", {}),
    }
    return curriculum_cfg


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
    run_dir = base_run_dir / f"{timestamp}_ppo_{suffix}_envs{num_envs}"
    run_dir.mkdir()

    # Save a copy of training_configs.yaml
    config_src = os.path.join(REPO_PATH, "configs/training_configs.yaml")
    shutil.copy2(config_src, run_dir / "training_configs.yaml")

    print(f"Saving run to: {run_dir.resolve()}")
    print(f"Configuration: {suffix}")
    return run_dir


def create_policy(obs_dim: int, action_dim: int, env) -> ActorCriticPPO:
    """Create policy network with appropriate architecture based on configuration."""
    train_cfg = load_training_config()
    hidden_dims = [obs_dim] + train_cfg["hidden_dims"] + [action_dim]
    initial_log_std = train_cfg.get("initial_log_std", 0.0)
    
    return ActorCriticPPO(
        hidden_dims,
        initial_log_std=initial_log_std
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
        algo="ppo",
    )
    return MinMaxObservationWrapper(env)


def create_train_state(drone_path: str, key_init, num_epochs=1000, dyn_config=None, initial_epoch=0, num_envs: int = 100) -> TrainState:
    """Create training state with appropriate configuration."""
    env = create_env(drone_path, num_epochs=num_epochs, dyn_config=dyn_config, 
                     initial_epoch=initial_epoch)
    action_dim = env.action_space.shape[0]
    obs_dim = env.observation_space.shape[0]

    policy_net = create_policy(obs_dim, action_dim, env)
    policy_params = policy_net.initialize(key_init)

    # Configure learning rate scheduler from config
    train_cfg = load_training_config()
    update_epochs = train_cfg.get("update_epochs", 4)
    num_minibatches = train_cfg.get("num_minibatches", 20)
    # Fixed decay steps (independent of num_epochs) to keep LR schedule consistent
    fixed_decay_epochs = 20000
    total_optimizer_steps = fixed_decay_epochs * num_minibatches * update_epochs
    scheduler = optax.warmup_cosine_decay_schedule(
        init_value=train_cfg["lr_init"],
        peak_value=train_cfg["lr_peak"],
        warmup_steps=train_cfg["lr_warmup_steps"],
        decay_steps=total_optimizer_steps,
        end_value=train_cfg["lr_end"]
    )
    
    tx = optax.chain(
        optax.clip_by_global_norm(train_cfg.get("max_grad_norm", 1.0)),
        optax.adam(scheduler),
    )
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


def get_curriculum_weights(epoch: int) -> dict:
    """Curriculum learning: linearly ramp reward weights between start_epoch and final_epoch.
    
    Parameters are loaded from training_configs.yaml under curriculum/ppo/<config_key>.
    Each weight entry is [start_value, end_value]. Linear interpolation is applied
    between start_epoch and final_epoch; values are clamped outside that range.
    """
    curriculum_cfg = load_curriculum_config()
    
    if not curriculum_cfg or not curriculum_cfg.get("enabled", False):
        return {}
    
    ramp_start = curriculum_cfg["start_epoch"]
    ramp_end = curriculum_cfg["final_epoch"]
    
    if epoch <= ramp_start:
        t = 0.0
    elif epoch >= ramp_end:
        t = 1.0
    else:
        t = (epoch - ramp_start) / (ramp_end - ramp_start)
    
    result = {}
    for weight_name, (start_val, end_val) in curriculum_cfg["weights"].items():
        result[weight_name] = start_val + t * (end_val - start_val)
    
    return result


def train_model(env, train_state, key_ppo, run_dir: Path, num_epochs: int, 
                num_rollouts: int, key_eval, dyn_config=None, drone_path=None, num_envs: int = 100, is_finetune: bool = False) -> TrainState:
    """Train the policy using BPTT."""
    print("Starting BPTT training...")
    time_start = time.time()
    
    # Create checkpoint directory
    model_dir = run_dir / "checkpoints"
    model_dir.mkdir(exist_ok=True)
    artifact_prefix = f"ppo_{get_config_suffix()}_envs{num_envs}"
    paper_paths = init_paper_logging(run_dir, file_prefix=artifact_prefix)
    print(f"🧾 Paper data export enabled: {paper_paths['paper_dir'].resolve()}")
    
    # Training in blocks to save checkpoints periodically
    train_cfg = load_training_config()
    checkpoint_interval = train_cfg.get("checkpoint_interval", 1000)
    num_blocks = (num_epochs + checkpoint_interval - 1) // checkpoint_interval
    
    all_reward_components = {}
    current_train_state = train_state
    initial_step = int(getattr(train_state, 'step', 0)) if train_state is not None else 0
    
    # Track whether we've switched to finetune LR (moving gap only)
    finetune_lr_applied = is_finetune  # True if --finetune-lr was used
    FINETUNE_EPOCH = 10000  # Switch to constant LR after this epoch (moving gap only)
    FINETUNE_LR = 5e-5
    
    # Create evaluation environment once (reused across all checkpoints to avoid JIT recompilation)
    env_eval = create_env(drone_path, training=False, dyn_config=dyn_config)
    
    for block_idx in range(num_blocks):
        block_start_epoch = block_idx * checkpoint_interval
        block_epochs = min(checkpoint_interval, num_epochs - block_start_epoch)
        block_time_start = time.time()
        
        print(f"\n{'='*60}")
        print(f"Training block {block_idx + 1}/{num_blocks}")
        print(f"Epochs: {block_start_epoch} to {block_start_epoch + block_epochs - 1}")
        print(f"{'='*60}")

        key_ppo, key_block = jax.random.split(key_ppo)
        
        # Curriculum learning: only for moving gap PPO
        total_epoch = initial_step + block_start_epoch
        
        # Recreate env each block so JAX JIT retraces with new weight values
        # (JAX captures env attributes as constants during tracing)
        env = create_env(drone_path, num_epochs=num_epochs, dyn_config=dyn_config,
                         initial_epoch=total_epoch)
        inner_env = env._env
        
        curriculum = get_curriculum_weights(total_epoch)
        if curriculum:
            for ckey, cval in curriculum.items():
                setattr(inner_env, ckey, cval)
            print(f"📈 Curriculum (epoch {total_epoch}): collision={inner_env.collision_weight:.1f}, "
                  f"near_gap={inner_env.near_gap_centering_weight:.1f}, alignment={inner_env.alignment_weight:.1f}, "
                  f"centering={inner_env.centering_weight:.1f}")
        else:
            print(f"📈 Fixed weights (from config): collision={inner_env.collision_weight:.1f}, "
                  f"near_gap={inner_env.near_gap_centering_weight:.1f}, alignment={inner_env.alignment_weight:.1f}, "
                  f"centering={inner_env.centering_weight:.1f}")
        
        # Entropy annealing: decay ent_coef from 0.02 to 0.002 over first 6000 epochs
        ent_anneal_end = 6000
        base_ent_coef = 0.02
        min_ent_coef = 0.002
        if total_epoch >= ent_anneal_end:
            annealed_ent_coef = min_ent_coef
        else:
            annealed_ent_coef = base_ent_coef - (base_ent_coef - min_ent_coef) * (total_epoch / ent_anneal_end)
        print(f"📉 Entropy coef: {annealed_ent_coef:.4f}")
        
        # Train for this block
        train_cfg_block = load_training_config()
        
        # Auto finetune: switch to constant LR after FINETUNE_EPOCH (moving gap only)
        if USE_MOVING_GAP and not finetune_lr_applied and total_epoch >= FINETUNE_EPOCH:
            ft_tx = optax.chain(
                optax.clip_by_global_norm(train_cfg_block.get("max_grad_norm", 1.0)),
                optax.adam(FINETUNE_LR),
            )
            current_train_state = TrainState.create(
                apply_fn=current_train_state.apply_fn,
                params=current_train_state.params,
                tx=ft_tx,
            )
            current_train_state = current_train_state.replace(step=total_epoch)
            finetune_lr_applied = True
            print(f"🔧 Auto fine-tune: switched to constant LR={FINETUNE_LR} at epoch {total_epoch}")
        
        # During finetune: tighter clip_eps and lower entropy for stability
        if finetune_lr_applied:
            annealed_ent_coef = 0.001
            print(f"📉 Finetune entropy coef override: {annealed_ent_coef:.4f}")
        
        finetune_clip_eps = 0.1 if finetune_lr_applied else train_cfg_block.get("clip_eps", 0.2)
        # Scale num_minibatches with num_envs to keep minibatch size constant
        # at the 100-envs reference (e.g. 10 envs -> 2 minibatches, 1000 -> 200).
        base_minibatches = train_cfg_block.get("num_minibatches", 20)
        total_samples = num_envs * env.max_steps_in_episode
        scaled_minibatches = max(1, base_minibatches * num_envs // 100)
        # Ensure num_minibatches divides total_samples exactly
        while scaled_minibatches > 1 and total_samples % scaled_minibatches != 0:
            scaled_minibatches -= 1
        ppo_config = ppo.Config(
            num_minibatches=scaled_minibatches,
            ent_coef=annealed_ent_coef,
            gamma=train_cfg_block.get("gamma", 0.99),
            gae_lambda=train_cfg_block.get("gae_lambda", 0.95),
            update_epochs=train_cfg_block.get("update_epochs", 4),
            clip_eps=finetune_clip_eps,
            vf_coef=train_cfg_block.get("vf_coef", 0.5),
        )
        res_dict = ppo.train(
            env=env,
            train_state=current_train_state,
            num_epochs=block_epochs,
            num_steps_per_epoch=env.max_steps_in_episode,
            num_envs=num_envs,
            key=key_block,
            config=ppo_config
        )
        block_train_seconds = time.time() - block_time_start
        block_compile_seconds = float(res_dict.get("compile_seconds", 0.0))
        block_exec_seconds = float(res_dict.get("exec_seconds", 0.0))

        current_train_state = res_dict['runner_state'].train_state
        reward_components = res_dict['reward_components']

        # Export per-epoch train losses (wide format: epoch, comp1_loss, ..., total_loss)
        reward_components_np = {
            comp_name: np.asarray(jax.device_get(comp_values)).reshape(-1)
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
        checkpoint_eval_dir = run_dir / f"eval_epoch_{checkpoint_step}"
        checkpoint_eval_dir.mkdir(exist_ok=True)
        
        # Create policy function for evaluation
        def policy_current(obs, key):
            pi = current_train_state.apply_fn(current_train_state.params, obs).pi
            return pi.sample(seed=key)
        
        # Split key_eval to ensure different rollouts at each checkpoint
        eval_time_start = time.time()
        key_eval, key_eval_checkpoint = jax.random.split(key_eval)
        transitions_eval = get_rollouts(env_eval, policy_current, num_rollouts=num_rollouts, key=key_eval_checkpoint)
        
        eval_stats = compute_and_print_gap_crossing_statistics(transitions_eval, env_eval, time_start)
        eval_seconds = time.time() - eval_time_start

        rollout_npz_path = paper_paths["rollouts_dir"] / f"{artifact_prefix}_eval_epoch_{checkpoint_step}.npz"
        save_eval_rollouts_npz(
            transitions_eval,
            rollout_npz_path,
            metadata={
                "checkpoint_epoch": int(checkpoint_step),
                "num_rollouts": int(num_rollouts),
                "algo": "ppo",
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
    key_init, key_ppo, key_eval = jax.random.split(key, 3)

    
    drone_path = FLIGHTNING_PATH + "/objects/quadrotor_files/example_quad.yaml"

    dyn_config = {
        "high_fidelity_forward": False,
        "forward_residual": False,
        "backprop_residual": False,
    }
    
    env, train_state = create_train_state(drone_path, key_init, num_epochs=args.epochs, dyn_config=dyn_config, num_envs=args.num_envs)
    
    
    # --- Resume training from an existing run/checkpoint (exact continuation) ---
    if args.resume is not None:
        # For finetune mode, load the exact checkpoint requested (don't auto-advance to latest)
        is_finetune = args.finetune_lr is not None
        train_state = load_model(train_state, args.resume, resume=not is_finetune)
        
        current_epoch = int(train_state.step)
        total_epochs = current_epoch + args.continue_epochs
        print(f"Resuming from epoch {current_epoch}, training until epoch {total_epochs}")
        
        # Fine-tune mode: replace optimizer with constant LR, keep policy weights
        if args.finetune_lr is not None:
            ft_lr = args.finetune_lr
            train_cfg_ft = load_training_config()
            ft_tx = optax.chain(
                optax.clip_by_global_norm(train_cfg_ft.get("max_grad_norm", 1.0)),
                optax.adam(ft_lr),
            )
            train_state = TrainState.create(
                apply_fn=train_state.apply_fn,
                params=train_state.params,
                tx=ft_tx,
            )
            # Restore the step counter so curriculum/entropy continue correctly
            train_state = train_state.replace(step=current_epoch)
            print(f"🔧 Fine-tune mode: constant LR={ft_lr}, optimizer state reset")
        
        # Create new env with updated epoch range
        env, _ = create_train_state(drone_path, key_init, num_epochs=total_epochs, 
                                     dyn_config=dyn_config, initial_epoch=current_epoch, num_envs=args.num_envs)
        
        train_state = train_model(env, train_state, key_ppo, run_dir, args.continue_epochs, 
                                  args.rollouts, key_eval, dyn_config=dyn_config, drone_path=drone_path, num_envs=args.num_envs, is_finetune=is_finetune)
    
    # --- Train a new model ---
    elif args.train:
        train_state = train_model(env, train_state, key_ppo, run_dir, args.epochs, 
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
        def policy_load(obs, key):
            pi = train_state.apply_fn(train_state.params, obs).pi
            return pi.sample(seed=key)
        
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

        artifact_prefix = f"ppo_{get_config_suffix()}_envs{args.num_envs}"
        paper_paths = init_paper_logging(run_dir, file_prefix=artifact_prefix)
        rollout_npz_path = paper_paths["rollouts_dir"] / f"{artifact_prefix}_evaluation_loaded_model.npz"
        save_eval_rollouts_npz(
            transitions_eval_load,
            rollout_npz_path,
            metadata={
                "mode": "load",
                "checkpoint_path": args.load,
                "num_rollouts": int(args.rollouts),
                "algo": "ppo",
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
    parser.add_argument("-l", "--load", type=str, default=None, help="Path to checkpoint to load (e.g. 'runs/.../checkpoints/checkpoint_295' or 'runs/.../checkpoints/' for latest).")
    parser.add_argument("-r", "--rollouts", type=int, default=100, help="Number of rollouts for evaluation.")
    parser.add_argument("-e", "--epochs", type=int, default=20000, help="Number of training epochs.")
    parser.add_argument("-s", "--seed", type=int, default=0, help="Random seed for reproducibility (default: 0).")
    
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from (e.g. 'runs/.../checkpoints/checkpoint_295' or 'runs/.../checkpoints/' for latest).")
    parser.add_argument("--continue-epochs", type=int, default=1000, help="Number of additional epochs to train when resuming.")
    parser.add_argument("--finetune-lr", type=float, default=None, help="Fine-tune with constant LR (resets optimizer state, keeps policy weights).")
    parser.add_argument("--num-envs", type=int, default=100, help="Number of parallel environments for PPO training.")
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
