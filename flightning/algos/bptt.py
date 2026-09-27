from functools import partial
from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp
from flax.struct import PyTreeNode
from flax.training.train_state import TrainState

from flightning.envs.env_base import Env, EnvState
from flightning.envs.wrappers import LogWrapper, VecEnv


class TrajectoryState(PyTreeNode):
    reward: jnp.array


def progress_callback_host(episode_loss):
    episode, loss = episode_loss
    print(f"Episode: {episode}, Loss: {loss:.2f}")


NUM_EPOCHS_PER_CALLBACK = 10


def progress_callback(episode, loss):
    jax.lax.cond(
        pred=episode % NUM_EPOCHS_PER_CALLBACK == 0,
        true_fun=lambda eps_lss: jax.debug.callback(
            progress_callback_host, eps_lss
        ),
        false_fun=lambda eps_lss: None,
        operand=(episode, loss),
    )


def grad_callback_host(episode_grad):
    episode, grad = episode_grad
    print(f"Episode: {episode}, Grad max: {grad:.4f}")


def grad_callback(episode, grad_norm):
    jax.lax.cond(
        pred=episode % NUM_EPOCHS_PER_CALLBACK == 0,
        true_fun=lambda eps_lss: jax.debug.callback(
            grad_callback_host, eps_lss
        ),
        false_fun=lambda eps_lss: None,
        operand=(episode, grad_norm),
    )


class RunnerState(NamedTuple):
    train_state: TrainState
    env_state: EnvState
    last_obs: jax.Array
    key: chex.PRNGKey
    epoch_idx: int


def make_train(
    env: Env,
    num_epochs: int,
    num_steps_per_epoch: int,
    num_envs: int,
):
    """Create a JIT-compiled training function that can be called multiple times
    without recompilation. Returns (train_fn, wrapped_env).
    
    Usage:
        train_fn, wrapped_env = make_train(env, num_epochs, num_steps_per_epoch, num_envs)
        # In loop:
        key_reset = jax.random.split(key, num_envs)
        env_state, obs = wrapped_env.reset(key_reset, None)
        runner_state = RunnerState(train_state, env_state, obs, key, epoch_idx=0)
        result = train_fn(runner_state)
    """
    env = LogWrapper(env)
    env = VecEnv(env)

    def _train(runner_state: RunnerState):
        def epoch_fn(epoch_state: RunnerState, _unused):
            # ============================================================
            # RESET ENVIRONMENT WITH NEW RANDOMIZATION AT EACH EPOCH
            # ============================================================
            key, reset_key = jax.random.split(epoch_state.key)
            key_reset = jax.random.split(reset_key, num_envs)
            env_state, obs = env.reset(key_reset, None)
            
            # Update epoch_idx in the newly reset environment
            # Add env.initial_epoch offset to handle resumed training correctly
            absolute_epoch = epoch_state.epoch_idx + env.initial_epoch
            inner_env_state = env_state.env_state  # unwrap LogWrapper
            updated_inner = inner_env_state.replace(
                current_epoch=jnp.full_like(inner_env_state.current_epoch, absolute_epoch)
            )
            updated_env_state = env_state.replace(env_state=updated_inner)
            
            # Update runner_state with new environment and key
            epoch_state = epoch_state._replace(
                env_state=updated_env_state,
                last_obs=obs,
                key=key
            )
            # ============================================================

            @partial(jax.value_and_grad, has_aux=True)
            def loss_fn(params, runner_state: RunnerState):

                def rollout(runner_state: RunnerState):

                    # Get reward component keys dynamically from the environment
                    reward_components_keys = env.get_reward_component_keys()
                  

                    components_acc = {comp: 0.0 for comp in reward_components_keys}

                    def step_fn(carry, _unsused):
                        old_runner_state, components_acc = carry
                        # extract states and obs
                        train_state, env_state, last_obs, key, epoch_idx = (
                            old_runner_state
                        )

                        # get action
                        action = train_state.apply_fn(params, last_obs)

                        # env step
                        key, key_ = jax.random.split(key)
                        key_step = jax.random.split(key_, num_envs)
                        (
                            env_state,
                            obs,
                            reward,
                            _terminated,
                            _truncated,
                            info,
                        ) = env.step(env_state, action, key_step)
                        runner_state = RunnerState(
                            train_state, env_state, obs, key, epoch_idx
                        )

                        reward_components = info["reward_components"]

                        for comp in components_acc.keys():
                            components_acc[comp] += jnp.sum(reward_components[comp] / num_envs)
        
                    

                        return (
                            (runner_state, components_acc),
                            TrajectoryState(reward=reward),
                        )

                    (runner_state, components_acc), trajectory = jax.lax.scan(
                        step_fn, (runner_state, components_acc), None, num_steps_per_epoch
                    )
                    return runner_state, trajectory, components_acc

                # collect data
                runner_state, trajectory, components_acc = rollout(runner_state)
                loss = -trajectory.reward.sum() / num_envs
                return loss, (runner_state, components_acc) # NOTE added to capture info

            # compute reward
            train_state = epoch_state.train_state
            (loss, (rollout_final_state, components_acc)), grad = loss_fn(
                train_state.params, epoch_state
            )
            # rollout_final_state is discarded - we reset env at each epoch

            clip_gradient = True
            def clip_by_global_norm(grad, max_norm):
                leaves = jax.tree_util.tree_leaves(grad)
                squared = [jnp.sum(jnp.square(leaf)) for leaf in leaves]
                global_norm = jnp.sqrt(jnp.sum(jnp.array(squared)))
                clip_coef = jnp.minimum(1.0, max_norm / (global_norm + 1e-6))
                clipped_grad = jax.tree_util.tree_map(
                    lambda g: g * clip_coef, grad
                )
                return clipped_grad, global_norm
            if clip_gradient:
                clipped_grad, grad_norm = clip_by_global_norm(grad, max_norm=300.0)
                train_state = train_state.apply_gradients(grads=clipped_grad)
                leaves = jax.tree_util.tree_leaves(clipped_grad)
            # update params
            else:
                train_state = train_state.apply_gradients(grads=grad)

            # calc stats on grad
                leaves = jax.tree_util.tree_leaves(grad)
            flattened_leaves = [jnp.ravel(leaf) for leaf in leaves]
            grad_vec = jnp.concatenate(flattened_leaves)
            grad_max = jnp.max(jnp.abs(grad_vec))

            progress_callback(epoch_state.epoch_idx, loss)
            grad_callback(epoch_state.epoch_idx, grad_max)
            epoch_state = epoch_state._replace(
                train_state=train_state, epoch_idx=epoch_state.epoch_idx + 1
            )

            return epoch_state, (loss, components_acc) # NOTE added to capture info

        # run epochs
        runner_state_final, (losses, components_acc_list) = jax.lax.scan(
            epoch_fn, runner_state, None, num_epochs
        )

        
        components_arrays = {comp: components_acc_list[comp] for comp in components_acc_list.keys()}

        return {"runner_state": runner_state_final, "metrics": losses, "reward_components": components_arrays} # NOTE added to capture info

    return jax.jit(_train), env


def train(
    env: Env,
    train_state: TrainState,
    num_epochs: int,
    num_steps_per_epoch: int,
    num_envs: int,
    key: chex.PRNGKey,
):

    train_fn, wrapped_env = make_train(env, num_epochs, num_steps_per_epoch, num_envs)

    # intialize environments
    key, key_ = jax.random.split(key)
    key_reset = jax.random.split(key_, num_envs)
    # Initialize with epoch_idx=0, environment will handle current_epoch
    env_state, obs = wrapped_env.reset(key_reset, None)
    runner_state = RunnerState(train_state, env_state, obs, key, epoch_idx=0)

    return train_fn(runner_state)


if __name__ == "__main__":
    # Example usage
    from flax.linen import Dense
    from flightning.envs import HoveringStateEnv
    import optax

    x = jnp.zeros(15)
    model = Dense(4)
    params = model.init(jax.random.key(0), x)
    tx = optax.adam(1e-3)

    train_state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )

    env = HoveringStateEnv()

    res_dict = train(
        env,
        train_state,
        num_envs=10,
        num_epochs=10,
        num_steps_per_epoch=10,
        key=jax.random.key(0),
    )

    loss = res_dict["metrics"]
    print(f"Final loss: {loss[-1]}")
