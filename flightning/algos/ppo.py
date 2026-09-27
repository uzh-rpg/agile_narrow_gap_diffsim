"""
PPO implementation in JAX based on
https://github.com/luchris429/purejaxrl/blob/main/purejaxrl/ppo_continuous_action.py
Modified to work with flightning environment API
"""

import time
from functools import partial
from typing import NamedTuple
from typing import Tuple

import chex
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState

from flightning.envs.env_base import Env, EnvState
from flightning.envs.wrappers import LogWrapper, VecEnv


class Config(NamedTuple):
    gamma: float = 0.99
    gae_lambda: float = 0.95
    update_epochs: int = 4
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.0
    num_minibatches: int = 20  # 32
    logging_freq: int = 10
    logging: bool = True


class PPOSample(NamedTuple):
    done: jax.Array
    action: jax.Array
    value: jax.Array
    reward: jax.Array
    log_prob: jax.Array
    obs: jax.Array
    info: jax.Array


class RunnerState(NamedTuple):
    train_state: TrainState
    env_state: EnvState
    last_obs: jax.Array
    key: chex.PRNGKey
    epoch_idx: int


class UpdateState(NamedTuple):
    train_state: TrainState
    trajectory: PPOSample
    advantages: jax.Array
    targets: jax.Array
    key: chex.PRNGKey


class LossInfo(NamedTuple):
    value_loss: jax.Array
    actor_loss: jax.Array
    entropy: jax.Array
    total_loss: jax.Array


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


def train(
    env: Env,
    train_state: TrainState,
    num_epochs: int,
    num_steps_per_epoch: int,
    num_envs: int,
    key: chex.PRNGKey,
    config: Config = Config(),
):

    env = LogWrapper(env)
    env = VecEnv(env)
    # env = RunningMeanVarWrapper(env)

    MINIBATCH_SIZE = num_envs * num_steps_per_epoch // config.num_minibatches

    def _train(runner_state: RunnerState):
        def epoch_fn(runner_state: RunnerState, _unused):
            # ============================================================
            # RESET ENVIRONMENT WITH NEW RANDOMIZATION AT EACH EPOCH
            # ============================================================
            key, reset_key = jax.random.split(runner_state.key)
            key_reset = jax.random.split(reset_key, num_envs)
            env_state, obs = env.reset(key_reset, None)
            
            # Update epoch_idx in the newly reset environment
            # Add env.initial_epoch offset to handle resumed training correctly
            absolute_epoch = runner_state.epoch_idx + env.initial_epoch
            inner_env_state = env_state.env_state  # unwrap LogWrapper
            updated_inner = inner_env_state.replace(
                current_epoch=jnp.full_like(inner_env_state.current_epoch, absolute_epoch)
            )
            updated_env_state = env_state.replace(env_state=updated_inner)
            
            # Update runner_state with new environment and key
            runner_state = runner_state._replace(
                env_state=updated_env_state,
                last_obs=obs,
                key=key
            )
            # ============================================================

            def step_fn(runner_state: RunnerState, _unused):
                train_state, env_state, last_obs, key, epoch_idx = runner_state

                # SELECT ACTION
                key, key_ = jax.random.split(key)
                pi, value = train_state.apply_fn(train_state.params, last_obs)
                action = pi.sample(seed=key_)
                log_prob = pi.log_prob(action)

                # STEP ENV
                key, key_ = jax.random.split(key)
                key_step = jax.random.split(key_, num_envs)
                env_state, obs, reward, terminated, truncated, info = env.step(
                    env_state, action, key_step
                )
                done = jnp.logical_or(terminated, truncated)
                # Keep negative reward as-is from environment (cost-based)
                transition = PPOSample(
                    done, action, value, reward, log_prob, last_obs, info
                )
                runner_state = RunnerState(
                    train_state, env_state, obs, key, epoch_idx
                )
                return runner_state, transition

            runner_state, samples = jax.lax.scan(
                step_fn, runner_state, None, num_steps_per_epoch
            )

            # calculate advantage
            train_state, env_state, last_obs, key, epoch_idx = runner_state
            _, last_value = train_state.apply_fn(train_state.params, last_obs)

            def calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, sample):
                    gae, next_value = gae_and_next_value
                    done = sample.done
                    value = sample.value
                    reward = sample.reward

                    # TODO: handle difference between truncated and terminated
                    delta = (
                        reward + config.gamma * next_value * (1 - done) - value
                    )
                    gae = (
                        delta
                        + config.gamma * config.gae_lambda * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = calculate_gae(samples, last_value)

            # update network
            def update_epoch(update_state: UpdateState, _unused):
                def update_minibatch(
                    train_state: TrainState,
                    batch_info: Tuple[PPOSample, jax.Array, jax.Array],
                ):
                    traj_batch, advantages, targets = batch_info

                    @partial(jax.value_and_grad, has_aux=True)
                    def loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        pi, value = train_state.apply_fn(
                            params, traj_batch.obs
                        )
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config.clip_eps, config.clip_eps)
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(
                            value_pred_clipped - targets
                        )
                        value_loss = (
                            0.5
                            * jnp.maximum(
                                value_losses, value_losses_clipped
                            ).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config.clip_eps,
                                1.0 + config.clip_eps,
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config.vf_coef * value_loss
                            - config.ent_coef * entropy
                        )

                        loss_info = LossInfo(
                            value_loss, loss_actor, entropy, total_loss
                        )

                        return total_loss, loss_info

                    (_, loss_info), grads = loss_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, loss_info

                train_state, traj_batch, advantages, targets, key = (
                    update_state
                )
                key, key_ = jax.random.split(key)
                batch_size = MINIBATCH_SIZE * config.num_minibatches
                assert (
                    batch_size == num_steps_per_epoch * num_envs
                ), "batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(key_, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config.num_minibatches, -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                train_state, loss_info = jax.lax.scan(
                    update_minibatch, train_state, minibatches
                )
                update_state = UpdateState(
                    train_state, traj_batch, advantages, targets, key
                )
                return update_state, loss_info

            update_state = UpdateState(
                train_state, samples, advantages, targets, key
            )
            update_state, loss_info = jax.lax.scan(
                update_epoch, update_state, None, config.update_epochs
            )

            train_state = update_state.train_state
            metric = samples.info
            # --- Aggregate reward components over time/envs for this epoch ---
            # samples.info["reward_components"] is expected to be a pytree/dict mapping
            # component_name -> array shaped (num_steps, num_envs)
            def _aggregate_reward_components(info):
                # Sum over steps, mean over envs (same as BPTT) so that
                # sum(components) == episodic_loss == -reward.sum()/num_envs
                rc = info.get("reward_components", None)
                if rc is None:
                    return {}
                # rc is a dict: comp -> array (num_steps, num_envs)
                return jax.tree_map(lambda x: jnp.sum(x) / num_envs, rc)

            reward_components_epoch = _aggregate_reward_components(metric)
            # Attach aggregated reward components to metric so they get stacked across epochs by lax.scan
            # Use a shallow copy to avoid mutating original metric in-place
            metric = dict(metric)
            metric["reward_components_epoch"] = reward_components_epoch
            key = update_state.key

            # Calculate loss same way as BPPT: negate negative rewards to get positive loss
            episodic_loss = -samples.reward.sum() / num_envs
            
            # Also compute average PPO training loss for logging (take mean since it's accumulated from minibatches)
            avg_ppo_loss = jnp.mean(loss_info.total_loss)

            def callback(info_tuple):
                episodic_loss, ppo_loss, epoch_idx = info_tuple
                print(
                    f"Epoch: {epoch_idx}, "
                    f"Episodic Loss: {episodic_loss:.2f}, "
                    f"PPO Loss: {ppo_loss:.4f}"
                )

            jax.lax.cond(
                pred=jnp.logical_and(
                    config.logging, epoch_idx % config.logging_freq == 0
                ),
                true_fun=partial(jax.debug.callback, callback),
                false_fun=lambda _: None,
                operand=(episodic_loss, avg_ppo_loss, runner_state.epoch_idx),
            )

            runner_state = RunnerState(
                train_state, env_state, last_obs, key, epoch_idx + 1
            )
            return runner_state, metric

        # run epochs
        runner_state, metric = jax.lax.scan(
            epoch_fn, runner_state, None, num_epochs
        )

        # Extract reward components stacked across epochs if available
        reward_components = {}
        if "reward_components_epoch" in metric:
            reward_components = metric["reward_components_epoch"]

        return {
            "runner_state": runner_state,
            "metrics": metric,
            "reward_components": reward_components,
        }

    # intialize environments
    key, key_ = jax.random.split(key)
    reset_rng = jax.random.split(key_, num_envs)
    env_state, obs = env.reset(reset_rng, None)
    runner_state = RunnerState(train_state, env_state, obs, key, epoch_idx=0)

    # Separately time JIT compile vs execution (instrumentation only — does
    # not change numerics).
    jitted = jax.jit(_train)
    _t0 = time.time()
    compiled = jitted.lower(runner_state).compile()
    compile_seconds = time.time() - _t0
    _t1 = time.time()
    result = compiled(runner_state)
    jax.block_until_ready(result["runner_state"].train_state.params)
    exec_seconds = time.time() - _t1
    result["compile_seconds"] = float(compile_seconds)
    result["exec_seconds"] = float(exec_seconds)
    return result


if __name__ == "__main__":

    # Example usage
    from flightning.modules.mlp import ActorCriticPPO
    from flightning.envs import HoveringStateEnv
    from flightning.envs.wrappers import NormalizeActionWrapper
    import optax

    env = HoveringStateEnv()
    env = NormalizeActionWrapper(env)

    num_actions = env.action_space.shape[0]
    num_obs = env.observation_space.shape[0]

    model = ActorCriticPPO([num_obs, 128, num_actions])
    params = model.initialize(jax.random.key(0))
    tx = optax.adam(3e-4)

    train_state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )

    train_res = train(
        env,
        train_state,
        num_epochs=10,
        num_steps_per_epoch=10,
        num_envs=10,
        key=jax.random.key(1),
        config=Config(num_minibatches=1, update_epochs=10),
    )
