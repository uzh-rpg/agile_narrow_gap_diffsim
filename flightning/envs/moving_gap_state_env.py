import functools
from typing import Optional

import chex
import jax
import jax_dataclasses as jdc
import numpy as np
from jax import numpy as jnp

from flightning.objects import Quadrotor, QuadrotorState, WorldBox
from flightning.utils import math as math_utils
from flightning.utils import spaces
from flightning.utils.pytrees import stack_pytrees
from flightning.utils.math import smooth_l1
from flightning.utils.random import random_rotation
import flightning.envs.env_base as env_base
from flightning.envs.env_base import EnvTransition
from flightning.envs.static_gap_state_env import StaticGapStateEnv, EnvState as StaticEnvState


@jdc.pytree_dataclass
class EnvState(StaticEnvState):
    """Extended state for moving gap environment with pendulum dynamics."""
    pend_theta: float        # current angle of the pendulum (rad)
    pend_theta_dot: float    # angular velocity of the pendulum (rad/s)
    pend_L: float            # length of the pendulum (m)
    pivot: jax.Array # pivot point of the pendulum (world frame)
    gap_center_prev: jax.Array  # previous gap center position for velocity estimation
    # camera_pitch: float = 0.0  # camera pitch angle (for obs, not used in state-based env)

class MovingGapStateEnv(StaticGapStateEnv):
    """Environment where a drone must fly through a moving gap (pendulum) and reach a goal."""

    def __init__(
        self,
        *,
        max_steps_in_episode=300,
        dt=0.02,
        delay=0.04,
        drone_path=None,
        dyn_config=None,  # Dynamics configuration for residual model
        res_model_params=None,  # Residual model parameters (loaded from file)
        num_last_quad_states=10,
        margin=jnp.array([2.0, 2.0, 1.0]),
        start_distance: float = 2.0, 
        start_jitter: float = 1.0,
        post_gap_distance: float = 2.0,
        gap_dims: jax.Array = jnp.array([0.80, 0.25]),
        drone_radius: float = 0.38 / 2 ,
        drone_height_radius: float = 0.10 / 2,
        world_box_min: Optional[jax.Array] = jnp.array([-10.0, -10.0, 0.0]),
        world_box_max: Optional[jax.Array] = jnp.array([10.0, 10.0, 10.0]),
        training: bool = True,
        num_epochs: int = 1000,
        max_rot_theta_degrees: float = 45.0,
        velocity_std: float = 0.1,
        omega_std: float = 0.1,
        initial_epoch: int = 0,
        algo: str = None,  # Algorithm type: "bptt" or "ppo"
    ):
        # Call parent class constructor
        super().__init__(
            max_steps_in_episode=max_steps_in_episode,
            dt=dt,
            delay=delay,
            drone_path=drone_path,
            dyn_config=dyn_config,  # Moving gap doesn't use residual dynamics
            res_model_params=res_model_params,
            num_last_quad_states=num_last_quad_states,
            margin=margin,
            start_distance=start_distance,
            start_jitter=start_jitter,
            post_gap_distance=post_gap_distance,
            gap_dims=gap_dims,
            drone_radius=drone_radius,
            drone_height_radius=drone_height_radius,
            world_box_min=world_box_min,
            world_box_max=world_box_max,
            training=training,
            num_epochs=num_epochs,
            max_rot_theta_degrees=max_rot_theta_degrees,
            velocity_std=velocity_std,
            omega_std=omega_std,
            initial_epoch=initial_epoch,
            algo=algo,
        )
        
        # Override reward weights from config
        self._load_reward_weights("moving_state")

    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(self, key, state: Optional[EnvState] = None) -> tuple[EnvState, jax.Array]:

        key_gapc, key_gn, key_start, key_dr, key_rot, key_vel, key_omega, key_delay = jax.random.split(key, 8)

        # ------------------------------------------------------------
        # Pendulum initialization
        # ------------------------------------------------------------
        pend_L = jax.random.uniform(key_gapc, minval=0.5, maxval=1.5)

        # Gap normal (fixed in XY plane as before)
        angle = jnp.pi / 2
        gap_normal = jnp.array([jnp.cos(angle), jnp.sin(angle), 0.0])
        gap_normal = gap_normal / (jnp.linalg.norm(gap_normal) + 1e-9)

        # ------------------------------------------------------------
        # Random oscillation amplitude A in [20°, max_angle]
        # ------------------------------------------------------------
        min_angle_deg = 20.0
        max_angle_deg = self.max_rot_theta_radians / jnp.pi * 180.0

        key_A, key_theta, key_sign = jax.random.split(key_rot, 3)

        A_deg = min_angle_deg + (max_angle_deg - min_angle_deg) * jax.random.uniform(key_A)
        A = jnp.deg2rad(A_deg)

        # Random initial theta uniformly in [-A, A]
        pend_theta = jax.random.uniform(key_theta, minval=-A, maxval=A)

        # Compute velocity from energy conservation (nonlinear pendulum)
        g = 9.81
        L = pend_L

        cos_term = jnp.cos(pend_theta) - jnp.cos(A)
        cos_term = jnp.maximum(cos_term, 0.0)  # numerical safety shouldn't be needed

        theta_dot_mag = jnp.sqrt(2.0 * g / L * cos_term)

        # Randomize direction of motion
        sign = jax.lax.select(
            jax.random.uniform(key_sign) < 0.5,
            1.0,
            -1.0
        )

        pend_theta_dot = sign * theta_dot_mag

        # ------------------------------------------------------------
        # Geometry
        # ------------------------------------------------------------
        gap_h = self.gap_dims[1]
        offset_len = pend_L + gap_h / 2.0

        pivot = jnp.array([0.0, 0.0, 5.0 + offset_len])

        vertical_axis = jnp.array([0.0, 0.0, 1.0])
        lateral_axis = jnp.cross(vertical_axis, gap_normal)
        lateral_axis = lateral_axis / (jnp.linalg.norm(lateral_axis) + 1e-9)

        rod_dir = jnp.sin(pend_theta) * lateral_axis - jnp.cos(pend_theta) * vertical_axis
        rod_dir = rod_dir / (jnp.linalg.norm(rod_dir) + 1e-9)

        gap_center = pivot + offset_len * rod_dir

        v_axis = -rod_dir
        u_axis = jnp.cross(v_axis, gap_normal)
        u_axis = u_axis / (jnp.linalg.norm(u_axis) + 1e-9)

        gap_R = jnp.stack([gap_normal, u_axis, v_axis], axis=1)

        current_epoch = 0 if state is None else state.current_epoch

        # ------------------------------------------------------------
        # Goal (computed at theta = 0 reference)
        # ------------------------------------------------------------
        zero_gap_center = pivot - offset_len * jnp.array([0.0, 0.0, 1.0])
        goal = zero_gap_center + self.post_gap_distance * gap_normal
        goal = jnp.clip(goal, self.world_box.min + 0.1, self.world_box.max - 0.1)

        # ------------------------------------------------------------
        # Start position
        # ------------------------------------------------------------
        key_start1, key_start2, key_start3 = jax.random.split(key_start, 3)

        dist_behind_plane = self.start_distance + self.start_jitter * jax.random.uniform(
            key_start1, minval=-1.0, maxval=1.0
        )

        jitter_vertical = self.start_jitter * jax.random.uniform(
            key_start2, minval=-1.0, maxval=1.0
        )

        jitter_lateral = self.start_jitter * jax.random.uniform(
            key_start3, minval=-1.0, maxval=1.0
        )

        lateral_axis_start = jnp.cross(gap_normal, vertical_axis)
        lateral_axis_start = lateral_axis_start / (jnp.linalg.norm(lateral_axis_start) + 1e-9)

        start = (
            zero_gap_center
            - dist_behind_plane * gap_normal
            + jitter_vertical * vertical_axis
            + jitter_lateral * lateral_axis_start
        )

        start = jnp.clip(start, self.world_box.min + 0.1, self.world_box.max - 0.1)

        # ------------------------------------------------------------
        # Drone orientation (yaw toward gap + randomization)
        # ------------------------------------------------------------
        vec_to_gap = zero_gap_center - start
        yaw_to_gap = jnp.arctan2(vec_to_gap[1], vec_to_gap[0])

        yaw_randomization_max = jnp.deg2rad(40.0)
        key_yaw = jax.random.split(key_dr)[0]
        yaw_offset = jax.random.uniform(
            key_yaw,
            minval=-yaw_randomization_max,
            maxval=yaw_randomization_max
        )

        yaw = yaw_to_gap + yaw_offset

        cos_yaw = jnp.cos(yaw)
        sin_yaw = jnp.sin(yaw)

        forward_dir = jnp.array([cos_yaw, sin_yaw, 0.0])
        right_dir = jnp.array([-sin_yaw, cos_yaw, 0.0])
        up_dir = jnp.array([0.0, 0.0, 1.0])

        R = jnp.stack([forward_dir, right_dir, up_dir], axis=1)

        # ------------------------------------------------------------
        # Drone velocity
        # ------------------------------------------------------------
        v_noise = jax.random.normal(key_vel, shape=(3,)) * self.velocity_std
        omega_noise = jax.random.normal(key_omega, shape=(3,)) * self.omega_std

        quadrotor_state = self.quadrotor.create_state(
            p=start,
            R=R,
            v=v_noise,
            omega=omega_noise,
            dr_key=key_dr
        )

        last_actions = jnp.tile(self.hovering_action, (self.num_last_actions, 1))
        last_quadrotor_states = [quadrotor_state] * self.num_last_quad_states
        last_quadrotor_states = stack_pytrees(last_quadrotor_states)

        last_gap_centers = jnp.tile(gap_center, (self.num_last_quad_states, 1))
        last_gap_Rs = jnp.tile(gap_R, (self.num_last_quad_states, 1, 1)) #! TODO check

        collision_vector = self._get_collision_vector(
            p=quadrotor_state.p,
            R=quadrotor_state.R,
            gap_center=gap_center,
            gap_dims=self.gap_dims,
            gap_u_axis=u_axis,
            gap_v_axis=v_axis,
            margin_horizontal=self.margin_gap_collision_horizontal,
            margin_vertical=self.margin_gap_collision_vertical,
        )

        # Randomize delay for this episode
        delay = jax.random.uniform(key_delay, minval=self.delay_min, maxval=self.delay_max)

        state = EnvState(
            time=0.0,
            step_idx=0,
            quadrotor_state=quadrotor_state,
            last_actions=last_actions,
            last_quadrotor_states=last_quadrotor_states,
            delay=delay,
            gap_center=gap_center,
            gap_normal=gap_normal,
            gap_dims=self.gap_dims,
            gap_u_axis=u_axis,
            gap_v_axis=v_axis,
            gap_R=gap_R,
            goal=goal,
            current_epoch=current_epoch,
            collision_vector=collision_vector,
            pend_theta=pend_theta,
            pend_theta_dot=pend_theta_dot,
            pend_L=pend_L,
            pivot=pivot,
            gap_center_prev=gap_center,
            projected_corners=jnp.zeros((4, 4, 2)),  # Placeholder, not used in state-based env
            projected_valid_flags=jnp.zeros((4, 4,)),  # Placeholder
            last_gap_centers=last_gap_centers,
            last_gap_Rs=last_gap_Rs
        )

        obs = self._get_obs(state)

        return state, obs

    

    



    # ------------------------------------------------------------
    # STEP
    # ------------------------------------------------------------
    @functools.partial(jax.jit, static_argnums=(0,))
    def _step(self, state: EnvState, action: jax.Array, key: chex.PRNGKey) -> EnvTransition:

        # remove the oldest action and add the current one
        action = jnp.clip(action, self.action_space.low, self.action_space.high)


        # add action to last actions
        last_actions = jnp.roll(state.last_actions, shift=-1, axis=0) # shift left all the vectore -> last_actions[-1] is the oldest action (to be replaced)
        last_actions = last_actions.at[-1].set(action) # replace the oldest with the most recent action

        # Use delay from state (randomized at reset)
        delay = state.delay
        
        # Calculate which action to use based on actual delay
        # Example: delay=50ms, dt=16ms → delay is 3.125 timesteps ago
        # - num_delay_steps = 3 (complete timesteps)
        # - We need commands from 4 and 3 steps ago
        num_delay_steps = jnp.floor(delay / self.dt).astype(int)
        
        # Indices in queue (last_actions[-1] is most recent, i.e., 0 steps ago)
        action_idx_1 = -(num_delay_steps + 2)
        action_idx_2 = -(num_delay_steps + 1)
        
        dt_1 = delay % self.dt 
        action_1 = last_actions[action_idx_1]
        f_1, omega_1 = action_1[0], action_1[1:]  # thrust and angular rates (wx, wy, wz)
        quadrotor_state = self.quadrotor.step(
            state.quadrotor_state, f_1, omega_1, dt_1
        )

        # 2nd step
        dt_2 = self.dt - dt_1
        action_2 = last_actions[action_idx_2]
        f_2, omega_2 = action_2[0], action_2[1:]
        quadrotor_state = self.quadrotor.step(
            quadrotor_state, f_2, omega_2, dt_2
        )
            
        # remove the oldest quadrotor state and add the current one
        last_quad_states_rolled = jax.tree_map(lambda arr: jnp.roll(arr, shift=-1, axis=0), state.last_quadrotor_states)
        last_quadrotor_states = jax.tree_map(
            lambda hist_arr, curr_val: hist_arr.at[self.num_last_quad_states - 1].set(curr_val),
            last_quad_states_rolled,
            quadrotor_state
        )
        
        # --- Pendulum dynamics (update BEFORE collision checks) ---
        g = 9.81
        L = state.pend_L

        theta = state.pend_theta
        theta_dot = state.pend_theta_dot

        theta_dd = - (g / L) * jnp.sin(theta)   # NON-linear pendulum
        theta_dot = theta_dot + theta_dd * self.dt
        theta = theta + theta_dot * self.dt



        # --- Geometry update ---
        gap_h = self.gap_dims[1]
        offset_len = L + gap_h / 2.0

        # Retrieve gap_normal from state (set during reset, remains constant during episode)
        gap_normal = state.gap_normal
        
        # Vertical axis
        vertical_axis = jnp.array([0.0, 0.0, 1.0])
        lateral_axis = jnp.cross(vertical_axis, gap_normal)
        lateral_axis = lateral_axis / (jnp.linalg.norm(lateral_axis) + 1e-9)
        # Pendulum rod oscillates in the plane perpendicular to gap_normal
        rod_dir = jnp.sin(theta) * lateral_axis - jnp.cos(theta) * vertical_axis
        rod_dir = rod_dir / (jnp.linalg.norm(rod_dir) + 1e-9)

        gap_center = state.pivot + offset_len * rod_dir

        v_axis = -rod_dir
        # For right-handed frame: gap_normal × u_axis = v_axis
        # So: u_axis = v_axis × gap_normal
        u_axis = jnp.cross(v_axis, gap_normal)
        u_axis = u_axis / (jnp.linalg.norm(u_axis) + 1e-9)

        # gap_R columns: drone X→gap_normal, Y→u_axis, Z→v_axis
        gap_R = jnp.stack([gap_normal, u_axis, v_axis], axis=1)

        last_gap_centers = jnp.roll(state.last_gap_centers, shift=-1, axis=0)
        last_gap_centers = last_gap_centers.at[-1].set(gap_center)

        last_gap_Rs = jnp.roll(state.last_gap_Rs, shift=-1, axis=0)
        last_gap_Rs = last_gap_Rs.at[-1].set(gap_R)

        # First update state with new gap position/orientation
        next_state = state.replace(
            time=state.time + self.dt,
            step_idx=state.step_idx + 1,
            quadrotor_state=quadrotor_state,
            last_actions=last_actions,
            last_quadrotor_states=last_quadrotor_states,
            last_gap_centers=last_gap_centers,
            last_gap_Rs=last_gap_Rs,
            pend_theta=theta,
            pend_theta_dot=theta_dot,
            gap_center=gap_center,
            gap_center_prev=state.gap_center,  # Store previous gap center for velocity
            gap_u_axis=u_axis,
            gap_v_axis=v_axis,
            gap_R=gap_R,
            gap_normal=gap_normal,
            # delay remains unchanged during rollout (set at reset)
        )
        
        # NOW compute collision with UPDATED gap position
        collision_vector = self._get_collision_vector(
            p=quadrotor_state.p,
            R=quadrotor_state.R,
            gap_center=next_state.gap_center,
            gap_dims=self.gap_dims,
            gap_u_axis=next_state.gap_u_axis,
            gap_v_axis=next_state.gap_v_axis,
            margin_horizontal=self.margin_gap_collision_horizontal,
            margin_vertical=self.margin_gap_collision_vertical,
        )
        
        # Update collision vector in state
        next_state = next_state.replace(collision_vector=collision_vector)


        obs = self._get_obs(next_state)
        reward, reward_components = self._get_reward(state, next_state)
        terminated = self._is_colliding(next_state)

        truncated = jnp.greater_equal(next_state.step_idx, self.max_steps_in_episode)

        info = {"reward_components" : reward_components}

        return EnvTransition(next_state, obs, reward, terminated, truncated, info)

    