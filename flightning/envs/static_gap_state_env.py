import functools
from typing import Optional
import os
import yaml

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
from flightning import REPO_PATH


@jdc.pytree_dataclass
class EnvState(env_base.EnvState):
    time: float
    step_idx: int
    quadrotor_state: QuadrotorState
    last_actions: jax.Array
    last_quadrotor_states: QuadrotorState
    delay: float  # Randomized delay for each episode
    gap_center: jax.Array
    gap_normal: jax.Array 
    gap_dims: jax.Array # [width, height]
    gap_u_axis: jax.Array
    gap_v_axis: jax.Array
    gap_R : jax.Array
    goal: jax.Array
    current_epoch: int  # Track current training epoch for curriculum learning
    collision_vector: jax.Array  # Collision vector for collision checking
    last_gap_centers: jax.Array  # History of gap centers for vision env
    last_gap_Rs: jax.Array  # History of gap rotations for vision env
    projected_corners: jax.Array      # shape (K, 4, 2)
    projected_valid_flags: jax.Array  # shape (K, 4)


class StaticGapStateEnv(env_base.Env[EnvState]):
    """Environment where a drone must fly through an oriented rectangular gap and reach a goal."""

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
        start_jitter: float = 1.0, # -> the spawn area is a cube of side 2*start_jitter centered in the start point (clipped to world bounds)
        post_gap_distance: float = 2.0, # To set the goal after the gap #! TODO 2 funziona bene -> a 1 funziona da dio anche
        gap_dims: jax.Array = jnp.array([0.80, 0.25]), #! TODO set this also in other envs (before was [0.4, 0.15])
        # drone_radius: float = 0.06,
        # drone_height_radius: float = 0.06 / 2, 
        drone_radius: float = 0.38 / 2 , #! TODO set this also in other envs (before was 0.18 / 2)
        drone_height_radius: float = 0.10 / 2, #! TODO set this also in other envs (before was 0.08 / 2)
        world_box_min: Optional[jax.Array] = jnp.array([-10.0, -10.0, 0.0]),
        world_box_max: Optional[jax.Array] = jnp.array([10.0, 10.0, 10.0]),
        training: bool = True,
        num_epochs: int = 1000,  # Total training epochs for curriculum learning
        max_rot_theta_degrees: float = 45.0,  # Max rotation angle for gap orientation
        velocity_std: float = 0.1,
        omega_std: float = 0.1,
        initial_epoch: int = 0,  # Starting epoch for curriculum learning (used when resuming training)
        algo: str = None,  # Algorithm type: "bptt" or "ppo" (for loading algo-specific reward weights)
    ):
        
        self._algo = algo
        self.training = training
        self.num_epochs = num_epochs
        self.max_rot_theta_radians = jnp.deg2rad(max_rot_theta_degrees)
        self.initial_epoch = initial_epoch  # Store initial epoch for curriculum learning
        # self.goal = jnp.array([0.0, 0.0, 1.0]) # placeholder
        self.world_box = WorldBox(
            world_box_min,
            world_box_max,
        )

        self.max_steps_in_episode = max_steps_in_episode
        self.dt = np.array(dt)

        self.num_last_quad_states = num_last_quad_states
        self.margin = margin

        self.start_distance = start_distance 
        self.start_jitter = start_jitter 
        self.post_gap_distance = post_gap_distance

        self.gap_dims = gap_dims
        self.margin_gap_collision_vertical = gap_dims[1] / 2 - drone_height_radius
        self.margin_gap_collision_horizontal = gap_dims[0] / 2 - drone_radius
        self.collision_space_high = 0.2 # max penetration for collision vector

        # Drone Ellipsoid Matrix Q for collision checking
        self.drone_Q = jnp.diag(jnp.array([drone_radius, drone_radius, drone_height_radius])) 
        self.drone_radius = drone_radius
        self.drone_height_radius = drone_height_radius

        self.res_model_params = res_model_params



        # quadrotor
        if drone_path is not None:
            self.quadrotor = Quadrotor.from_yaml(drone_path, dyn_config=dyn_config)
        else:
            self.quadrotor = Quadrotor.default_quadrotor()

        self.omega_min = self.quadrotor._omega_max * -1
        self.omega_max = self.quadrotor._omega_max
        self.thrust_min = self.quadrotor._thrust_min
        self.thrust_max = self.quadrotor._thrust_max
        self.v_min = jnp.array([-10.0, -10.0, -10.0])
        self.v_max = jnp.array([10.0, 10.0, 10.0])


        assert delay >= 0.0, "Delay must be non-negative"
        self.delay_nominal = np.array(delay)  # Nominal delay value
        self.delay_min = delay - 0.005  # Minimum delay for randomization
        self.delay_max = delay + 0.005   # Maximum delay for randomization
        # Size queue to handle maximum possible delay
        self.num_last_actions = int(np.floor(self.delay_max / dt)) + 2

        thrust_hover = 9.81 * self.quadrotor._mass
        self.hovering_action = jnp.array([thrust_hover, 0.0, 0.0, 0.0])

        self.velocity_std = velocity_std
        self.omega_std = omega_std
        # Set reward cost parameters as instance attributes
        self.reward_sharpness_progress = 5.0
        self.reward_sharpness_centering = 20.0


        self.smoothness_weight = 0.0
        self.action_penalty_weight = 0.5
        self.vel_norm_penalty_weight = 0.0
        self.vel_penalty_weight = 0.1
        self.omega_penalty_weight = 0.1 
        self.acc_penalty_weight = 0.1
        self.progress_weight = 0.5
        self.centering_weight = 3.0
        self.near_gap_centering_weight = 5.0 #5.0
        self.alignment_weight = 5.0
        self.collision_weight = 24.0
        self.camera_alignment_weight = 0.0 #! NOTE not used for now
        
        # Override with config file values if available
        self._load_reward_weights("static_state")

    def _load_reward_weights(self, config_key: str):
        """Load reward weights from training_configs.yaml.
        
        Looks up reward_weights/{algo}/{config_key} if self._algo is set,
        otherwise skips.
        """
        algo = getattr(self, '_algo', None)
        if algo is None:
            return
        config_path = os.path.join(REPO_PATH, "configs/training_configs.yaml")
        if not os.path.exists(config_path):
            return
        try:
            with open(config_path, 'r') as f:
                data = yaml.safe_load(f)
            weights = data.get("reward_weights", {}).get(algo, {}).get(config_key)
            if weights is None:
                return
            for key, value in weights.items():
                if hasattr(self, key):
                    setattr(self, key, value)
            print(f"[{self.__class__.__name__}] Loaded reward weights: {algo}/{config_key}")
        except Exception as e:
            print(f"[{self.__class__.__name__}] Warning: could not load reward weights: {e}")

    # ------------------------------------------------------------
    # RESET 
    # ------------------------------------------------------------
    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(self, key, state: Optional[EnvState] = None) -> tuple[EnvState, jax.Array]:
        key_gapc, key_gn, key_start, key_dr, key_rot, key_vel, key_omega, key_delay = jax.random.split(key, 8)

        # ------------------------------------------------------------
        # GAP randomization
        # ------------------------------------------------------------
        # randomize gap position
        min_c = self.world_box.min + self.margin
        max_c = self.world_box.max - self.margin
        gap_center = jax.random.uniform(key_gapc, shape=(3,), minval=min_c, maxval=max_c)

        # randomize gap normal
        # angle = jax.random.uniform(key_gn, minval=-jnp.pi, maxval=jnp.pi)
        angle = jnp.pi / 2 #! NOTE fixed 
        gap_center_start = jnp.array([0.0, 0.0, 5.0]) 
        # gap_center_offset = jax.random.uniform(key_gapc, shape=(3,), minval=-0.5, maxval=0.5)
        
        gap_center = gap_center_start #+ gap_center_offset #! NOTE this is equal to randomize the drone starting position
        
        cos_a, sin_a = jnp.cos(angle), jnp.sin(angle)
        
        # The normal is [cos(a), sin(a), 0.0], so z component is always 0.
        gap_normal = jnp.array([cos_a, sin_a, 0.0]) # -> y axis 
        gap_normal = gap_normal / (jnp.linalg.norm(gap_normal) + 1e-9)

        # Curriculum learning for rot_theta based on current epoch - use initial_epoch if this is the first reset
        current_epoch = self.initial_epoch if state is None else state.current_epoch
        
        # Initialize gap frame axes
        v_axis_init = jnp.array([0.0, 0.0, 1.0])
        u_axis_init = jnp.cross(v_axis_init, gap_normal)
        u_axis_init = u_axis_init / (jnp.linalg.norm(u_axis_init) + 1e-9)
        
        # Apply rotation around gap_normal by rot_theta (Rodrigues' formula)
        rot_theta = jax.random.uniform(key_rot, minval=-self.max_rot_theta_radians, maxval=self.max_rot_theta_radians)
        
        u_axis = u_axis_init * jnp.cos(rot_theta) + jnp.cross(gap_normal, u_axis_init) * jnp.sin(rot_theta)
        u_axis = u_axis / (jnp.linalg.norm(u_axis) + 1e-9)
        
        v_axis = v_axis_init * jnp.cos(rot_theta) + jnp.cross(gap_normal, v_axis_init) * jnp.sin(rot_theta)
        v_axis = v_axis / (jnp.linalg.norm(v_axis) + 1e-9)
        

        # ------------------------------------------------------------
        # GOAL position (after the gap along the normal)
        # ------------------------------------------------------------
        goal = gap_center + self.post_gap_distance * gap_normal
        goal = jnp.clip(goal, self.world_box.min + 0.1, self.world_box.max - 0.1)

        # ------------------------------------------------------------
        # Randomize START position (before the gap along the normal)
        # ------------------------------------------------------------
        # ranomized distance before the gap along the normal
        key_start1, key_start2, key_start3 = jax.random.split(key_start, 3)
        dist_behind_plane = self.start_distance + self.start_jitter * jax.random.uniform(key_start1, minval=-1.0, maxval=1.0) # start_distance +/- start_jitter
        jitter_vertical = self.start_jitter * jax.random.uniform(key_start2, minval=-1.0, maxval=1.0) # +/- start_jitter
        jitter_lateral = self.start_jitter * jax.random.uniform(key_start3, minval=-1.0, maxval=1.0) # +/- start_jitter

        vertical_axis = jnp.array([0.0, 0.0, 1.0])
        lateral_axis = jnp.cross(gap_normal, vertical_axis)
        lateral_axis = lateral_axis / (jnp.linalg.norm(lateral_axis) + 1e-9)
        start = (
            gap_center
            - dist_behind_plane * gap_normal # set the start before the gap 
            + jitter_vertical * vertical_axis
            + jitter_lateral * lateral_axis
        )

        # start = gap_center - self.start_distance * gap_normal 


        start = jnp.clip(start, self.world_box.min + 0.1, self.world_box.max - 0.1)

        # ------------------------------------------------------------
        # DRONE ORIENTATION - roll=0, pitch=0, yaw towards gap center + randomization
        # ------------------------------------------------------------
        p = start
        
        # Compute yaw angle to point towards gap center (in XY plane)
        vec_to_gap = gap_center - p
        yaw_to_gap = jnp.arctan2(vec_to_gap[1], vec_to_gap[0])
        
        # Add small randomization to yaw (±20 degrees by default)
        yaw_randomization_max = jnp.deg2rad(20.0)
        key_yaw = jax.random.split(key_dr)[0]
        yaw_offset = jax.random.uniform(key_yaw, minval=-yaw_randomization_max, maxval=yaw_randomization_max)
        yaw = yaw_to_gap + yaw_offset
        
        # Build rotation matrix with roll=0, pitch=0, yaw=yaw
        # R = Rz(yaw) where Rz rotates around Z-axis
        cos_yaw = jnp.cos(yaw)
        sin_yaw = jnp.sin(yaw)
        
        # X_B: forward direction (rotated by yaw in XY plane)
        forward_dir = jnp.array([cos_yaw, sin_yaw, 0.0])
        
        # Y_B: right direction (perpendicular to forward in XY plane)
        right_dir = jnp.array([-sin_yaw, cos_yaw, 0.0])
        
        # Z_B: up direction (world up, unchanged)
        up_dir = jnp.array([0.0, 0.0, 1.0])
        
        # Build rotation matrix with columns [X_B, Y_B, Z_B]
        R = jnp.stack([forward_dir, right_dir, up_dir], axis=1)


        # ------------------------------------------------------------
        # Drone initial velocity and angular rates
        # ------------------------------------------------------------
        v_noise = jax.random.normal(key_vel, shape=(3,)) * self.velocity_std
        v = v_noise
        omega_noise = jax.random.normal(key_omega, shape=(3,)) * self.omega_std
        omega = omega_noise

        quadrotor_state = self.quadrotor.create_state(p=p, R=R, v=v, omega=omega, dr_key=key_dr)

        last_actions = jnp.tile(self.hovering_action, (self.num_last_actions, 1))
        last_quadrotor_states = [quadrotor_state] * self.num_last_quad_states
        last_quadrotor_states = stack_pytrees(last_quadrotor_states)

        gap_R = jnp.stack([gap_normal, u_axis, v_axis], axis=1)  # shape (3, 3)

        last_gap_centers = jnp.tile(gap_center, (self.num_last_quad_states, 1))
        last_gap_Rs = jnp.tile(gap_R, (self.num_last_quad_states, 1, 1))

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
            projected_corners=jnp.zeros((4, 4, 2)),  # Placeholder, not used in state-based env
            projected_valid_flags=jnp.zeros((4, 4,)),  # Placeholder
            last_gap_centers=last_gap_centers,
            last_gap_Rs=last_gap_Rs,
        )

        obs = self._get_obs(state)
        return state, obs
    
    def _get_collision_vector(
        self,
        p: jax.Array,
        R: jax.Array,
        gap_center: jax.Array,
        gap_dims: jax.Array,
        gap_u_axis: jax.Array,
        gap_v_axis: jax.Array,
        margin_vertical: float = 0.0,
        margin_horizontal: float = 0.0,

    ) -> jax.Array:
        """
        Computes the collision constraint vector C(t).
        A positive value means penetration (collision).
        """
        # define the 4 planes (A, b) of the gap in world frame
        w_half = (gap_dims[0] - 2 * margin_horizontal) / 2.0
        h_half = (gap_dims[1] - 2 * margin_vertical) / 2.0

        # normals of the 4 planes
        A = jnp.stack([gap_u_axis, -gap_u_axis, gap_v_axis, -gap_v_axis])

        # distances of planes from origin
        b = jnp.array(
            [
                w_half + jnp.dot(gap_u_axis, gap_center),
                w_half - jnp.dot(gap_u_axis, gap_center),
                h_half + jnp.dot(gap_v_axis, gap_center),
                h_half - jnp.dot(gap_v_axis, gap_center),
            ]
        )

        # implement formula 98: C(t) = sqrt(sum((A @ R @ Q)^2)) + A @ p - b
        # note: The paper's formula is for constraint <= 0.
        # we compute -C(t) from the paper to get a "penetration" value.
        # C_paper = [[A R Q]^2 1]^{1/2} + A p - b <= 0
        # we compute: penetration = [[A R Q]^2 1]^{1/2} + A p - b
        AR = A @ R
        ARQ = AR @ self.drone_Q
        
        # term1 = jnp.sqrt(jnp.sum(jnp.square(ARQ), axis=1))
        # more stable computation for sqrt(sum(diag(M M^T)))
        term1 = jnp.linalg.norm(ARQ, axis=1)

        term2 = A @ p
        
        # C_paper = term1 + term2 - b
        # if C_paper > 0, we are colliding.
        collision_vector = term1 + term2 - b

        collision_vector = jnp.maximum(collision_vector, 0.0) # only keep positive values (penetration)
        return collision_vector
    

    def _get_obs(self, state: EnvState) -> jax.Array:

    

        obs = jnp.concatenate(
            [
                state.quadrotor_state.p,
                math_utils.vec(state.quadrotor_state.R),
                state.quadrotor_state.v,  
                state.quadrotor_state.omega,
                state.goal,
                state.gap_center,
                state.gap_R.flatten(),
                state.last_actions.flatten(),  # last actions (for smoothness penalty)  
            ]
        )
        return obs
        


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

        # TODO remove last quadrotor states 
        # remove the oldest quadrotor state and add the current one
        last_quad_states_rolled = jax.tree_map(lambda arr: jnp.roll(arr, shift=-1, axis=0), state.last_quadrotor_states)
        last_quadrotor_states = jax.tree_map(
            lambda hist_arr, curr_val: hist_arr.at[self.num_last_quad_states - 1].set(curr_val),
            last_quad_states_rolled,
            quadrotor_state
        )
        
        # NOTE -> with moving gap the gap_R has to be changed over time
        collision_vector = self._get_collision_vector(
            p=quadrotor_state.p,
            R=quadrotor_state.R,
            gap_center=state.gap_center,
            gap_dims=self.gap_dims,
            gap_u_axis=state.gap_R[:, 0],
            gap_v_axis=state.gap_R[:, 1],
            margin_horizontal=self.margin_gap_collision_horizontal,
            margin_vertical=self.margin_gap_collision_vertical,
        )
        next_state = state.replace(
            time=state.time + self.dt,
            step_idx=state.step_idx + 1,
            quadrotor_state=quadrotor_state,
            last_actions=last_actions,
            last_quadrotor_states=last_quadrotor_states, #TODO remove
            collision_vector=collision_vector,
            # delay remains unchanged during rollout (set at reset)
            # res_model_params remains unchanged during rollout
            # current_epoch remains unchanged during rollout (updated by bptt.py)
        )


        obs = self._get_obs(next_state)
        reward, reward_components = self._get_reward(state, next_state)
        terminated = self._is_colliding(next_state)

        truncated = jnp.greater_equal(next_state.step_idx, self.max_steps_in_episode)

        info = {"reward_components" : reward_components}

        return EnvTransition(next_state, obs, reward, terminated, truncated, info)


    # ============================================================================
    # INDIVIDUAL COST COMPONENT METHODS (used by both _get_reward and debugging)
    # ============================================================================
    
    def compute_progress_cost(self, state: EnvState) -> jax.Array:
        """Progress cost: distance from drone to goal along gap normal."""
        p = state.quadrotor_state.p
        g = state.goal
        d_p_goal = jnp.linalg.norm(g - p)  # straight-line distance to goal
    
        progress_cost = self.progress_weight * smooth_l1(self.reward_sharpness_progress * d_p_goal) / self.reward_sharpness_progress
        
        return progress_cost
    
    def compute_centering_cost(self, state: EnvState) -> jax.Array:
        """Centering cost: distance from gap center in u and v directions."""
        p = state.quadrotor_state.p
        c = state.gap_center
        u_axis = state.gap_u_axis
        v_axis = state.gap_v_axis
        n = state.gap_normal
        
        # Distance from gap plane
        d_curr = jnp.dot(p - c, n)
        
        sigmoid_weight = self.centering_weight * jax.nn.sigmoid(-3.0 * (d_curr - 0.5))
                
        # Horizontal (u) centering: moderate strength
        weight_u = sigmoid_weight 
        
        # Vertical (v) centering: MUCH stronger - critical for narrow height tolerance
        weight_v = 3.0 * sigmoid_weight #! NOTE this was 2.667
        
        error_u = jnp.dot(p - c, u_axis)
        error_v = jnp.dot(p - c, v_axis)
        
        # Use smooth_l1 for soft penalty that doesn't explode
        centering_cost_u = weight_u * smooth_l1(self.reward_sharpness_centering * error_u) / self.reward_sharpness_centering
        centering_cost_v = weight_v * smooth_l1(self.reward_sharpness_centering * error_v) / self.reward_sharpness_centering
        
        return centering_cost_u + centering_cost_v
    
    def compute_near_gap_centering_cost(self, state: EnvState) -> jax.Array:
        """Centering cost: distance from gap center in u and v directions."""
        p = state.quadrotor_state.p
        c = state.gap_center
        u_axis = state.gap_u_axis
        v_axis = state.gap_v_axis
        n = state.gap_normal
        
        # Distance from gap plane
        d_curr = jnp.dot(p - c, n)
        gaussian_weight = self.near_gap_centering_weight * jnp.exp(-5.0 * jnp.square(d_curr))        
        # Horizontal (u) centering: moderate strength
        weight_u = gaussian_weight
        
        # Vertical (v) centering: MUCH stronger - critical for narrow height tolerance
        weight_v = 3.0 * gaussian_weight
        
        error_u = jnp.dot(p - c, u_axis)
        error_v = jnp.dot(p - c, v_axis)
        
        # Use smooth_l1 for soft penalty that doesn't explode
        centering_cost_u = weight_u * smooth_l1(self.reward_sharpness_centering * error_u) / self.reward_sharpness_centering
        centering_cost_v = weight_v * smooth_l1(self.reward_sharpness_centering * error_v) / self.reward_sharpness_centering
        
        return centering_cost_u + centering_cost_v
    

    def compute_alignment_cost(self, state: EnvState) -> jax.Array:
        """Alignment cost: penalizes roll, pitch, yaw misalignment with gap plane when crossing."""
        p = state.quadrotor_state.p
        R_drone = state.quadrotor_state.R
        c = state.gap_center
        R_gap = state.gap_R  # Gap rotation matrix [gap_normal, gap_u_axis, gap_v_axis]
        n = state.gap_normal
        
        # Distance from gap plane along normal
        d_curr = jnp.dot(p - c, n)
        
        # Strong weight near the gap plane (peaks at d=0, decays rapidly)
        distance_weight = self.alignment_weight * jnp.exp(-4.0 * jnp.square(d_curr))
        
        # Compute relative rotation matrix: R_rel = R_gap^T @ R_drone
        R_rel = R_gap.T @ R_drone
        
        # Extract Euler angles (ZYX convention) from relative rotation
        # Roll (rotation around x-axis)
        roll_error = jnp.arctan2(R_rel[2, 1], R_rel[2, 2])
        
        # Pitch (rotation around y-axis)
        pitch_error = jnp.arctan2(-R_rel[2, 0], jnp.sqrt(R_rel[2, 1]**2 + R_rel[2, 2]**2))
        
        # Yaw (rotation around z-axis)
        yaw_error = jnp.arctan2(R_rel[1, 0], R_rel[0, 0])
        
        # Normalize errors to [0, 1] based on maximum acceptable angles
        roll_max = self.max_rot_theta_radians   # 45° maximum roll error
        pitch_max = self.max_rot_theta_radians  # 45° maximum pitch error
        yaw_max = self.max_rot_theta_radians    # 45° maximum yaw error
        
        # Compute normalized costs (squared for sharper gradients near zero)
        roll_cost = jnp.square(jnp.clip(jnp.abs(roll_error) / roll_max, 0.0, 1.0))
        pitch_cost = jnp.square(jnp.clip(jnp.abs(pitch_error) / pitch_max, 0.0, 1.0))
        yaw_cost = jnp.square(jnp.clip(jnp.abs(yaw_error) / yaw_max, 0.0, 1.0))
        
        # Weighted combination: yaw is most important, then pitch, then roll
        combined_cost = yaw_cost + 2.0 * pitch_cost + 2.0 * roll_cost
        # combined_cost = 2.0 * pitch_cost + roll_cost
        
        return distance_weight * combined_cost
    
    
    def compute_gap_collision_cost(self, state: EnvState) -> jax.Array:
        """Gap collision cost: penetration into gap boundaries."""
        p = state.quadrotor_state.p
        R = state.quadrotor_state.R
        c = state.gap_center
        u_axis = state.gap_u_axis
        v_axis = state.gap_v_axis
        n = state.gap_normal
        
        d_curr = jnp.dot(p - c, n)
        distance_weight_coll = self.collision_weight * jnp.exp(-6.0 * jnp.square(d_curr))
        
        collision_vector = self._get_collision_vector(
            p, R, c, state.gap_dims, u_axis, v_axis,
            margin_horizontal=self.margin_gap_collision_horizontal,
            margin_vertical=self.margin_gap_collision_vertical
        )
        
        return distance_weight_coll * jnp.sum(collision_vector)
    
    # def compute_stabilization_cost(self, state: EnvState, last_state: EnvState) -> jax.Array:
    #     """Stabilization cost: velocity, angular velocity, acceleration, and action penalties."""
    #     p = state.quadrotor_state.p
    #     v = state.quadrotor_state.v
    #     omega = state.quadrotor_state.omega
    #     acc = state.quadrotor_state.acc
    #     action = state.last_actions[-1]
    #     action_last = last_state.last_actions[-1]
        
    #     c = state.gap_center
    #     n = state.gap_normal
    #     d_curr = jnp.dot(p - c, n)
        
        
        
    #     # sum of sigmoids to reduce stabilization cost when close to the gap  
    #     min_val = 0.2
    #     reduction_factor1 = 1.0
    #     reduction_factor2 = 1.0
    #     coeff1 = reduction_factor1 - min_val
    #     coeff2 = reduction_factor2 - min_val
    #     sigmoid1 = coeff1 * jax.nn.sigmoid(20.0 * (d_curr + 0.7))
    #     sigmoid2 = coeff2 * jax.nn.sigmoid(20.0 * (d_curr - 0.7))
    #     sum_of_sigmoids =  sigmoid1 - sigmoid2

    #     weight = 1.0 - sum_of_sigmoids
    #     # inv_sigmoid_vel = jax.nn.sigmoid(-20.0 * (d_curr))
    #     inv_sigmoid_vel = min_val + jax.nn.sigmoid(-20.0 * (d_curr + 0.5))  # Numerically stable version
    #     gaussian_vel = jnp.exp(-1.0 * jnp.square(d_curr + 0.6))
    #     vel_weight = inv_sigmoid_vel

    #     vel_cost = vel_weight * self.vel_penalty_weight * smooth_l1(v)
    #     omega_cost = weight * self.omega_penalty_weight * smooth_l1(omega)
    #     acc_cost = weight * self.acc_penalty_weight * smooth_l1(acc)
    #     action_cost = weight * self.action_penalty_weight * smooth_l1(action - self.hovering_action)
    #     smoothness_cost = weight * self.smoothness_weight * jnp.inner(action, action_last)
        
    #     stabilization_cost = vel_cost + omega_cost + acc_cost + action_cost + smoothness_cost
    #     return stabilization_cost
    def compute_stabilization_cost(self, state: EnvState, last_state: EnvState) -> jax.Array:
        """Stabilization cost: velocity, angular velocity, acceleration, and action penalties."""
        p = state.quadrotor_state.p
        v = state.quadrotor_state.v
        omega = state.quadrotor_state.omega
        acc = state.quadrotor_state.acc
        action = state.last_actions[-1]
        action_last = last_state.last_actions[-1]
        
        c = state.gap_center
        n = state.gap_normal
        d_curr = jnp.dot(p - c, n)
        
        
        
        # sum of sigmoids to reduce stabilization cost when close to the gap  
        min_val = 0.2
        reduction_factor1 = 1.0
        reduction_factor2 = 1.0
        coeff1 = reduction_factor1 - min_val
        coeff2 = reduction_factor2 - min_val
        sigmoid1 = coeff1 * jax.nn.sigmoid(20.0 * (d_curr + 0.7))
        sigmoid2 = coeff2 * jax.nn.sigmoid(20.0 * (d_curr - 0.7))
        sum_of_sigmoids =  sigmoid1 - sigmoid2

        weight = 1.0 - sum_of_sigmoids
        # inv_sigmoid_vel = jax.nn.sigmoid(-20.0 * (d_curr))
        # inv_sigmoid_vel = min_val + jax.nn.sigmoid(-20.0 * (d_curr + 0.5))  # Numerically stable version
        # vel_weight = inv_sigmoid_vel
        # vel_weight = 0.1 + jnp.exp(-0.5 * d_curr - 1.0)  

        # vel_cost = vel_weight * self.vel_penalty_weight * smooth_l1(v)
        vel_weight = jnp.maximum(0.0, -d_curr) * jax.nn.sigmoid(-8.0 * d_curr)

        # Penalize only velocity component parallel to gap normal
        v_parallel = jnp.dot(v, n)  # Scalar: velocity component along gap normal
        vel_norm_cost = vel_weight * self.vel_norm_penalty_weight * smooth_l1(jnp.array([v_parallel]))
        vel_cost = weight * self.vel_penalty_weight * smooth_l1(v)
        omega_cost = weight * self.omega_penalty_weight * smooth_l1(omega)
        acc_cost = weight * self.acc_penalty_weight * smooth_l1(acc)
        action_cost = weight * self.action_penalty_weight * smooth_l1(action - self.hovering_action)
        smoothness_cost = weight * self.smoothness_weight * jnp.sum((action - action_last) ** 2)
        stabilization_cost = vel_cost + vel_norm_cost + omega_cost + acc_cost + action_cost + smoothness_cost
        return stabilization_cost
    

    def compute_camera_alignment_cost(self, state: EnvState) -> jax.Array:
        """Camera alignment cost: angle between camera optical axis (x-axis of drone) and vector from drone to gap center."""
        p = state.quadrotor_state.p
        R_drone = state.quadrotor_state.R
        c = state.gap_center
        n = state.gap_normal
        
        # Vector from drone to gap center
        vec_to_gap = c - p
        distance_to_gap = jnp.linalg.norm(vec_to_gap)
        vec_to_gap_normalized = vec_to_gap / (distance_to_gap + 1e-8)
        
        # Camera optical axis (x-axis of drone body frame)
        camera_axis = R_drone[:, 0]
        
        # Cosine of angle between camera axis and vector to gap
        cos_angle = jnp.dot(camera_axis, vec_to_gap_normalized)
        
        # Convert to angle error: angle = arccos(cos_angle)
        # Cost increases as alignment worsens (angle → 90°)
        # At perfect alignment (angle=0°): cos=1, cost≈0
        # At 90° misalignment: cos=0, cost≈1
        angle_error = jnp.arccos(jnp.clip(cos_angle, -0.999, 0.999))
        
        # Normalize angle error to [0, 1] and square for sharper gradients
        normalized_angle_error = angle_error / jnp.pi  # angle in [0, π] → [0, 1]

        return self.camera_alignment_weight * jnp.square(normalized_angle_error)
    

    # ============================================================================

    def _get_reward(
        self, last_state: EnvState, next_state: EnvState
    ) -> jax.Array:
        
        # Compute individual cost components using the dedicated methods
        progress_cost = self.compute_progress_cost(next_state)
        centering_cost = self.compute_centering_cost(next_state)
        centering_near_gap_cost = self.compute_near_gap_centering_cost(next_state)
        alignment_cost = self.compute_alignment_cost(next_state)
        gap_collision_cost = self.compute_gap_collision_cost(next_state)
        stabilization_cost = self.compute_stabilization_cost(next_state, last_state)
        camera_alignment_cost = self.compute_camera_alignment_cost(next_state)
        
        # TOTAL COST
        cost = (progress_cost 
            + centering_cost
            + centering_near_gap_cost
            + alignment_cost
            + gap_collision_cost
            + stabilization_cost
            + camera_alignment_cost
            )

    
    
        # scale by time
        reward = -self.dt * cost
        return reward, {
            "progress_cost": -self.dt * progress_cost,
            "centering_cost": -self.dt * centering_cost,
            "centering_near_gap_cost": -self.dt * centering_near_gap_cost,
            "alignment_cost": -self.dt * alignment_cost,
            "gap_collision_cost": -self.dt * gap_collision_cost,
            "stabilization_cost": -self.dt * stabilization_cost,
            "camera_alignment_cost": -self.dt * camera_alignment_cost,
        }

    def get_reward_component_keys(self) -> list[str]:
        """
        Returns the list of reward component keys used in this environment.
        This is extracted from the reward_components dictionary in _get_reward().
        """
        reward_components_keys = [
            "progress_cost",
            "centering_cost",
            "centering_near_gap_cost",
            "alignment_cost",
            "gap_collision_cost",
            "stabilization_cost",
            "camera_alignment_cost",
        ]
        return reward_components_keys




    # ------------------------------------------------------------
    # COLLISION CHECK
    # ------------------------------------------------------------
    def _is_colliding(self, state: EnvState) -> jax.Array:
        p = state.quadrotor_state.p
        world_collision = jnp.logical_not(self.world_box.contains(p))
        is_colliding = world_collision
        return is_colliding
    
    
    # ------------------------------------------------------------
    # SPACES
    # ------------------------------------------------------------
    @property
    def action_space(self) -> spaces.Box:
        low = jnp.concatenate([jnp.array([self.thrust_min * 4]), self.omega_min])
        high = jnp.concatenate([jnp.array([self.thrust_max * 4]), self.omega_max])
        return spaces.Box(low, high, shape=(4,))

    
    @property
    def observation_space(self) -> spaces.Box:
        n = self.num_last_actions
        action_high_repeated = jnp.concatenate([self.action_space.high] * n)
        action_low_repeated = jnp.concatenate([self.action_space.low] * n)

        
        low = jnp.concatenate(
            
            [
                self.world_box.min,  # p (3)
                -jnp.ones(9),  # R_vec (9)
                self.v_min,  # v (3)
                self.omega_min,  # omega (3)
                self.world_box.min,  # goal (3)
                self.world_box.min,  # gap_center (3)
                -jnp.ones(9),  # gap_R (9)
                action_low_repeated,  # last actions (for smoothness penalty)
            ]
        )
        high = jnp.concatenate(
            [
                self.world_box.max,  # p (3)
                jnp.ones(9),  # R_vec (9)
                self.v_max,  # v (3) 
                self.omega_max,  # omega (3)
                self.world_box.max,  # goal (3)  
                self.world_box.max,  # gap_center (3)
                jnp.ones(9),  # gap_R (9)
                action_high_repeated,  # last actions (for smoothness penalty)
            ]
        )
        
        shape = (low.shape[0],)
        return spaces.Box(low=low, high=high, shape=shape)
    
    
    # ------------------------------------------------------------
    # VISUALIZATION: Visualization methods have been moved to flightning.envs.visualization.visualize_static_narrow_gap
    # ------------------------------------------------------------

