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
from flightning.sensors import CameraNames, DoubleSphereCamera
from flightning.envs.static_gap_state_env import StaticGapStateEnv
from flightning.envs.static_gap_state_env import EnvState


class StaticGapVisionEnv(StaticGapStateEnv):
    """Environment where a drone must fly through an oriented rectangular gap and reach a goal."""

    def __init__(
        self,
        *,
        max_steps_in_episode=300,
        dt=0.02,
        delay=0.02,
        drone_path=None,
        dyn_config=None,  # Dynamics configuration for residual model
        res_model_params=None,  # Residual model parameters (loaded from file)
        num_last_quad_states=10,
        margin=jnp.array([2.0, 2.0, 1.0]),
        start_distance: float = 2.0, 
        start_jitter: float = 1.0, # -> the spawn area is a cube of side 2*start_jitter centered in the start point (clipped to world bounds)
        post_gap_distance: float = 2.0, # To set the goal after the gap
        gap_dims: jax.Array = jnp.array([0.80, 0.25]),
        training: bool = True,
        num_epochs: int = 1000,  # Total training epochs for curriculum learning
        skip_frames: int = 1,
        deployment: bool = False,
        max_rot_theta_degrees: float = 45.0,  # Max rotation angle for gap orientation
        velocity_std: float = 0.1,
        omega_std: float = 0.1,
        initial_epoch: int = 0,  # Starting epoch for curriculum learning (used when resuming training)
        algo: str = None,  # Algorithm type: "bptt" or "ppo"
    ):
        
        super().__init__(
            max_steps_in_episode=max_steps_in_episode,
            dt=dt,
            delay=delay,
            drone_path=drone_path,
            dyn_config=dyn_config,  # Dynamics configuration for residual model
            res_model_params=res_model_params,  # Residual model parameters (loaded from file)
            num_last_quad_states=num_last_quad_states,
            margin=margin,
            start_distance=start_distance,
            start_jitter=start_jitter,
            post_gap_distance=post_gap_distance,
            gap_dims=gap_dims,
            training=training,
            num_epochs=num_epochs,
            max_rot_theta_degrees=max_rot_theta_degrees,
            velocity_std=velocity_std,
            omega_std=omega_std,
            initial_epoch=initial_epoch,
            algo=algo,
        )

        self.cam = DoubleSphereCamera.from_name(CameraNames.EXAMPLE_QUAD) #! CAM change
        self.skip_frames = skip_frames

        self.num_vis_frames = 4  # will be set in _get_obs
        self.deployment = deployment

        # Override reward weights from config
        self._load_reward_weights("static_vision")

    @functools.partial(jax.jit, static_argnums=(0,))
    def _get_obs(self, state: EnvState) -> jax.Array:
        K = self.num_vis_frames


        # project the 4 gap corners into the image plane (with normalized coords) for each frame -> (u, v) coordinates 

        # concatenate observations + last actions (if needed)
        def project_fn(p, R):
            feature_pos = jnp.array([
                state.gap_center + 0.5 * self.gap_dims[0] * state.gap_u_axis + 0.5 * self.gap_dims[1] * state.gap_v_axis,
                state.gap_center - 0.5 * self.gap_dims[0] * state.gap_u_axis + 0.5 * self.gap_dims[1] * state.gap_v_axis,
                state.gap_center - 0.5 * self.gap_dims[0] * state.gap_u_axis - 0.5 * self.gap_dims[1] * state.gap_v_axis,
                state.gap_center + 0.5 * self.gap_dims[0] * state.gap_u_axis - 0.5 * self.gap_dims[1] * state.gap_v_axis,
            ])  # shape (4, 3)

            points_proj = self.cam.project_points_with_pose(feature_pos, p, R)  # shape (4, 3) -> (u, v, valid)
            
            # Extract coordinates and validity flags
            points_C = points_proj[:, :2]  # (u, v) coordinates (in pixels or normalized depending on camera)
            valid_flags = points_proj[:, 2]  # validity flags (1.0 if valid, 0.0 otherwise)
            
            def sort_corners_clockwise_start_tr(points, valid):
                center = jnp.mean(points, axis=0)
                angles = jnp.arctan2(points[:, 1] - center[1],
                                    points[:, 0] - center[0])
                idx = jnp.argsort(-angles)
                pts = points[idx]
                val = valid[idx]
                tr_idx = jnp.argmax(pts[:, 0] + pts[:, 1])
                pts = jnp.roll(pts, -tr_idx, axis=0)
                val = jnp.roll(val, -tr_idx, axis=0)
                return pts, val

            # sort points and valid flags in clockwise order starting from top-right
            points_C, valid_flags = sort_corners_clockwise_start_tr(points_C, valid_flags)

            if not self.deployment:
                # add random noise to the porjected points to simulate detection noise
                # Combine step_idx and current_epoch to create unique noise per step AND epoch
                noise_seed = state.step_idx * 100000 + state.current_epoch
                key_noise = jax.random.PRNGKey(noise_seed)
                noise = jax.random.uniform(key_noise, shape=points_C.shape, minval=-10.0, maxval=10.0)  # 5 pixel noise
                points_C = points_C + noise

            # Now compute normalized coordinates from the (possibly pixel) coordinates
            points_normalized = points_C / jnp.array([self.cam.width, self.cam.height], dtype=float) * jnp.array([2.0, 2.0]) - jnp.array([1.0, 1.0])

            # Set invalid points to a default value (e.g., -0.999)
            # This is done by broadcasting valid_flags to match shape (4, 1)
            valid_mask = valid_flags[:, None]  # shape (4, 1)
            points_normalized = jnp.where(valid_mask > 0.5, points_normalized, jnp.full_like(points_normalized, -0.999))

            # Compute area and aspect ratio only if all points are valid
            all_valid = jnp.all(valid_flags > 0.5)
            center_normalized = jnp.where(all_valid, jnp.mean(points_normalized, axis=0), -0.999)

            
            # Compute area using Shoelace formula (for arbitrary quadrilateral)
            x = points_normalized[:, 0]
            y = points_normalized[:, 1]

            area_norm = 0.5 * jnp.abs(
                x[0] * y[1] - x[1] * y[0] +
                x[1] * y[2] - x[2] * y[1] +
                x[2] * y[3] - x[3] * y[2] +
                x[3] * y[0] - x[0] * y[3]
            )

            normalized_area = area_norm / 4.0

            
            # Set area and aspect ratio to 0 if not all points are valid
            normalized_area = jnp.where(all_valid, normalized_area, 0.0)
            
            # visibility value -> between 0 and 1
            visibility = jnp.mean(valid_mask)
              # ---------- GAP DIRECTION VECTORS (computed from projected corners) ----------
            # Gap corners in normalized image coordinates:
            # Corner 0: +u/2, +v/2  (top-right)
            # Corner 1: -u/2, +v/2  (top-left)
            # Corner 2: -u/2, -v/2  (bottom-left)
            # Corner 3: +u/2, -v/2  (bottom-right)
            
            # Compute u-axis direction: average of right edge - left edge
            # Right edge: (corner[0] + corner[3]) / 2
            # Left edge: (corner[1] + corner[2]) / 2
            right_edge_center = (points_C[0] + points_C[3]) / 2.0
            left_edge_center = (points_C[1] + points_C[2]) / 2.0
            dir_u = right_edge_center - left_edge_center
            
            # Compute v-axis direction: average of top edge - bottom edge
            # Top edge: (corner[0] + corner[1]) / 2
            # Bottom edge: (corner[2] + corner[3]) / 2
            top_edge_center = (points_C[0] + points_C[1]) / 2.0
            bottom_edge_center = (points_C[2] + points_C[3]) / 2.0
            dir_v = top_edge_center - bottom_edge_center
            
            # Normalize direction vectors
            norm_u = jnp.linalg.norm(dir_u) + 1e-9
            norm_v = jnp.linalg.norm(dir_v) + 1e-9

            aspect_ratio = jnp.where(all_valid, norm_u / (norm_v + 1e-9), 0.0)

            dir_u_normalized = dir_u / norm_u
            dir_v_normalized = dir_v / norm_v
            
            # Set to zero if not all corners are valid
            dir_u_normalized = jnp.where(all_valid, dir_u_normalized, jnp.zeros_like(dir_u_normalized))
            dir_v_normalized = jnp.where(all_valid, dir_v_normalized, jnp.zeros_like(dir_v_normalized))
            
            return points_normalized, center_normalized, normalized_area, aspect_ratio, visibility, dir_u_normalized, dir_v_normalized, valid_mask  # shape (4, 2), scalar, scalar, scalar
        

        # -------------------------------------------------
        # Visual history
        # -------------------------------------------------
        idx = jnp.arange(K) * self.skip_frames
        p_hist = state.last_quadrotor_states.p[-1 - idx]
        R_hist = state.last_quadrotor_states.R[-1 - idx]

        points_normalized, center_normalized, normalized_area, aspect_ratio, visibility, dir_u, dir_v, valid_mask = jax.vmap(project_fn)(
            p_hist, R_hist
    )   
        alignment = jnp.dot(dir_u, jnp.array([1, 0]))

        # -------------------------------------------------
        # Temporal deltas (vision dynamics)
        # -------------------------------------------------

        def temporal_delta(x):
            """
            x: (K, D)
            returns: (D,) = x_t - x_{t-1} if K >= 2 else zeros
            """
            return jnp.where(
                K > 1,
                x[-1] - x[-2],
                jnp.zeros_like(x[-1])
            )

        delta_center = temporal_delta(center_normalized)        # shape (2,)
        delta_area   = temporal_delta(normalized_area[:, None]) # shape (1,)


        # last K actions
        last_actions = state.last_actions[-K:]  # shape (K, 4)

        obs = jnp.concatenate([
            math_utils.vec(state.quadrotor_state.R),
            state.quadrotor_state.omega,
            points_normalized.flatten(),  # projected points (u, v) for each corner
            center_normalized.flatten(),  # center of the gap in image plane
            # valid_mask.flatten(),
            normalized_area,  # area normalized by image area
            aspect_ratio.flatten(),  # aspect ratio of bounding box
            visibility.flatten(),  # number of visible points (0-4)
            dir_u.flatten(),  # direction of u axis in image plane
            dir_v.flatten(),  # direction of v axis in image plane
            alignment.flatten(),
            delta_center.flatten(),  # delta center
            delta_area.flatten(),  # delta area
            last_actions.flatten(),  # last K actions 4*K=4K
            
        ])  # total obs dim: 9 + 3 + 2*4*K + 2*K + 1*K + 1*K + 1*K + 2*K + 2*K + 1*K + 2 + 1 = 12 + 3 + 18K = 15 + 18K
        
        return obs
    
    @property
    def observation_space(self) -> spaces.Box:
        K = self.num_vis_frames

        low_action = self.action_space.low
        high_action = self.action_space.high

        low = jnp.concatenate(
            [
                # self.world_box.min,  # p (3)
                -jnp.ones(9),  # R_vec (9)
                # self.v_min,  # v (3)
                self.omega_min,  # omega (3)
                # self.world_box.min,  # goal (3)
                -jnp.ones(2 * 4 * K),  # projected points (u, v) for each corner (4) for each frame
                -jnp.ones(2 * K),  # center of the gap in image plane
                -jnp.ones(1 * K),  # normalized area (always >= 0)
                -jnp.ones(1 * K) * 10.0,  # aspect ratio (always >= 0)
                -jnp.ones(1 * K),  # number of visible points (0-4)
                -jnp.ones(2 * K),  # direction of u axis in image plane
                -jnp.ones(2 * K),  # direction of v axis in image plane
                -jnp.ones(1 * K),  # alignment
                -jnp.ones(2),  # delta center (single, last-frame delta)
                -jnp.ones(1),  # delta area (single)
                jnp.tile(low_action, K),  # last K actions
                
            ]
        )
        high = jnp.concatenate(
            [
                # self.world_box.max,  # p (3)
                jnp.ones(9),  # R_vec (9)
                # self.v_max,  # v (3) 
                self.omega_max,  # omega (3)
                # self.world_box.max,  # goal (3)  
                jnp.ones(2 * 4 * K),  # projected points (u, v) for each corner (4) for each frame
                jnp.ones(2 * K),  # center of the gap in image plane
                jnp.ones(1 * K),  # normalized area (max = full image)
                jnp.ones(1 * K) * 10.0,  # aspect ratio (reasonable max)
                jnp.ones(1 * K),  # visibility (max = 1)
                jnp.ones(2 * K),  # direction of u axis in image plane
                jnp.ones(2 * K),  # direction of v axis in image plane
                jnp.ones(1 * K),  # alignment
                jnp.ones(2),  # delta center (single)
                jnp.ones(1),  # delta area (single)
                jnp.tile(high_action, K),  # last K actions
            ]
        )
        shape = (low.shape[0],)
        return spaces.Box(low=low, high=high, shape=shape) # 2 * 4 * num_frames = (u, v) for each box corner (4 corners) for each frame
    
        
    
    # ============================================================================

    # def _get_reward(
    #     self, last_state: EnvState, next_state: EnvState
    # ) -> jax.Array:
        
    #     # Compute individual cost components using the dedicated methods
    #     progress_cost = self.compute_progress_cost(next_state, last_state)
    #     centering_cost = self.compute_centering_cost(next_state)
    #     alignment_cost = self.compute_alignment_cost(next_state)
    #     gap_collision_cost = self.compute_gap_collision_cost(next_state)
    #     stabilization_cost = self.compute_stabilization_cost(next_state)
    #     camera_alignment_cost = self.compute_camera_alignment_cost(next_state)
        
    #     alignment_weight = 0.3
    #     gap_collision_weight = 0.8
    #     # centering_weight = 1.5
    #     progress_weight = 1.4

    #     # TOTAL COST
    #     cost = (
    #         progress_weight * progress_cost 
    #         + centering_cost
    #         + alignment_weight * alignment_cost
    #         + gap_collision_weight * gap_collision_cost
    #         + stabilization_cost
    #         + camera_alignment_cost
    #         )
        

    #     #* FINETUNING -> set to false during training
    #     if finetuning := False:
    #         # Compute finetuning cost components (always computed, but only applied if is_finetuning=True)
    #         max_weight = 5.0
    #         min_weight = 1.0
    #         weight_progress_epoch = 5000
    #         weight_gap_collision = min_weight + (max_weight - min_weight) * (next_state.current_epoch / weight_progress_epoch)
            
    #         p = next_state.quadrotor_state.p
    #         c = next_state.gap_center
    #         u_axis = next_state.gap_u_axis
    #         v_axis = next_state.gap_v_axis
    #         n = next_state.gap_normal
            
    #         # Distance from gap plane
    #         d_curr = jnp.dot(p - c, n)
    #         gaussian_weight = jnp.exp(-15.0 * jnp.square(d_curr))
            
    #         # Horizontal (u) centering: moderate strength
    #         weight_u = 5.0 * gaussian_weight
            
    #         # Vertical (v) centering: MUCH stronger - critical for narrow height tolerance
    #         weight_v = 15.0 * gaussian_weight
            
    #         error_u = jnp.dot(p - c, u_axis)
    #         error_v = jnp.dot(p - c, v_axis)
            
    #         # Use smooth_l1 for soft penalty that doesn't explode
    #         centering_cost_u = weight_u * smooth_l1(self.reward_sharpness_centering * error_u) / self.reward_sharpness_centering
    #         centering_cost_v = weight_v * smooth_l1(self.reward_sharpness_centering * error_v) / self.reward_sharpness_centering
            
    #         finetuning_cost = (gap_collision_cost + centering_cost_u + centering_cost_v) * weight_gap_collision
            
    #         cost += finetuning_cost    

    #     # scale by time
    #     reward = -self.dt * cost
    #     return reward, {
    #         "progress_cost": -self.dt * progress_cost,
    #         "centering_cost": -self.dt * centering_cost,
    #         "alignment_cost": -self.dt * alignment_cost,
    #         "gap_collision_cost": -self.dt * gap_collision_cost,
    #         "stabilization_cost": -self.dt * stabilization_cost,
    #         "camera_alignment_cost": -self.dt * camera_alignment_cost,
    #     }

    # def get_reward_component_keys(self) -> list[str]:
    #     """
    #     Returns the list of reward component keys used in this environment.
    #     This is extracted from the reward_components dictionary in _get_reward().
    #     """
    #     reward_components_keys = [
    #         "progress_cost",
    #         "centering_cost",
    #         "alignment_cost",
    #         "gap_collision_cost",
    #         "stabilization_cost",
    #         "camera_alignment_cost",
    #     ]
    #     return reward_components_keys


