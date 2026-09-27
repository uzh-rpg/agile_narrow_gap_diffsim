"""
Minimal visualization utilities for gap-crossing trajectories.

Provides two visualizations only:
- MultiRolloutVisualizer.plot_gap_crossing_2d:
    2D plot in the gap plane showing every trajectory's drone ellipse
    projected onto the (u, v) gap frame at the moment of plane crossing.
- SingleRolloutVisualizer.plot_bbox_heatmap:
    Heatmap of the projected gap bounding-box centers in the camera image
    plane along a single rollout.

Both visualizers handle static and moving gap environments uniformly: gap
geometry is read at the actual crossing/projection step, so for moving
gaps the per-step gap state is used, while for static gaps the value is
constant.
"""

from typing import Optional

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from flightning.sensors import DoubleSphereCamera, CameraNames


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_step(arr, i, k):
    """Return arr[i, k] for per-step fields (works for both static and moving)."""
    a = np.asarray(arr)
    if a.ndim >= 2 and a.shape[0] > i:
        return np.asarray(a[i, k]).astype(float)
    return np.asarray(a).astype(float)


def _traj_len(done_i):
    """Number of valid steps for a single trajectory's done mask."""
    where = np.where(done_i)[0]
    if where.size == 0:
        return len(done_i)
    return int(where[0]) + 1


# ---------------------------------------------------------------------------
# Multi-rollout: 2D ellipses on gap plane
# ---------------------------------------------------------------------------

class MultiRolloutVisualizer:
    """Generates the gap-plane 2D ellipse plot from a batch of rollouts."""

    def __init__(self, traj, env):
        self.traj = traj
        self.env = env
        self.state = traj.state

        done = np.logical_or(np.asarray(traj.terminated), np.asarray(traj.truncated))
        if done.ndim == 1:
            done = done[None, :]
        self.done = done
        self.num_trajs = done.shape[0]

        self.drone_Q = np.asarray(env.drone_Q)
        self.drone_Q2 = self.drone_Q @ self.drone_Q
        self.cmap = plt.get_cmap("rainbow", max(self.num_trajs, 1))

    # ---- public API -------------------------------------------------------

    def plot_gap_crossing_2d(self, save_path: Optional[str] = None, show: bool = False):
        """Render the 2D gap-plane plot with all crossing ellipses.

        Returns dict with aggregate statistics.
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        cross_plane = 0
        cross_gap = 0
        cross_no_collision = 0
        sum_dist_goal = 0.0
        sum_uv = np.zeros(2)
        sum_dist_center = 0.0
        sum_orient_err = np.zeros(3)  # roll, pitch, yaw

        gap_drawn = False

        for i in range(self.num_trajs):
            done_i = self.done[i] if self.num_trajs > 1 else self.done[0]
            idx = _traj_len(done_i)

            p_all = np.asarray(self.state.quadrotor_state.p)
            R_all_full = np.asarray(self.state.quadrotor_state.R)
            if p_all.ndim == 3 and self.num_trajs > 1:
                p = p_all[i, :idx]
                R_all = R_all_full[i, :idx]
            else:
                p = p_all[:idx] if p_all.ndim == 2 else p_all[i, :idx]
                R_all = R_all_full[:idx] if R_all_full.ndim == 3 else R_all_full[i, :idx]

            goal = _get_step(self.state.goal, i, 0)
            final_dist_to_goal = float(np.linalg.norm(p[-1] - goal))
            color = self.cmap(i)

            # Detect plane crossings using initial gap state
            gap_center0 = _get_step(self.state.gap_center, i, 0)
            gap_normal0 = _get_step(self.state.gap_normal, i, 0)
            gap_normal0 = gap_normal0 / (np.linalg.norm(gap_normal0) + 1e-9)
            d_signed = (p - gap_center0) @ gap_normal0
            sign_change = (d_signed[:-1] * d_signed[1:]) < -1e-9
            cross_ids = np.where(sign_change)[0]
            if cross_ids.size == 0:
                continue

            k_cross = int(cross_ids[0])

            # Read gap state AT the crossing step (per-step for moving, constant for static)
            gap_center = _get_step(self.state.gap_center, i, k_cross)
            gap_normal = _get_step(self.state.gap_normal, i, k_cross)
            gap_normal = gap_normal / (np.linalg.norm(gap_normal) + 1e-9)
            gap_dims = _get_step(self.state.gap_dims, i, k_cross)
            gap_u_axis = _get_step(self.state.gap_u_axis, i, k_cross)
            gap_v_axis = _get_step(self.state.gap_v_axis, i, k_cross)
            gap_half = gap_dims / 2.0

            if not gap_drawn:
                self._draw_gap_frame(ax, gap_half)
                gap_drawn = True

            # Interpolate to the exact crossing point in 3D
            denom = d_signed[k_cross + 1] - d_signed[k_cross]
            if abs(denom) < 1e-12:
                continue
            s = float(np.clip(-d_signed[k_cross] / denom, 0.0, 1.0))
            p_cross = p[k_cross] + s * (p[k_cross + 1] - p[k_cross])

            # Project crossing point onto the gap plane (uv coords)
            rel = p_cross - gap_center
            u = float(rel @ gap_u_axis)
            v = float(rel @ gap_v_axis)

            R_cross = R_all[min(k_cross + 1, len(R_all) - 1)]

            # Ellipse: project drone ellipsoid onto gap plane (u, v)
            B_proj = np.stack([gap_u_axis, gap_v_axis], axis=1)
            A = R_cross @ self.drone_Q2 @ R_cross.T
            try:
                M = np.linalg.inv(B_proj.T @ A @ B_proj)
            except np.linalg.LinAlgError:
                continue
            eigvals, eigvecs = np.linalg.eigh(M)
            if not np.all(eigvals > 1e-9):
                continue

            sa1, sa2 = 1.0 / np.sqrt(eigvals[0]), 1.0 / np.sqrt(eigvals[1])
            angle_deg = float(np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0])))
            ellipse = patches.Ellipse(
                xy=(u, v), width=2.0 * sa1, height=2.0 * sa2,
                angle=angle_deg, facecolor=color, alpha=0.20, zorder=5,
            )
            ax.add_patch(ellipse)
            ax.scatter(u, v, color=color, s=20, zorder=10,
                       label="Drone crossing" if i == 0 else None)

            # Drone body axes projected on gap plane
            axes_world = R_cross @ self.drone_Q
            for axis_idx, (c, lab) in enumerate(
                [("red", "Drone x"), ("green", "Drone y"), ("blue", "Drone z")]
            ):
                aw = axes_world[:, axis_idx]
                au = float(np.dot(aw, gap_u_axis))
                av = float(np.dot(aw, gap_v_axis))
                ax.arrow(u, v, 0.7 * au, 0.7 * av,
                         head_width=0.01, head_length=0.01, fc=c, ec=c,
                         linewidth=1.2, alpha=0.7,
                         label=lab if i == 0 else None, zorder=8)

            # Statistics
            cross_plane += 1
            sum_dist_goal += final_dist_to_goal
            sum_uv += np.array([abs(u), abs(v)])
            sum_dist_center += float(np.hypot(u, v))

            gap_R = np.stack([gap_normal, gap_u_axis, gap_v_axis], axis=1)
            R_rel = gap_R.T @ R_cross
            roll, pitch, yaw = self._rotation_to_rpy(R_rel)
            sum_orient_err += np.array([abs(roll), abs(pitch), abs(yaw)])

            inside = (-gap_half[0] < u < gap_half[0]) and (-gap_half[1] < v < gap_half[1])
            if inside:
                cross_gap += 1
                if not self._has_collision(p_cross, R_cross, gap_center,
                                           gap_u_axis, gap_v_axis, gap_half):
                    cross_no_collision += 1

        denom_cross = max(1, cross_plane)
        stats = {
            "num_trajs": int(self.num_trajs),
            "success_rate": cross_no_collision / self.num_trajs * 100.0,
            "cross_gap_rate": cross_gap / self.num_trajs * 100.0,
            "cross_plane_rate": cross_plane / self.num_trajs * 100.0,
            "avg_dist_to_goal_cm": 100.0 * sum_dist_goal / denom_cross,
            "avg_dist_center_cm": 100.0 * sum_dist_center / denom_cross,
            "avg_u_cm": 100.0 * sum_uv[0] / denom_cross,
            "avg_v_cm": 100.0 * sum_uv[1] / denom_cross,
            "avg_roll_deg": float(np.rad2deg(sum_orient_err[0] / denom_cross)),
            "avg_pitch_deg": float(np.rad2deg(sum_orient_err[1] / denom_cross)),
            "avg_yaw_deg": float(np.rad2deg(sum_orient_err[2] / denom_cross)),
        }

        ax.set_title(
            f"Success (no collision): {stats['success_rate']:.2f}%  |  "
            f"Cross gap: {stats['cross_gap_rate']:.2f}%  |  "
            f"Cross plane: {stats['cross_plane_rate']:.2f}%\n"
            f"Avg dist to goal: {stats['avg_dist_to_goal_cm']:.1f} cm  |  "
            f"Avg dist from center: {stats['avg_dist_center_cm']:.1f} cm  "
            f"(u={stats['avg_u_cm']:.1f}cm, v={stats['avg_v_cm']:.1f}cm)\n"
            f"Avg orient errors - R: {stats['avg_roll_deg']:.2f}°, "
            f"P: {stats['avg_pitch_deg']:.2f}°, Y: {stats['avg_yaw_deg']:.2f}°"
        )
        ax.set_xlabel("u (gap width axis)")
        ax.set_ylabel("v (gap height axis)")
        ax.set_aspect("equal")
        ax.invert_xaxis()  # match debug visualization convention
        ax.legend(loc="best", fontsize=8)
        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"[viz] saved gap-crossing 2D plot -> {save_path}")
        if show:
            plt.show()
        plt.close(fig)
        return stats

    # ---- internal helpers -------------------------------------------------

    @staticmethod
    def _draw_gap_frame(ax, gap_half):
        rect_x = np.array([-gap_half[0], gap_half[0], gap_half[0], -gap_half[0], -gap_half[0]])
        rect_y = np.array([-gap_half[1], -gap_half[1], gap_half[1], gap_half[1], -gap_half[1]])
        ax.plot(rect_x, rect_y, linestyle="--", linewidth=1.2, alpha=0.4,
                color="black", label="Gap edges")
        ax.scatter(0.0, 0.0, marker="x", color="black", s=60, alpha=0.7,
                   label="Gap frame")
        scale = float(np.linalg.norm(gap_half) * 0.3)
        ax.arrow(0, 0, scale, 0, head_width=0.02, head_length=0.02,
                 fc="orange", ec="orange", linewidth=2, label="Gap u-axis")
        ax.arrow(0, 0, 0, scale, head_width=0.02, head_length=0.02,
                 fc="brown", ec="brown", linewidth=2, label="Gap v-axis")

    @staticmethod
    def _rotation_to_rpy(R):
        pitch = np.arcsin(np.clip(-R[2, 0], -1.0, 1.0))
        if abs(np.cos(pitch)) > 1e-6:
            roll = np.arctan2(R[2, 1], R[2, 2])
            yaw = np.arctan2(R[1, 0], R[0, 0])
        else:
            roll = 0.0
            yaw = np.arctan2(-R[0, 1], R[1, 1])
        return float(roll), float(pitch), float(yaw)

    def _has_collision(self, p, R, gap_center, gap_u, gap_v, gap_half):
        """Check ellipsoid-vs-gap-boundary collision (matches env collision check)."""
        A = np.stack([gap_u, -gap_u, gap_v, -gap_v])
        w_half, h_half = float(gap_half[0]), float(gap_half[1])
        b = np.array([
            w_half + np.dot(gap_u, gap_center),
            w_half - np.dot(gap_u, gap_center),
            h_half + np.dot(gap_v, gap_center),
            h_half - np.dot(gap_v, gap_center),
        ])
        RQ = R @ self.drone_Q
        norms = np.linalg.norm(A @ RQ, axis=1)
        coll = A @ p + norms - b
        return bool(np.any(np.maximum(coll, 0.0) > 0.0))


# ---------------------------------------------------------------------------
# Single-rollout: bbox center heatmap on the camera image plane
# ---------------------------------------------------------------------------

class SingleRolloutVisualizer:
    """Generates the gap bounding-box heatmap for a single rollout.

    The heatmap simulates what the on-board camera would see (regardless of
    whether the policy is state- or vision-based).
    """

    def __init__(self, env, state, traj_idx: int = 0):
        self.env = env

        # Extract single-trajectory arrays
        p = np.asarray(state.quadrotor_state.p)
        R = np.asarray(state.quadrotor_state.R)
        if p.ndim == 3:
            p = p[traj_idx]
            R = R[traj_idx]
        # Determine valid length using terminated/truncated if available
        try:
            term = np.asarray(state.terminated) if hasattr(state, "terminated") else None
        except Exception:
            term = None
        idx = len(p)
        if term is not None and term.ndim >= 1:
            done_i = term[traj_idx] if term.ndim > 1 else term
            where = np.where(done_i)[0]
            if where.size > 0:
                idx = int(where[0]) + 1

        self.positions = p[:idx]
        self.rotations = R[:idx]
        self.state = state
        self.traj_idx = traj_idx

        # Camera: use the canonical drone camera (same across all tasks).
        self.cam = getattr(env, "cam", None) or DoubleSphereCamera.from_name(
            CameraNames.EXAMPLE_QUAD
        )

    def _gap_corners_at(self, k):
        gc = _get_step(self.state.gap_center, self.traj_idx, k)
        gd = _get_step(self.state.gap_dims, self.traj_idx, k)
        gu = _get_step(self.state.gap_u_axis, self.traj_idx, k)
        gv = _get_step(self.state.gap_v_axis, self.traj_idx, k)
        hw, hh = gd[0] / 2.0, gd[1] / 2.0
        return np.array([
            gc + hw * gu + hh * gv,
            gc - hw * gu + hh * gv,
            gc - hw * gu - hh * gv,
            gc + hw * gu - hh * gv,
        ]), gc

    def plot_bbox_heatmap(self, save_path: Optional[str] = None, show: bool = False):
        """Plot all visible gap-bbox projections in the camera image plane."""
        fig, ax = plt.subplots(figsize=(10, 10), facecolor="white")
        ax.set_facecolor("white")

        cam = self.cam
        T = len(self.positions)

        centers_u, centers_v = [], []
        bbox_corners_list = []
        valid_steps = []
        num_visible_per_frame = []

        for t in range(T):
            corners_3d, _ = self._gap_corners_at(t)
            projected = np.asarray(
                cam.project_points_with_pose(corners_3d, self.positions[t], self.rotations[t])
            )
            visible = projected[:, 2] > 0.5
            num_vis = int(visible.sum())
            if num_vis == 0:
                continue
            centers_u.append(float(np.mean(projected[:, 0])))
            centers_v.append(float(np.mean(projected[:, 1])))
            bbox_corners_list.append(projected[:, :2])
            valid_steps.append(t)
            num_visible_per_frame.append(num_vis)

        if not centers_u:
            ax.text(0.5, 0.5, "No valid projections", transform=ax.transAxes,
                    ha="center", va="center", fontsize=14)
            ax.set_xlim([0, cam.width]); ax.set_ylim([cam.height, 0])
            ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
            ax.set_title("Gap Bbox Center Heatmap (Image Plane)")
            if save_path is not None:
                plt.savefig(save_path, bbox_inches="tight")
                print(f"[viz] saved bbox heatmap -> {save_path}")
            if show:
                plt.show()
            plt.close(fig)
            return

        centers_u = np.array(centers_u)
        centers_v = np.array(centers_v)
        valid_steps = np.array(valid_steps)
        colors = cm.viridis(valid_steps / max(T - 1, 1))

        # Right colorbar shows the timestep mapping.
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = ColorbarBase(cax, cmap=cm.viridis,
                            norm=Normalize(vmin=0, vmax=max(T - 1, 1)),
                            orientation="vertical")
        cbar.set_label("Timestep", fontsize=12)

        # Bboxes (truncated rectangles allowed) + corner markers.
        for i, corners in enumerate(bbox_corners_list):
            poly = plt.Polygon(corners, fill=False, edgecolor=colors[i],
                               linewidth=1, alpha=0.3)
            ax.add_patch(poly)
            for c in corners:
                ax.plot(c[0], c[1], "x", color=colors[i], markersize=2, alpha=0.5)

        # Center trajectory.
        for i in range(len(centers_u)):
            ax.plot(centers_u[i], centers_v[i], "o", color=colors[i], markersize=3)

        ax.plot(centers_u[0], centers_v[0], "o", color=colors[0], markersize=10,
                markeredgecolor="black", markeredgewidth=0.5,
                label=f"Start (t={valid_steps[0]})", zorder=10)

        # Project the (final) gap center for reference.
        last_corners, last_gc = self._gap_corners_at(T - 1)
        gc_proj = np.asarray(
            cam.project_points_with_pose(last_gc[None, :], self.positions[-1], self.rotations[-1])
        )
        if gc_proj[0, 2] > 0.5:
            ax.plot(gc_proj[0, 0], gc_proj[0, 1], "*", color="yellow",
                    markersize=20, label="Gap Center (goal)", zorder=11,
                    markeredgecolor="black", markeredgewidth=0.5)

        visibility_pct = len(centers_u) / T * 100.0
        avg_visible = float(np.mean(num_visible_per_frame))
        ax.set_title(
            f"Gap Bbox Center Heatmap (Image Plane)\n"
            f"{len(centers_u)}/{T} frames visible ({visibility_pct:.1f}%) | "
            f"avg {avg_visible:.1f}/4 corners"
        )
        ax.set_xlim([0, cam.width])
        ax.set_ylim([cam.height, 0])
        ax.set_xlabel("u [px]")
        ax.set_ylabel("v [px]")
        ax.set_aspect("equal")
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
            print(f"[viz] saved bbox heatmap -> {save_path}")
        if show:
            plt.show()
        plt.close(fig)
