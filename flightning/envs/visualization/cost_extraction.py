"""
Cost extraction utilities for static gap environment.

This module provides helper functions to extract individual cost components
from environment states. These functions delegate to the environment's internal
cost computation methods for consistency.
"""


def extract_progress_cost(env, state, last_state):
    """Progress cost: distance from drone to goal along gap normal."""
    return env.compute_progress_cost(state)


def extract_centering_cost(env, state, last_state):
    """Centering cost: distance from gap center in u and v directions."""
    return env.compute_centering_cost(state)

def extract_near_gap_centering_cost(env, state, last_state):
    """Near-gap centering cost: increased penalty for being off-center near the gap."""
    return env.compute_near_gap_centering_cost(state)


def extract_alignment_cost(env, state, last_state):
    """Alignment cost: rotation misalignment between drone and gap."""
    return env.compute_alignment_cost(state)


def extract_gap_collision_cost(env, state, last_state):
    """Gap collision cost: penetration into gap boundaries."""
    return env.compute_gap_collision_cost(state)


def extract_stabilization_cost(env, state, last_state):
    """Stabilization cost: velocity, angular velocity, acceleration, and action penalties."""
    return env.compute_stabilization_cost(state, last_state)


def extract_camera_alignment_cost(env, state, last_state):
    """Camera alignment cost: misalignment of gap in camera frame."""
    return env.compute_camera_alignment_cost(state)

