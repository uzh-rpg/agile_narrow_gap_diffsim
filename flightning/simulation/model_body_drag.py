import jax
import jax.numpy as jnp

from typing import NamedTuple
import yaml


class BodyDragParams(NamedTuple):
    horizontal_drag_coefficient: float = 1.04
    vertical_drag_coefficient: float = 1.04
    frontarea_x: float = 1.5e-2
    frontarea_y: float = 1.5e-2
    frontarea_z: float = 3.0e-2
    air_density: float = 1.204


def compute_drag_force(state, key, params: BodyDragParams) -> jnp.ndarray:
    # unpack state
    v = state.v
    R = state.R

    # compute drag force
    v_body = R.T @ v
    rho = params.air_density
    area = jnp.array(
        [params.frontarea_x, params.frontarea_y, params.frontarea_z]
    )
    drag_coeff = jnp.array(
        [
            params.horizontal_drag_coefficient,
            params.horizontal_drag_coefficient,
            params.vertical_drag_coefficient,
        ]
    )

    # domain randomization
    drag_coeff = jax.random.uniform(
        key, drag_coeff.shape, minval=0.5 * drag_coeff, maxval=1.5 * drag_coeff
    )

    f_drag = -0.5 * rho * drag_coeff * area * v_body * jnp.abs(v_body)

    return f_drag
