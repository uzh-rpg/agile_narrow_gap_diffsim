import jax
import jax_dataclasses as jdc

from flightning.utils.math import rotation_matrix_from_vector
from flightning.utils.pytrees import field_jnp, CustomPyTree
from jax import numpy as jnp 


@jdc.pytree_dataclass
class QuadrotorSimpleState(CustomPyTree):
    p: jax.Array = field_jnp([0.0, 0.0, 0.0])
    R: jax.Array = field_jnp(jnp.eye(3))
    v: jax.Array = field_jnp([0.0, 0.0, 0.0])


def quadrotor_dyn_euler(
    p: jax.Array,
    R: jax.Array,
    v: jax.Array,
    a: jax.Array,
    omega: jax.Array,
    dt: jax.Array,
    gravity: jax.Array = jnp.array([0, 0, -9.81]),
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """
    Quadrotor dynamics model.
    :param p: position
    :param R: orientation matrix
    :param v: velocity
    :param a: acceleration
    :param omega: body rates
    :param dt: time step
    :param gravity: gravity vector
    :return: new position, orientation matrix, and velocity
    """
    # Euler step for position and velocity
    p_new = p + dt * v
    v_new = v + dt * (gravity + R @ jnp.array([0, 0, a]))
    # Exact step for orientation
    R_delta = rotation_matrix_from_vector(dt * omega)
    R_new = R @ R_delta

    return p_new, R_new, v_new


def quadrotor_dyn_rk4(
    p: jax.Array,
    R: jax.Array,
    v: jax.Array,
    a: jax.Array,
    omega: jax.Array,
    dt: jax.Array,
    gravity: jax.Array = jnp.array([0, 0, -9.81]),
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """
    Quadrotor dynamics with RK4 integration for position and velocity.
    :param p: position
    :param R: orientation matrix
    :param v: velocity
    :param a: acceleration in body z direction
    :param omega: body rates
    :param dt: time step
    :param gravity: gravity vector
    :return: new position, orientation matrix, and velocity
    """

    def dynamics(p, v):
        dvdt = gravity + R @ jnp.array([0.0, 0.0, a])
        dpdt = v
        return dpdt, dvdt

    # RK4 for position and velocity
    k1_p, k1_v = dynamics(p, v)
    k2_p, k2_v = dynamics(p + 0.5 * dt * k1_p, v + 0.5 * dt * k1_v)
    k3_p, k3_v = dynamics(p + 0.5 * dt * k2_p, v + 0.5 * dt * k2_v)
    k4_p, k4_v = dynamics(p + dt * k3_p, v + dt * k3_v)

    p_new = p + (dt / 6.0) * (k1_p + 2*k2_p + 2*k3_p + k4_p)
    v_new = v + (dt / 6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)

    # Exact step for orientation
    R_delta = rotation_matrix_from_vector(dt * omega)
    R_new = R @ R_delta

    return p_new, R_new, v_new



class QuadrotorSimple:
    """
    Simplified quadrotor model. No low-level control or aerodynamic effects.
    """

    def __init__(self, mass=0.752):
        """
        :param mass: quadrotor mass in kg
        """
        self.mass = mass

    def step(
        self,
        state: QuadrotorSimpleState,
        f: jax.Array,
        omega: jax.Array,
        dt: jax.Array,
        gravity: jax.Array = jnp.array([0, 0, -9.81]),
    ) -> QuadrotorSimpleState:
        """
        :param state: quadrotor state
        :param f: cumulative thrust
        :param omega: body rates
        :param dt: time step length in s
        :param gravity: gravity vector
        :return: next state of the quadrotor
        """
        a = f / self.mass
        p, R, v = quadrotor_dyn_rk4(
            state.p, state.R, state.v, a, omega, dt, gravity
        )
        # noinspection PyArgumentList
        next_state = state.replace(p=p, R=R, v=v)
        return next_state
