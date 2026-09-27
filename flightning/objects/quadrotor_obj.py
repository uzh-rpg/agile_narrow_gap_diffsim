from functools import partial
import os
import yaml

import chex
import jax
import jax_dataclasses as jdc
from jax import numpy as jnp

from flightning.objects.quadrotor_simple_obj import quadrotor_dyn_rk4 as lowfid_dynamics_rk4
from flightning.utils.pytrees import CustomPyTree, field_jnp


@jdc.pytree_dataclass
class QuadrotorState(CustomPyTree):
    p: jax.Array = field_jnp([0.0, 0.0, 0.0])
    R: jax.Array = field_jnp(jnp.eye(3))
    v: jax.Array = field_jnp([0.0, 0.0, 0.0])
    omega: jax.Array = field_jnp([0.0, 0.0, 0.0])
    domega: jax.Array = field_jnp([0.0, 0.0, 0.0])
    motor_omega: jax.Array = field_jnp([0.0, 0.0, 0.0, 0.0])
    acc: jax.Array = field_jnp([0.0, 0.0, 0.0])
    dr_key: chex.PRNGKey = field_jnp(jax.random.key(0))

    def detached(self):
        return QuadrotorState(
            p=jax.lax.stop_gradient(self.p),
            R=jax.lax.stop_gradient(self.R),
            v=jax.lax.stop_gradient(self.v),
            omega=jax.lax.stop_gradient(self.omega),
            domega=jax.lax.stop_gradient(self.domega),
            motor_omega=jax.lax.stop_gradient(self.motor_omega),
            acc=jax.lax.stop_gradient(self.acc),
            dr_key=jax.lax.stop_gradient(self.dr_key),
        )

    def as_vector(self):
        return jnp.concatenate(
            [
                self.p,
                self.R.flatten(),
                self.v,
                self.omega,
                self.domega,
                self.motor_omega,
            ]
        )

    @classmethod
    def from_vector(cls, vector):
        p = vector[:3]
        R = vector[3:12].reshape(3, 3)
        v = vector[12:15]
        omega = vector[15:18]
        domega = vector[18:21]
        motor_omega = vector[21:25]
        return cls(p=p, R=R, v=v, omega=omega, domega=domega, motor_omega=motor_omega)


class Quadrotor:
    """Low-fidelity quadrotor model integrated with RK4 dynamics only.

    This class keeps the public interface used by the environments, but removes
    residual dynamics, high-fidelity motor dynamics, and all project-specific
    dependencies.
    """

    def __init__(
        self,
        *,
        drone_name="example_quad",
        mass=0.75,
        tbm_fr=jnp.array([0.075, -0.10, 0.0]),
        tbm_bl=jnp.array([-0.075, 0.10, 0.0]),
        tbm_br=jnp.array([-0.075, -0.10, 0.0]),
        tbm_fl=jnp.array([0.075, 0.10, 0.0]),
        inertia=jnp.array([0.002410, 0.001800, 0.003759]),
        motor_omega_min=150.0,
        motor_omega_max=2800.0,
        motor_tau=0.033,
        motor_inertia=5.64e-6,
        omega_max=jnp.array([10.0, 10.0, 4.0]),
        thrust_map=jnp.array([1.562522e-6, 0.0, 0.0]),
        kappa=0.022,
        thrust_min=0.0,
        thrust_max=8.5,
        rotors_config="cross",
        dt_low_level=0.001,
        sim_dyn_config=None,
    ):
        assert rotors_config == "cross", "Only cross rotors configuration is supported"

        if sim_dyn_config is not None:
            unsupported_flags = [
                name
                for name in ("high_fidelity_forward", "forward_residual", "backprop_residual")
                if sim_dyn_config.get(name, False)
            ]
            if unsupported_flags:
                raise ValueError(
                    "This public Quadrotor implementation only supports low-fidelity RK4 "
                    f"dynamics. Unsupported flags: {unsupported_flags}"
                )

        self._drone_name = drone_name
        self._mass = mass
        self._tbm_fr = tbm_fr
        self._tbm_bl = tbm_bl
        self._tbm_br = tbm_br
        self._tbm_fl = tbm_fl
        self._inertia = inertia
        self._motor_omega_min = motor_omega_min
        self._motor_omega_max = motor_omega_max
        self._motor_tau = motor_tau
        self._motor_inertia = motor_inertia
        self._omega_max = omega_max
        self._thrust_map = thrust_map
        self._kappa = kappa
        self._thrust_min = thrust_min
        if thrust_min <= 0.0:
            self._thrust_min += thrust_map[0] * motor_omega_min**2
        self._thrust_max = thrust_max
        self._rotors_config = rotors_config
        self._dt_low_level = dt_low_level
        self._gravity = jnp.array([0.0, 0.0, -9.81])

        self.high_fidelity_forward = False
        self.forward_residual = False
        self.backprop_residual = False

    @classmethod
    def from_name(cls, name: str, dyn_config=None) -> "Quadrotor":
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, "quadrotor_files")

        if name == "example_quad":
            path = os.path.join(filename, "example_quad.yaml")
        else:
            raise ValueError(
                f'Unknown quadrotor name: {name}, only "example_quad" is supported'
            )

        return cls.from_yaml(path, dyn_config)

    @classmethod
    def from_yaml(cls, path: str, dyn_config=None) -> "Quadrotor":
        with open(path) as stream:
            config = yaml.safe_load(stream)
        return cls.from_dict(config, dyn_config)

    @classmethod
    def from_dict(cls, config: dict, dyn_config=None) -> "Quadrotor":
        return cls(
            drone_name=config["name"],
            mass=config["mass"],
            tbm_fr=jnp.array(config["tbm_fr"]),
            tbm_bl=jnp.array(config["tbm_bl"]),
            tbm_br=jnp.array(config["tbm_br"]),
            tbm_fl=jnp.array(config["tbm_fl"]),
            inertia=jnp.array(config["inertia"]),
            motor_omega_min=config["motor_omega_min"],
            motor_omega_max=config["motor_omega_max"],
            motor_tau=config["motor_tau"],
            motor_inertia=config["motor_inertia"],
            omega_max=jnp.array(config["omega_max"]),
            thrust_map=jnp.array(config["thrust_map"]),
            kappa=config["kappa"],
            thrust_min=config["thrust_min"],
            thrust_max=config["thrust_max"],
            rotors_config=config["rotors_config"],
            sim_dyn_config=dyn_config,
        )

    @classmethod
    def default_quadrotor(cls) -> "Quadrotor":
        return cls.from_name("example_quad")

    @property
    def hovering_motor_speed(self) -> float:
        return jnp.sqrt(self._mass * 9.81 / (4 * self._thrust_map[0]))

    def default_state(self):
        hovering_motor_speeds = jnp.ones(4) * self.hovering_motor_speed
        return QuadrotorState(motor_omega=hovering_motor_speeds)

    def create_state(self, p, R, v, **kwargs):
        if "motor_omega" not in kwargs:
            kwargs["motor_omega"] = jnp.ones(4) * self.hovering_motor_speed
        return QuadrotorState(p=p, R=R, v=v, **kwargs)

    @property
    def allocation_matrix(self):
        rotor_coordinates = jnp.stack(
            [self._tbm_fr, self._tbm_bl, self._tbm_br, self._tbm_fl]
        )
        x = rotor_coordinates[:, 0]
        y = rotor_coordinates[:, 1]

        return jnp.array(
            [
                [1.0, 1.0, 1.0, 1.0],
                y,
                -x,
                self._kappa * jnp.array([-1.0, -1.0, 1.0, 1.0]),
            ],
            dtype=jnp.float32,
        )

    @property
    def inertial_matrix(self):
        return jnp.diag(self._inertia)

    @property
    def allocation_matrix_inv(self):
        return jnp.linalg.inv(self.allocation_matrix)

    def step(
        self,
        state: QuadrotorState,
        f_d: jax.Array,
        omega_d: jax.Array,
        dt: jax.Array,
        res_model_params=None,
    ) -> QuadrotorState:
        del res_model_params

        dr_key, key_f, _key_omega = jax.random.split(state.dr_key, 3)
        f_noise = jax.random.uniform(key_f, minval=-0.10, maxval=0.10)
        f_d = f_d * (1.0 + f_noise)
        dt = jnp.maximum(jnp.round(dt, 5), 0.001)

        @jax.custom_jvp
        def _lowfid_step(p, R, v, omega, f_d, omega_d, dt):
            del omega
            dt_stopped = jax.lax.stop_gradient(dt)
            p_new, R_new, v_new = self.lowfid_dynamics_simple(
                p,
                R,
                v,
                f_d / self._mass,
                omega_d,
                dt_stopped,
                method="rk4",
            )
            return p_new, R_new, v_new, omega_d

        @_lowfid_step.defjvp
        def _lowfid_step_jvp(primals, tangents):
            p, R, v, omega, f_d, omega_d, dt = primals
            p_dot, R_dot, v_dot, omega_dot, f_d_dot, omega_d_dot, _dt_dot = tangents

            primal_out = _lowfid_step(p, R, v, omega, f_d, omega_d, dt)
            lowfid_rk4_step = partial(self.lowfid_dynamics_simple, dt=dt, method="rk4")
            primals_simple = (p, R, v, f_d / self._mass, omega_d)
            tangents_simple = (p_dot, R_dot, v_dot, f_d_dot / self._mass, omega_d_dot)
            _, (p_tan, R_tan, v_tan) = jax.jvp(
                lowfid_rk4_step, primals_simple, tangents_simple
            )
            tangent_out = (p_tan, R_tan, v_tan, omega_dot)
            return primal_out, tangent_out

        p_new, R_new, v_new, omega_new = _lowfid_step(
            state.p, state.R, state.v, state.omega, f_d, omega_d, dt
        )
        return state.replace(p=p_new, R=R_new, v=v_new, omega=omega_new, dr_key=dr_key)

    @partial(jax.jit, static_argnums=(0, 7))
    def lowfid_dynamics_simple(
        self,
        p: jax.Array,
        R: jax.Array,
        v: jax.Array,
        a: jax.Array,
        omega: jax.Array,
        dt: jax.Array,
        method: str = "rk4",
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        if method != "rk4":
            raise ValueError('This Quadrotor only supports method="rk4"')
        return lowfid_dynamics_rk4(p, R, v, a, omega, dt)

    def print_config(self):
        print(f"[QUAD OBJ] Drone name: {self._drone_name}")
        print("[QUAD OBJ] Dynamics: low-fidelity RK4")
        print("[QUAD OBJ] Use high-fidelity forward: False")
        print("[QUAD OBJ] Use forward residual: False")
        print("[QUAD OBJ] Use backprop residual: False")


if __name__ == "__main__":
    quad = Quadrotor(mass=1.0)
    state = QuadrotorState(motor_omega=jnp.ones(4) * 1200, dr_key=jax.random.key(0))
    state_new = quad.step(state, jnp.array(9.81), jnp.array([0.0, 0.0, 0.01]), 1.0)
    print(state_new.p)
    print(state_new.R)
