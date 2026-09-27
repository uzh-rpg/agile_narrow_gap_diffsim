# Flightning Environments

Gap-crossing environments for training quadrotor policies. Each environment
combines a gap configuration (static or pendulum-driven) with an observation
modality (full state or onboard vision).

## Environments

| Environment           | Gap     | Observations                                                       |
|-----------------------|---------|--------------------------------------------------------------------|
| `Env`                 | -       | Base class (see `env_base.py`)                                     |
| `StaticGapStateEnv`   | Static  | Quadrotor state (p, R, v, ω), gap pose, action/state history       |
| `StaticGapVisionEnv`  | Static  | Onboard camera features + quadrotor state history                  |
| `MovingGapStateEnv`   | Moving  | Quadrotor state + pendulum state (θ, θ̇, L, pivot)                |
| `MovingGapVisionEnv`  | Moving  | Onboard camera features + state history                            |

The "moving" variants extend the static ones by attaching the gap to a
pendulum that swings during the episode.

