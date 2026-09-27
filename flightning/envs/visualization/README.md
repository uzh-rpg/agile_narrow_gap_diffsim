# Visualization Module

This directory contains visualization tools for analyzing drone trajectories in the static gap environment.

## Structure

```
visualization/
├── __init__.py                      # Module exports
├── visualize_static_narrow_gap.py  # Main dashboard classes
├── cost_extraction.py              # Cost extraction utilities
└── README.md                       # This file
```

## Dashboard Classes

### `MultiRolloutVisualizationDashboard`

For analyzing **multiple rollout trajectories**.

**Features:**
- Gap crossing statistics with collision detection
- 2D and 3D trajectory visualization
- XY/XZ projection views
- Action statistics across rollouts (mean ± std)
- Success rate analysis

**Usage:**
```python
from flightning.envs.visualization import MultiRolloutVisualizationDashboard

dashboard = MultiRolloutVisualizationDashboard(traj, env)

# Generate various plots
dashboard.generate_gap_crossing_plot(save_path="gap_crossing.png")
dashboard.generate_trajectories_plot(save_path="trajectories.png")
dashboard.generate_3d_trajectories_plot(save_path="3d_trajectories.png")
dashboard.generate_actions_statistics_plot(save_path="actions_stats.png")
```

### `SingleRolloutVisualizationDashboard`

For **detailed single trajectory analysis**.

**Features:**
- 3D trajectory with gap geometry
- X-Z projection view with gap crossing details
- Individual cost components visualization
- Position and rotation gradients
- Total cost analysis
- Actions for single trajectory
- Interactive legend toggling

**Usage:**
```python
from flightning.envs.visualization import SingleRolloutVisualizationDashboard

# Collect trajectory data and costs
# ... (run simulation, collect positions, rotations, costs, gradients)

dashboard = SingleRolloutVisualizationDashboard(
    env, state, positions, rotations,
    cost_data, grad_p_data, grad_R_data,
    enabled_costs
)

# Generate complete dashboard
dashboard.generate_and_display(show=True, save_path="cost_analysis.png")

# Generate only actions plot
dashboard.generate_actions_plot(actions, save_path="actions.png")
```

## Cost Extraction Functions

Located in `cost_extraction.py`, these functions extract individual cost components:

```python
from flightning.envs.visualization import (
    extract_progress_cost,
    extract_centering_cost,
    extract_alignment_cost,
    extract_alignment_reward,
    extract_gap_collision_cost,
    extract_stabilization_cost,
    extract_wall_collision_cost,
)
```

## Legacy Compatibility

Legacy aliases and wrapper functions are maintained for backward compatibility:

- `TrajectoryVisualizationDashboard` → `MultiRolloutVisualizationDashboard`
- `CostVisualizationDashboard` → `SingleRolloutVisualizationDashboard`
- `plot_gap_crossing()` - wrapper function
- `plot_trajectories()` - wrapper function
- `plot_gap_trajectory()` - wrapper function
- `plot_full_trajectory_3d()` - wrapper function

## Integration Examples

### In Training Script (`train_bptt_static_gap_state.py`)

```python
from flightning.envs.visualization import (
    MultiRolloutVisualizationDashboard,
    SingleRolloutVisualizationDashboard,
    extract_progress_cost,
    # ... other cost functions
)

# Multi-rollout analysis
transitions = get_rollouts(env, policy, num_rollouts, key)
dashboard = MultiRolloutVisualizationDashboard(transitions, env)
dashboard.generate_gap_crossing_plot(save_path=eval_dir / "gap_crossing.png")
dashboard.generate_actions_statistics_plot(save_path=eval_dir / "actions_stats.png")

# Single-rollout analysis with costs
# ... (collect positions, rotations, costs, gradients)
dashboard = SingleRolloutVisualizationDashboard(
    env, state, positions, rotations,
    cost_data, grad_p_data, grad_R_data, enabled_costs
)
dashboard.generate_and_display(save_path=eval_dir / "cost_dashboard.png")
```

### In Debug Script (`debug_collision_grads.py`)

```python
from flightning.envs.visualization import (
    SingleRolloutVisualizationDashboard,
    extract_progress_cost,
    # ... other cost functions
)

# Run trajectory and collect data
# ...

# Visualize costs and gradients
dashboard = SingleRolloutVisualizationDashboard(
    env, state, positions, rotations,
    cost_data, grad_p_data, grad_R_data, enabled_costs
)
dashboard.generate_and_display(show=True, save_path="debug_output.png")
```

## Migration Guide

If you have existing code using the old structure:

### Before (scattered imports)
```python
from flightning.envs.visualization.visualize_static_narrow_gap import plot_gap_crossing
from debug_collision_grads import CostVisualizationDashboard, extract_progress_cost
```

### After (unified module)
```python
from flightning.envs.visualization import (
    MultiRolloutVisualizationDashboard,  # or legacy alias TrajectoryVisualizationDashboard
    SingleRolloutVisualizationDashboard,  # or legacy alias CostVisualizationDashboard
    plot_gap_crossing,                     # legacy wrapper still available
    extract_progress_cost,
)
```

All existing code should continue to work without changes thanks to legacy aliases!
