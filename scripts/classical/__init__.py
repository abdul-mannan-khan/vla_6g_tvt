# Classical Optimization Module for UAV Relay Positioning
from .sca_solver import SCASolver, SCAConfig, solve_with_sca
from .analytical_gradients import (
    compute_throughput_gradient,
    compute_fairness_gradient,
    compute_composite_gradient,
    GradientOracle
)
