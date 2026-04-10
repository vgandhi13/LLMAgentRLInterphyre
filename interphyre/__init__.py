"""Interphyre - Physics-based puzzle environment for reinforcement learning.

Example usage:
    from interphyre import InterphyreEnv

    env = InterphyreEnv("catapult", seed=42, render_mode="human")
    obs, info = env.reset()
    obs, reward, term, trunc, info = env.step([(0.5, 3.0, 0.6)])
"""

__version__ = "0.0.1"

from interphyre.environment import InterphyreEnv, InterventionContext
from interphyre.level import Level
from interphyre.config import SimulationConfig
from interphyre.levels import list_levels

__all__ = [
    "InterphyreEnv",
    "InterventionContext",
    "Level",
    "SimulationConfig",
    "list_levels",
    "__version__",
]
