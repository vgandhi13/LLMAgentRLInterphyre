"""Interphyre Viewer - Visualization tools for physics puzzles.

This module provides tools for visualizing levels, solutions, and running demos.

Usage:
    # As CLI
    python -m interphyre.viewer catapult --seed 42 --action 0.5 3.0 0.6

    # As module
    from interphyre.viewer import visualize_action
    visualize_action("catapult", 42, (0.5, 3.0, 0.6))
"""

# Re-export main viewer functionality from _viewer
# This allows: from interphyre.viewer import visualize_action
from interphyre.viewer._viewer import (
    visualize_action,
    visualize_solution_from_file,
    visualize_all_solutions,
    run_random_demo,
)

__all__ = [
    "visualize_action",
    "visualize_solution_from_file",
    "visualize_all_solutions",
    "run_random_demo",
]
