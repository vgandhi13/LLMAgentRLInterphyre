from interphyre.level import Level
from typing import Callable
import importlib

# Registry for level builders
_level_registry: dict[str, Callable[[int | None], Level]] = {}


# Decorator to build and register a level without instantiating it at import time
def register_level(func: Callable[[int | None], Level]):
    def wrapper(seed: int | None = None) -> Level:
        return func(seed)

    # Use the module name as the level name (matches filenames like "tipping_point")
    level_name = func.__module__.split(".")[-1]
    _level_registry[level_name] = wrapper

    return wrapper


def load_level(name: str, seed: int | None = None) -> Level:
    if name not in _level_registry:
        # Try to dynamically import it
        importlib.import_module(f"interphyre.levels.{name}")
        if name not in _level_registry:
            raise ValueError(f"Level '{name}' could not be registered.")
    return _level_registry[name](seed)


def list_levels() -> list[str]:
    """List all registered level names.

    Returns:
        List of level names sorted alphabetically

    Example:
        >>> from interphyre.levels import list_levels
        >>> levels = list_levels()
        >>> print(levels[:3])
        ['basket_case', 'catapult', 'dive_bomb']
    """
    return sorted(_level_registry.keys())


# Import all level modules to register them
from interphyre.levels import (
    basket_case,
    catapult,
    cliffhanger,
    dive_bomb,
    down_to_earth,
    end_of_line,
    falling_into_place,
    flagpole_sitta,
    just_a_nudge,
    keyhole,
    locust_swarm,
    marble_race,
    mind_the_gap,
    off_the_rails,
    pass_the_parcel,
    pinball_machine,
    seesaw,
    staircase,
    straight_face,
    the_cradle,
    the_funnel,
    tipping_point,
    two_body_problem,
    wedge_issue,
    zebra_gate,
)
