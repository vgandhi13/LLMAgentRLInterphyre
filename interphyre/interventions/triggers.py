"""
Trigger system for intervention scheduling.

This module defines triggers that determine when interventions should fire
during simulation. Supports time-based, event-based, and condition-based triggers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, FrozenSet, TYPE_CHECKING

if TYPE_CHECKING:
    from interphyre.engine import Box2DEngine


@dataclass
class Trigger(ABC):
    """
    Abstract base class for all intervention triggers.

    A trigger determines when an intervention should fire during simulation.
    Triggers are evaluated each simulation step.

    Attributes:
        priority: Lower values execute first (default: 0)
    """

    priority: int = 0

    @abstractmethod
    def should_fire(self, step_index: int, engine: "Box2DEngine") -> bool:
        """
        Check if this trigger should fire at the current step.

        Args:
            step_index: Current simulation step index
            engine: The Box2DEngine being simulated

        Returns:
            True if trigger should fire, False otherwise
        """
        pass

    def reset(self) -> None:
        """
        Reset trigger state (for once-only triggers that need to be reused).

        Override in subclasses if needed.
        """
        pass


@dataclass
class TimeBasedTrigger(Trigger):
    """
    Trigger that fires at a specific simulation step.

    This is the simplest trigger type - fires once at a predetermined step.

    Attributes:
        step_index: The step index at which to fire
        priority: Execution priority (default: 0)
    """

    step_index: int = 0
    priority: int = 0

    def should_fire(self, step_index: int, engine: "Box2DEngine") -> bool:
        """Fire when current step matches target step."""
        return step_index == self.step_index

    def __repr__(self) -> str:
        return f"TimeBasedTrigger(step={self.step_index}, priority={self.priority})"


@dataclass
class EventBasedTrigger(Trigger):
    """
    Trigger that fires when a simulation event occurs.

    Supports contact events, success conditions, and other state-based events.

    Attributes:
        event_type: Type of event ("contact", "success", etc.)
        object_names: Names of objects involved in the event
        once_only: If True, fire only once (default: True)
        priority: Execution priority (default: 0)
    """

    event_type: str = "contact"
    object_names: tuple[str, ...] = field(default_factory=tuple)
    once_only: bool = True
    priority: int = 0
    _fired: bool = field(default=False, init=False, repr=False)

    def should_fire(self, step_index: int, engine: "Box2DEngine") -> bool:
        """Fire when specified event occurs."""
        if self.once_only and self._fired:
            return False

        fired = self._check_event(engine)

        if fired and self.once_only:
            self._fired = True

        return fired

    def _check_event(self, engine: "Box2DEngine") -> bool:
        """Check if the event has occurred."""
        if self.event_type == "contact":
            if len(self.object_names) == 2:
                # Specific contact pair
                contact_pair = frozenset(self.object_names)
                return contact_pair in engine.contact_listener.contacts
            elif len(self.object_names) == 1:
                # Any contact involving this object
                obj_name = self.object_names[0]
                for contact in engine.contact_listener.contacts:
                    if obj_name in contact:
                        return True
                return False
            else:
                raise ValueError("EventBasedTrigger requires 1 or 2 object names")

        elif self.event_type == "success":
            # Success condition met
            if engine.level and engine.level.success_condition:
                return engine.level.success_condition(engine)
            return False

        else:
            raise ValueError(f"Unknown event type: {self.event_type}")

    def reset(self) -> None:
        """Reset fired state for reuse."""
        self._fired = False

    def __repr__(self) -> str:
        once_str = "once" if self.once_only else "repeat"
        return (
            f"EventBasedTrigger(type={self.event_type}, "
            f"objects={self.object_names}, {once_str}, priority={self.priority})"
        )


@dataclass
class ConditionBasedTrigger(Trigger):
    """
    Trigger that fires when a custom condition is met.

    This is the most flexible trigger type - accepts any callable that
    returns a boolean based on engine state.

    Attributes:
        condition: Callable that takes engine and returns bool
        once_only: If True, fire only once (default: True)
        priority: Execution priority (default: 0)
    """

    condition: Callable[["Box2DEngine"], bool] = field(default=lambda e: False)
    once_only: bool = True
    priority: int = 0
    _fired: bool = field(default=False, init=False, repr=False)

    def should_fire(self, step_index: int, engine: "Box2DEngine") -> bool:
        """Fire when custom condition returns True."""
        if self.once_only and self._fired:
            return False

        try:
            result = self.condition(engine)
        except Exception as e:
            # Log warning but don't crash simulation
            import logging

            logging.warning(f"Condition evaluation failed: {e}")
            return False

        if result and self.once_only:
            self._fired = True

        return result

    def reset(self) -> None:
        """Reset fired state for reuse."""
        self._fired = False

    def __repr__(self) -> str:
        once_str = "once" if self.once_only else "repeat"
        condition_name = getattr(self.condition, "__name__", "custom")
        return f"ConditionBasedTrigger(condition={condition_name}, {once_str}, priority={self.priority})"


# Convenience functions for creating triggers


def at_step(step_index: int, priority: int = 0) -> TimeBasedTrigger:
    """
    Create a time-based trigger that fires at a specific step.

    Args:
        step_index: Step index at which to fire
        priority: Execution priority (default: 0)

    Returns:
        TimeBasedTrigger configured for the specified step
    """
    return TimeBasedTrigger(step_index=step_index, priority=priority)


def on_contact(
    obj1: str, obj2: str, once_only: bool = True, priority: int = 0
) -> EventBasedTrigger:
    """
    Create an event-based trigger that fires when two objects contact.

    Args:
        obj1: First object name
        obj2: Second object name
        once_only: If True, fire only once (default: True)
        priority: Execution priority (default: 0)

    Returns:
        EventBasedTrigger configured for contact between obj1 and obj2
    """
    return EventBasedTrigger(
        event_type="contact",
        object_names=(obj1, obj2),
        once_only=once_only,
        priority=priority,
    )


def on_contact_with(obj: str, once_only: bool = True, priority: int = 0) -> EventBasedTrigger:
    """
    Create an event-based trigger that fires when an object contacts anything.

    Args:
        obj: Object name
        once_only: If True, fire only once (default: True)
        priority: Execution priority (default: 0)

    Returns:
        EventBasedTrigger configured for any contact involving obj
    """
    return EventBasedTrigger(
        event_type="contact",
        object_names=(obj,),
        once_only=once_only,
        priority=priority,
    )


def on_success(once_only: bool = True, priority: int = 0) -> EventBasedTrigger:
    """
    Create an event-based trigger that fires when success condition is met.

    Args:
        once_only: If True, fire only once (default: True)
        priority: Execution priority (default: 0)

    Returns:
        EventBasedTrigger configured for success condition
    """
    return EventBasedTrigger(
        event_type="success",
        object_names=(),
        once_only=once_only,
        priority=priority,
    )


def when(
    condition: Callable[["Box2DEngine"], bool],
    once_only: bool = True,
    priority: int = 0,
) -> ConditionBasedTrigger:
    """
    Create a condition-based trigger that fires when a custom condition is met.

    Args:
        condition: Callable that takes Box2DEngine and returns bool
        once_only: If True, fire only once (default: True)
        priority: Execution priority (default: 0)

    Returns:
        ConditionBasedTrigger configured with the custom condition

    Example:
        >>> trigger = when(lambda e: e.bodies["ball"].position.y < -2.0)
    """
    return ConditionBasedTrigger(
        condition=condition,
        once_only=once_only,
        priority=priority,
    )


def on_position_threshold(
    obj: str,
    axis: str,
    threshold: float,
    direction: str = "any",
    once_only: bool = True,
    priority: int = 0,
) -> ConditionBasedTrigger:
    """
    Create a trigger that fires when an object crosses a position threshold.

    Args:
        obj: Object name
        axis: Axis to check ('x' or 'y')
        threshold: Position threshold value
        direction: Direction to check ('above', 'below', or 'any')
        once_only: If True, fire only once (default: True)
        priority: Execution priority (default: 0)

    Returns:
        ConditionBasedTrigger configured for position threshold

    Example:
        >>> # Fire when ball falls below y=-2.0
        >>> trigger = on_position_threshold("ball", axis="y", threshold=-2.0, direction="below")
        >>> # Fire when ball crosses x=3.0 in any direction
        >>> trigger = on_position_threshold("ball", axis="x", threshold=3.0, direction="any")
    """
    if axis not in ("x", "y"):
        raise ValueError(f"axis must be 'x' or 'y', got {axis}")
    if direction not in ("above", "below", "any"):
        raise ValueError(f"direction must be 'above', 'below', or 'any', got {direction}")

    # Track whether we've crossed threshold (for 'any' direction)
    _state = {"crossed": False, "last_value": None}

    def _check_position(engine: "Box2DEngine") -> bool:
        if obj not in engine.bodies:
            return False

        body = engine.bodies[obj]
        current_value = body.position.x if axis == "x" else body.position.y

        if direction == "above":
            return current_value > threshold
        elif direction == "below":
            return current_value < threshold
        else:  # 'any' - fire when crossing threshold in either direction
            if _state["last_value"] is None:
                _state["last_value"] = current_value
                return False

            crossed = (_state["last_value"] <= threshold < current_value) or (
                _state["last_value"] >= threshold > current_value
            )
            _state["last_value"] = current_value
            return crossed

    return ConditionBasedTrigger(condition=_check_position, once_only=once_only, priority=priority)


def on_velocity_threshold(
    obj: str,
    speed_threshold: float,
    above: bool = True,
    once_only: bool = True,
    priority: int = 0,
) -> ConditionBasedTrigger:
    """
    Create a trigger that fires based on object velocity (speed magnitude).

    Args:
        obj: Object name
        speed_threshold: Speed threshold value
        above: If True, fire when speed exceeds threshold; if False, fire when below
        once_only: If True, fire only once (default: True)
        priority: Execution priority (default: 0)

    Returns:
        ConditionBasedTrigger configured for velocity threshold

    Example:
        >>> # Fire when ball speed exceeds 5.0
        >>> trigger = on_velocity_threshold("ball", speed_threshold=5.0, above=True)
        >>> # Fire when ball comes to rest (speed < 0.1)
        >>> trigger = on_velocity_threshold("ball", speed_threshold=0.1, above=False)
    """

    def _check_velocity(engine: "Box2DEngine") -> bool:
        if obj not in engine.bodies:
            return False

        body = engine.bodies[obj]
        velocity = body.linearVelocity
        speed = (velocity.x**2 + velocity.y**2) ** 0.5

        if above:
            return speed > speed_threshold
        else:
            return speed < speed_threshold

    return ConditionBasedTrigger(condition=_check_velocity, once_only=once_only, priority=priority)


@dataclass
class SequenceTrigger(Trigger):
    """
    Trigger that fires when a sequence of triggers fire in order.

    This is a stateful trigger that tracks which triggers in the sequence have
    already fired. Only fires when all triggers have fired in the specified order.

    Attributes:
        triggers: Tuple of triggers that must fire in sequence
        reset_on_failure: If True, reset sequence if wrong trigger fires
        once_only: If True, fire only once (default: True)
        priority: Execution priority (default: 0)
    """

    triggers: tuple[Trigger, ...] = field(default_factory=tuple)
    reset_on_failure: bool = True
    once_only: bool = True
    priority: int = 0
    _current_index: int = field(default=0, init=False, repr=False)
    _fired: bool = field(default=False, init=False, repr=False)

    def should_fire(self, step_index: int, engine: "Box2DEngine") -> bool:
        """Fire when all triggers in sequence have fired."""
        if self.once_only and self._fired:
            return False

        if not self.triggers:
            return False

        # Check if current trigger in sequence fires
        if self._current_index < len(self.triggers):
            current_trigger = self.triggers[self._current_index]
            if current_trigger.should_fire(step_index, engine):
                self._current_index += 1

                # If we've completed the sequence, fire
                if self._current_index == len(self.triggers):
                    if self.once_only:
                        self._fired = True
                    return True

            elif self.reset_on_failure:
                # Check if any earlier trigger in sequence fires (out of order)
                for i in range(self._current_index):
                    if self.triggers[i].should_fire(step_index, engine):
                        # Reset sequence
                        self._current_index = 0
                        break

        return False

    def reset(self) -> None:
        """Reset sequence state for reuse."""
        self._current_index = 0
        self._fired = False
        for trigger in self.triggers:
            trigger.reset()

    def __repr__(self) -> str:
        once_str = "once" if self.once_only else "repeat"
        reset_str = "reset_on_fail" if self.reset_on_failure else "no_reset"
        return (
            f"SequenceTrigger({len(self.triggers)} triggers, "
            f"{once_str}, {reset_str}, priority={self.priority})"
        )


@dataclass
class AnyTrigger(Trigger):
    """
    Trigger that fires when any of the provided triggers fire.

    This combinator trigger checks multiple triggers and fires if any of them
    report they should fire.

    Attributes:
        triggers: Tuple of triggers to check
        once_only: If True, fire only once (default: True)
        priority: Execution priority (default: 0)
    """

    triggers: tuple[Trigger, ...] = field(default_factory=tuple)
    once_only: bool = True
    priority: int = 0
    _fired: bool = field(default=False, init=False, repr=False)

    def should_fire(self, step_index: int, engine: "Box2DEngine") -> bool:
        """Fire if any trigger fires."""
        if self.once_only and self._fired:
            return False

        if not self.triggers:
            return False

        # Check if any trigger fires
        fired = any(trigger.should_fire(step_index, engine) for trigger in self.triggers)

        if fired and self.once_only:
            self._fired = True

        return fired

    def reset(self) -> None:
        """Reset fired state for reuse."""
        self._fired = False
        for trigger in self.triggers:
            trigger.reset()

    def __repr__(self) -> str:
        once_str = "once" if self.once_only else "repeat"
        return f"AnyTrigger({len(self.triggers)} triggers, {once_str}, priority={self.priority})"


def on_sequence(
    triggers: list[Trigger],
    reset_on_failure: bool = True,
    once_only: bool = True,
    priority: int = 0,
) -> SequenceTrigger:
    """
    Create a sequence trigger that fires when triggers fire in order.

    Args:
        triggers: List of triggers that must fire in sequence
        reset_on_failure: If True, reset sequence if wrong trigger fires (default: True)
        once_only: If True, fire only once (default: True)
        priority: Execution priority (default: 0)

    Returns:
        SequenceTrigger configured for the sequence

    Example:
        >>> # Fire when green-blue contact happens, then blue-red contact
        >>> trigger = on_sequence([
        ...     on_contact("green_ball", "blue_ball"),
        ...     on_contact("blue_ball", "red_ball")
        ... ])
    """
    return SequenceTrigger(
        triggers=tuple(triggers),
        reset_on_failure=reset_on_failure,
        once_only=once_only,
        priority=priority,
    )


def on_any(triggers: list[Trigger], once_only: bool = True, priority: int = 0) -> AnyTrigger:
    """
    Create a trigger that fires when any of the provided triggers fire.

    Args:
        triggers: List of triggers to combine
        once_only: If True, fire only once (default: True)
        priority: Execution priority (default: 0)

    Returns:
        AnyTrigger configured with the provided triggers

    Example:
        >>> # Fire when ball contacts either wall
        >>> trigger = on_any([
        ...     on_contact("ball", "left_wall"),
        ...     on_contact("ball", "right_wall")
        ... ])
    """
    return AnyTrigger(triggers=tuple(triggers), once_only=once_only, priority=priority)
