from Box2D import b2World, b2ContactListener, b2Contact, b2_pi
from typing import Any, Dict, List, Tuple, Optional, Union
import math

from interphyre.level import Level
from interphyre.objects import (
    Ball,
    Bar,
    Basket,
    PhyreObject,
    create_basket,
    create_ball,
    create_bar,
    create_walls,
)
from interphyre.config import (
    SimulationConfig,
    PerformanceProfiler,
    PRECISION,
    CONTACT_DISTANCE_TOLERANCE,
)


class GoalContactListener(b2ContactListener):
    """Contact listener for tracking object collisions and success conditions.

    This listener monitors all contact events in the physics world and tracks
    which objects are in contact with each other. It supports both performance-
    optimized tracking (relevant contacts only) and comprehensive research logging.

    Attributes:
        track_all_contacts (bool): Whether to track all contact events
        track_relevant_only (bool): Whether to only track relevant contact pairs
        profiler (Optional[PerformanceProfiler]): Performance profiler for timing
        relevant_pairs (set): Set of contact pairs to track for performance
        contacts (set): Currently active contact pairs
        contact_duration (dict): Duration of each contact pair
        contact_start_time (dict): Start time of each contact
        current_time (float): Current simulation time
        all_contacts_log (list): Complete log of all contact events
        contact_events (list): Detailed list of contact events
    """

    def __init__(
        self,
        track_all_contacts: bool = True,
        track_relevant_only: bool = False,
        profiler: Optional[PerformanceProfiler] = None,
        relevant_pairs: Optional[set] = None,
    ):
        """Initialize the contact listener.

        Args:
            track_all_contacts: Whether to track all contact events (default: True)
            track_relevant_only: Whether to only track relevant pairs for performance (default: False)
            profiler: Performance profiler for timing analysis (default: None)
            relevant_pairs: Set of contact pairs to track for performance (default: None)
        """
        super().__init__()
        self.track_all_contacts = track_all_contacts
        self.track_relevant_only = track_relevant_only
        self.profiler = profiler
        self.relevant_pairs = relevant_pairs or set()

        # Use tuples instead of frozensets for faster lookups
        self.contacts = set()
        self.contact_duration = {}
        self.contact_start_time = {}
        self.current_time = 0

        # Logging
        self.all_contacts_log = []
        self.contact_events = []

    def BeginContact(self, contact: b2Contact):
        a = contact.fixtureA.body.userData
        b = contact.fixtureB.body.userData
        if a and b:
            # Use frozenset for consistent contact pair representation
            contact_pair = frozenset((a, b))

            # Check if we should track this contact
            should_track = (
                self.track_all_contacts
                or not self.track_relevant_only
                or contact_pair in self.relevant_pairs
            )

            if should_track:
                self.contacts.add(contact_pair)
                self.contact_start_time[contact_pair] = self.current_time

            # Only log if profiling is enabled
            if self.track_all_contacts and self.profiler:
                self.contact_events.append(
                    {
                        "time": self.current_time,
                        "event": "begin",
                        "pair": contact_pair,
                        "objects": (a, b),
                    }
                )

    def EndContact(self, contact: b2Contact):
        a = contact.fixtureA.body.userData
        b = contact.fixtureB.body.userData
        if a and b:
            contact_pair = frozenset((a, b))

            # Check if we should track this contact
            should_track = (
                self.track_all_contacts
                or not self.track_relevant_only
                or contact_pair in self.relevant_pairs
            )

            if should_track:
                self.contacts.discard(contact_pair)
                # Reset the contact start time when contact ends
                if contact_pair in self.contact_start_time:
                    del self.contact_start_time[contact_pair]

            # Only log if profiling is enabled
            if self.track_all_contacts and self.profiler:
                self.contact_events.append(
                    {
                        "time": self.current_time,
                        "event": "end",
                        "pair": contact_pair,
                        "objects": (a, b),
                    }
                )

    def Update(self, dt):
        """Update the internal simulation time counter.

        Args:
            dt: Time delta in seconds to add to the current simulation time.
        """
        self.current_time += dt

    def GetContactDuration(self, a, b):
        """Get the current unbroken contact duration between two objects.

        Args:
            a: Name of the first object
            b: Name of the second object

        Returns:
            float: Duration in seconds that objects have been in continuous contact.
                  Returns 0 if objects are not currently in contact.
        """
        contact_pair = frozenset((a, b))
        if contact_pair in self.contacts and contact_pair in self.contact_start_time:
            return self.current_time - self.contact_start_time[contact_pair]
        return 0

    def IsInContactForDuration(self, a, b, required_duration):
        """Check if objects are currently in unbroken contact for at least the required duration.

        Args:
            a: Name of the first object
            b: Name of the second object
            required_duration: Minimum contact duration in seconds

        Returns:
            bool: True if objects are in contact and have been for at least required_duration seconds.
        """
        contact_pair = frozenset((a, b))

        # First check: Are they in the contact tracking set?
        if contact_pair not in self.contacts or contact_pair not in self.contact_start_time:
            return False

        # Check duration
        current_duration = self.current_time - self.contact_start_time[contact_pair]
        return current_duration >= required_duration

    def get_contact_log(self):
        """Get the full contact event log for research purposes."""
        return self.contact_events.copy()

    def get_contact_statistics(self):
        """Get statistics about all contacts for research purposes."""
        # Skip calculation if profiling disabled
        if not self.profiler or not self.contact_events:
            return {
                "total_events": 0,
                "unique_pairs": 0,
                "pair_counts": {},
                "current_contacts": len(self.contacts),
            }

        # Count contact events by object pairs
        pair_counts = {}
        for event in self.contact_events:
            pair = event["pair"]
            if pair not in pair_counts:
                pair_counts[pair] = {"begins": 0, "ends": 0, "invalidates": 0}
            event_type = event["event"]
            # Map event types to dictionary keys
            if event_type == "begin":
                pair_counts[pair]["begins"] += 1
            elif event_type == "end":
                pair_counts[pair]["ends"] += 1
            elif event_type == "invalidate":
                pair_counts[pair]["invalidates"] += 1

        return {
            "total_events": len(self.contact_events),
            "unique_pairs": len(pair_counts),
            "pair_counts": pair_counts,
            "current_contacts": len(self.contacts),
        }

    def ClearContacts(self):
        """Clear all contact tracking data and reset the simulation time.

        Removes all active contacts, contact start times, and resets the internal
        time counter to zero. This method is intended for full simulation resets
        only (e.g., when resetting the simulation or loading a new level). It should
        not be called mid-simulation as it will incorrectly reset the time counter,
        potentially breaking contact duration tracking.
        """
        self.contacts = set()
        self.contact_start_time = {}
        self.current_time = 0.0

    def invalidate_contact(self, contact_pair):
        """Invalidate a tracked contact pair when external validation determines it is invalid.

        Centralizes contact invalidation to avoid direct state mutation.
        """
        # Remove from active contacts
        self.contacts.discard(contact_pair)
        # Remove any recorded start time
        if contact_pair in self.contact_start_time:
            del self.contact_start_time[contact_pair]
        # Log contact invalidation event
        if self.track_all_contacts and self.profiler:
            self.contact_events.append(
                {
                    "time": self.current_time,
                    "event": "invalidate",
                    "pair": contact_pair,
                    "objects": tuple(contact_pair),
                }
            )


class Box2DEngine:
    """Main physics engine for the Interphyre simulation.

    This engine manages the Box2D physics world, object creation, contact tracking,
    and simulation stepping. It provides the core physics simulation functionality
    for the Interphyre environment.

    Attributes:
        config (SimulationConfig): Configuration parameters for the simulation
        profiler (PerformanceProfiler): Performance profiler for timing analysis
        world (b2World): The Box2D physics world
        contact_listener (GoalContactListener): Contact listener for collision tracking
        level (Optional[Level]): Current level being simulated
        bodies (Dict[str, b2Body]): Dictionary mapping object names to Box2D bodies
    """

    def __init__(self, level: Optional[Level] = None, config: Optional[SimulationConfig] = None):
        """Initialize the physics engine.

        Args:
            level: Initial level to load (default: None)
            config: Simulation configuration parameters (default: SimulationConfig())
        """
        self.config = config or SimulationConfig()
        self.profiler = PerformanceProfiler(self.config.enable_profiling)

        self.world = b2World(gravity=self.config.gravity, doSleep=self.config.do_sleep)
        self.world.warmStarting = self.config.warm_starting
        self.world.subStepping = self.config.substepping
        self.world.continuousPhysics = self.config.continuous_physics

        self.contact_listener = GoalContactListener(
            track_all_contacts=self.config.track_all_contacts,
            track_relevant_only=self.config.track_relevant_contacts_only,
            profiler=self.profiler,
        )
        self.world.contactListener = self.contact_listener

        # Velocity history for time-based stationary detection
        self._velocity_history = []

        self.reset(level)

    def reset(self, level: Optional[Level] = None):
        """Reset the engine with a new level.

        Clears the current physics world and loads a new level. This destroys
        all existing bodies and creates new ones based on the level definition.

        Args:
            level: New level to load (default: None, clears the world)
        """
        self.world.ClearForces()
        for body in self.world.bodies:
            self.world.DestroyBody(body)
        self.level = level
        self.contact_listener.ClearContacts()
        self.bodies = {}
        self._velocity_history = []  # Clear velocity history on reset
        if level is not None:
            self._create_world(level)
            # Update relevant contact pairs based on level
            self._update_relevant_contacts()

    def _create_world(self, level):
        # Create walls on the edges of the screen
        left_wall, right_wall, top_wall, bottom_wall = create_walls(self.world, 0.01, 10, 10)
        self.bodies["left_wall"] = left_wall
        self.bodies["right_wall"] = right_wall
        self.bodies["top_wall"] = top_wall
        self.bodies["bottom_wall"] = bottom_wall

        # Create objects in a deterministic order to ensure reproducibility
        for name in sorted(level.objects.keys()):
            obj = level.objects[name]
            # Skip placement of the action object
            if name in level.action_objects:
                continue
            if isinstance(obj, Ball):
                assert (
                    self.world is not None
                ), "World is not initialized. Call reset() before placing objects."
                body = create_ball(
                    self.world,
                    obj,
                    name,
                    use_ccd=self.config.continuous_collision_detection,
                )
            elif isinstance(obj, Bar):
                body = create_bar(
                    self.world,
                    obj,
                    name,
                    use_ccd=self.config.continuous_collision_detection,
                )
            elif isinstance(obj, Basket):
                body = create_basket(
                    self.world,
                    obj,
                    name,
                    use_ccd=self.config.continuous_collision_detection,
                )
            else:
                raise ValueError(f"Unknown object type for '{name}': {type(obj)}")
            self.bodies[name] = body

    def _update_relevant_contacts(self):
        """Update the list of relevant contact pairs based on the level's success condition."""
        if self.level is None:
            return

        # Only track contacts that are likely to be relevant for success conditions
        # This reduces memory usage and processing overhead
        relevant_pairs = set()

        # Track contacts between action objects and other objects
        for action_obj in self.level.action_objects:
            for obj_name in self.level.objects.keys():
                if obj_name != action_obj:
                    pair = frozenset((action_obj, obj_name))
                    relevant_pairs.add(pair)

        # Also track contacts between green objects and other objects (common success targets)
        for obj_name in self.level.objects.keys():
            if "green" in obj_name.lower():
                for other_obj in self.level.objects.keys():
                    if other_obj != obj_name:
                        pair = frozenset((obj_name, other_obj))
                        relevant_pairs.add(pair)

        self.contact_listener.relevant_pairs = relevant_pairs

    def place_action_objects(
        self,
        positions: List[Tuple[Union[int, float], Union[int, float], Union[int, float]]],
    ):
        """Place action objects at the start of the simulation.

        Args:
            positions: List of (x, y, size) tuples for each action object.
                For bars and baskets, size is ignored but must be provided.

        Note:
            All position and size values are rounded to the configured PRECISION
            (see interphyre.config.PRECISION) to ensure determinism.
        """
        if self.level is None:
            raise ValueError(
                "The level is not set. Please call reset() with a valid level before placing action objects."
            )
        assert (
            self.world is not None
        ), "World is not initialized. Call reset() before placing objects."

        for name, pos in zip(self.level.action_objects, positions):
            obj = self.level.objects[name]
            if isinstance(obj, Ball):
                x, y, size = pos
                obj.x = round(float(x), PRECISION)
                obj.y = round(float(y), PRECISION)
                obj.radius = round(float(size), PRECISION)
                body = create_ball(
                    self.world,
                    obj,
                    name,
                    use_ccd=self.config.continuous_collision_detection,
                )
            elif isinstance(obj, Bar):
                x, y, _ = pos
                obj.x = round(float(x), PRECISION)
                obj.y = round(float(y), PRECISION)
                body = create_bar(
                    self.world,
                    obj,
                    name,
                    use_ccd=self.config.continuous_collision_detection,
                )
            elif isinstance(obj, Basket):
                x, y, _ = pos
                obj.x = round(float(x), PRECISION)
                obj.y = round(float(y), PRECISION)
                body = create_basket(
                    self.world,
                    obj,
                    name,
                    use_ccd=self.config.continuous_collision_detection,
                )
            else:
                raise ValueError(f"Unknown object type for '{name}': {type(obj)}")
            self.bodies[name] = body

    def get_state(self) -> Dict[str, Any]:
        """
        Return the current simulation state.

        Returns:
            Dictionary containing the current physics state including:
            - object positions, velocities, angles, and angular velocities
            - contact information
            - world properties
        """
        if self.world is None or self.level is None:
            return {}

        state = {
            "objects": {},
            "contacts": {},
            "world_properties": {
                "gravity": self.world.gravity,
                "body_count": self.world.bodyCount,
                "contact_count": self.world.contactCount,
            },
        }

        # Get object states
        for name, obj in self.level.objects.items():
            if name in self.bodies:
                body = self.bodies[name]
                state["objects"][name] = {
                    "position": (body.position.x, body.position.y),
                    "velocity": (body.linearVelocity.x, body.linearVelocity.y),
                    "angle": body.angle,
                    "angular_velocity": body.angularVelocity,
                    "type": type(obj).__name__,
                    "dynamic": body.type == 2,  # b2_dynamicBody
                }
            else:
                # Object not yet placed (e.g., action objects)
                state["objects"][name] = {
                    "position": (obj.x, obj.y),
                    "velocity": (0.0, 0.0),
                    "angle": obj.angle,
                    "angular_velocity": 0.0,
                    "type": type(obj).__name__,
                    "dynamic": obj.dynamic,
                }

        # Get contact information
        for contact_pair in self.contact_listener.contacts:
            obj1, obj2 = contact_pair
            state["contacts"][f"{obj1}_{obj2}"] = {
                "objects": contact_pair,
                "duration": self.contact_listener.GetContactDuration(obj1, obj2),
            }

        return state

    def objects(self) -> Dict[str, PhyreObject]:
        if self.level is None:
            raise ValueError(
                "The level is not set. Please call reset() with a valid level before accessing objects."
            )
        return self.level.objects

    def has_contact(self, name1: str, name2: str) -> bool:
        """
        Check if the two object names have come into contact.
        """
        contact_pair = frozenset((name1, name2))
        return contact_pair in self.contact_listener.contacts

    def world_is_stationary(self) -> bool:
        """Check if the world is stationary using time-based averaging.

        Uses a sliding window of recent frames to determine if all objects have been
        stationary for a sustained period. This prevents false positives from momentary
        oscillations or floating-point jitter.

        Returns:
            bool: True if all objects have been below stationary_tolerance for the
                  last stationary_check_frames frames, False otherwise.

        Raises:
            ValueError: If world or level is not initialized.
        """
        if self.world is None:
            raise ValueError(
                "World is not initialized. Call reset() before checking for stationary bodies."
            )
        if self.level is None:
            raise ValueError(
                "Level is not set. Please call reset() before checking for stationary bodies."
            )

        # Check current frame's maximum velocity
        max_velocity = 0.0
        for body in self.world.bodies:
            if body.userData in self.level.objects:
                linear_vel = body.linearVelocity.length
                angular_vel = abs(body.angularVelocity)
                max_velocity = max(max_velocity, linear_vel, angular_vel)

        # Add current frame to history
        self._velocity_history.append(max_velocity)

        # Keep only the last N frames
        if len(self._velocity_history) > self.config.stationary_check_frames:
            self._velocity_history.pop(0)

        # Need full window before we can reliably say world is stationary
        if len(self._velocity_history) < self.config.stationary_check_frames:
            return False

        # World is stationary if ALL frames in window are below tolerance
        return all(vel <= self.config.stationary_tolerance for vel in self._velocity_history)

    def _is_point_inside_polygon(
        self, x: float, y: float, polygon: List[Tuple[float, float]]
    ) -> bool:
        n = len(polygon)
        inside = False
        p1x, p1y = polygon[0]
        for i in range(n + 1):
            p2x, p2y = polygon[i % n]
            if min(p1y, p2y) < y <= max(p1y, p2y) and x <= max(p1x, p2x):
                if p1y != p2y:
                    xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def is_in_basket(self, basket_name: str, target_name: str) -> bool:
        """
        Check if a ball is inside a basket using the sensor fixture.

        Args:
            basket_name: Name of the basket object
            target_name: Name of the ball object

        Returns:
            bool: True if the ball is inside the basket, False otherwise
        """
        if self.level is None or self.world is None:
            raise ValueError("Level or world not initialized.")
        if target_name not in self.level.objects or basket_name not in self.level.objects:
            return False
        basket = self.level.objects[basket_name]
        if not isinstance(basket, Basket):
            raise ValueError(f"{basket_name} is not a basket.")
        target = self.level.objects[target_name]
        if not isinstance(target, Ball):
            raise ValueError(
                f"{target_name} is a {type(target)}, is_in_basket_sensor currently only works with Balls."
            )

        # Get the basket and target bodies from the world
        basket_body = None
        target_body = None
        for body in self.world.bodies:
            if body.userData == basket_name:
                basket_body = body
            elif body.userData == target_name:
                target_body = body

        if basket_body is None or target_body is None:
            return False

        # Check if the target is in contact with the basket's sensor fixture
        for contact in self.world.contacts:
            # Check if this contact involves our basket and target
            if (contact.fixtureA.body == basket_body and contact.fixtureB.body == target_body) or (
                contact.fixtureA.body == target_body and contact.fixtureB.body == basket_body
            ):
                # Check if one of the fixtures is a sensor (our basket's interior)
                if contact.fixtureA.sensor or contact.fixtureB.sensor:
                    return True

        return False

    def _distance_ball_to_bar(self, ball_pos, bar_obj):
        """Calculate the distance from a ball's center to the closest point on a bar's surface.

        Args:
            ball_pos: Ball position (x, y) as a tuple or object with .x and .y attributes
            bar_obj: Bar object with x, y, angle, length, thickness attributes

        Returns:
            float: Distance from ball center to bar surface
        """
        # Transform ball center into bar's local coordinate system
        angle_rad = math.radians(-bar_obj.angle)  # negative for inverse rotation
        dx = ball_pos.x - bar_obj.x
        dy = ball_pos.y - bar_obj.y
        local_x = dx * math.cos(angle_rad) - dy * math.sin(angle_rad)
        local_y = dx * math.sin(angle_rad) + dy * math.cos(angle_rad)

        half_length = bar_obj.length / 2
        half_thickness = bar_obj.thickness / 2

        # Clamp local_x and local_y to the rectangle bounds
        closest_x = max(-half_length, min(half_length, local_x))
        closest_y = max(-half_thickness, min(half_thickness, local_y))

        # Compute distance from ball center to closest point on rectangle
        dist = math.sqrt((local_x - closest_x) ** 2 + (local_y - closest_y) ** 2)
        return dist

    def _distance_bar_to_bar(self, bar_a_body, bar_a_obj, bar_b_body, bar_b_obj):
        """Calculate minimum edge-to-edge distance between two bars.

        Uses vertex-to-polygon distance for convex rectangles. Checks all vertices
        of each bar against the edges of the other bar.

        Args:
            bar_a_body: Box2D body for bar A (current position and angle)
            bar_a_obj: Bar object A (dimensions)
            bar_b_body: Box2D body for bar B (current position and angle)
            bar_b_obj: Bar object B (dimensions)

        Returns:
            float: Minimum distance between bar surfaces
        """
        # Get all corners for both bars using current body positions and angles
        corners_a = self._get_bar_corners(bar_a_body, bar_a_obj)
        corners_b = self._get_bar_corners(bar_b_body, bar_b_obj)

        # For two convex polygons, minimum distance is either:
        # 1. Distance from a vertex of A to an edge of B, or
        # 2. Distance from a vertex of B to an edge of A
        min_dist = float("inf")

        # Check all vertices of A against B's edges
        for corner_a in corners_a:
            dist = self._distance_point_to_polygon(corner_a, corners_b)
            min_dist = min(min_dist, dist)

        # Check all vertices of B against A's edges
        for corner_b in corners_b:
            dist = self._distance_point_to_polygon(corner_b, corners_a)
            min_dist = min(min_dist, dist)

        return min_dist

    def _distance_ball_to_basket(self, ball_pos, basket_body, basket_obj):
        """Calculate minimum distance from a ball center to basket fixtures.

        Computes distance to the floor and wall polygons that make up the basket.
        The ball position is transformed into the basket's local frame so the
        basket geometry can be evaluated in local coordinates.
        """
        # Transform ball center into basket's local coordinate system
        angle_rad = -basket_body.angle
        dx = ball_pos.x - basket_body.position.x
        dy = ball_pos.y - basket_body.position.y
        local_x = dx * math.cos(angle_rad) - dy * math.sin(angle_rad)
        local_y = dx * math.sin(angle_rad) + dy * math.cos(angle_rad)
        point = (local_x, local_y)

        bw = basket_obj.bottom_width
        tw = basket_obj.top_width
        h = basket_obj.height
        wt = basket_obj.wall_thickness
        ft = basket_obj.floor_thickness
        anchor_offset_x, anchor_offset_y = basket_obj.get_anchor_offset()

        polygons = []

        # Floor rectangle
        floor_half_width = (bw + 2 * wt) / 2
        floor_half_height = ft / 2
        floor_center_x = anchor_offset_x
        floor_center_y = anchor_offset_y + ft / 2
        polygons.append(
            [
                (floor_center_x - floor_half_width, floor_center_y - floor_half_height),
                (floor_center_x + floor_half_width, floor_center_y - floor_half_height),
                (floor_center_x + floor_half_width, floor_center_y + floor_half_height),
                (floor_center_x - floor_half_width, floor_center_y + floor_half_height),
            ]
        )

        # Left wall trapezoid
        polygons.append(
            [
                (-bw / 2 - wt + anchor_offset_x, ft + anchor_offset_y),
                (-tw / 2 - wt + anchor_offset_x, ft + h + anchor_offset_y),
                (-tw / 2 + anchor_offset_x, ft + h + anchor_offset_y),
                (-bw / 2 + anchor_offset_x, ft + anchor_offset_y),
            ]
        )

        # Right wall trapezoid
        polygons.append(
            [
                (bw / 2 + wt + anchor_offset_x, ft + anchor_offset_y),
                (bw / 2 + anchor_offset_x, ft + anchor_offset_y),
                (tw / 2 + anchor_offset_x, ft + h + anchor_offset_y),
                (tw / 2 + wt + anchor_offset_x, ft + h + anchor_offset_y),
            ]
        )

        # Optional inner walls for anti-tunneling
        if basket_obj.double_walls:
            inner_gap = 0.03
            polygons.append(
                [
                    (-bw / 2 + inner_gap + anchor_offset_x, ft + anchor_offset_y),
                    (-tw / 2 + inner_gap + anchor_offset_x, ft + h + anchor_offset_y),
                    (-tw / 2 + inner_gap + wt / 2 + anchor_offset_x, ft + h + anchor_offset_y),
                    (-bw / 2 + inner_gap + wt / 2 + anchor_offset_x, ft + anchor_offset_y),
                ]
            )
            polygons.append(
                [
                    (bw / 2 - inner_gap + anchor_offset_x, ft + anchor_offset_y),
                    (bw / 2 - inner_gap - wt / 2 + anchor_offset_x, ft + anchor_offset_y),
                    (tw / 2 - inner_gap - wt / 2 + anchor_offset_x, ft + h + anchor_offset_y),
                    (tw / 2 - inner_gap + anchor_offset_x, ft + h + anchor_offset_y),
                ]
            )

        min_dist = float("inf")
        for poly in polygons:
            dist = self._distance_point_to_polygon(point, poly)
            if dist < min_dist:
                min_dist = dist

        return min_dist

    def _distance_point_to_polygon(self, point, polygon_corners):
        """Calculate minimum distance from a point to a polygon defined by corners.

        Args:
            point: (x, y) tuple
            polygon_corners: List of (x, y) tuples defining polygon vertices in order

        Returns:
            float: Minimum distance from point to polygon (0 if point is inside)
        """
        if self._point_in_convex_polygon(point, polygon_corners):
            return 0.0
        min_dist = float("inf")
        n = len(polygon_corners)

        # Check distance to each edge
        for i in range(n):
            p1 = polygon_corners[i]
            p2 = polygon_corners[(i + 1) % n]
            dist = self._distance_point_to_segment(point, p1, p2)
            min_dist = min(min_dist, dist)

        return min_dist

    def _point_in_convex_polygon(self, point, polygon_corners):
        """Check if a point lies inside a convex polygon."""
        px, py = point
        sign = None
        n = len(polygon_corners)
        for i in range(n):
            x1, y1 = polygon_corners[i]
            x2, y2 = polygon_corners[(i + 1) % n]
            cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
            if cross == 0:
                continue
            current_sign = cross > 0
            if sign is None:
                sign = current_sign
            elif sign != current_sign:
                return False
        return True
    def _distance_point_to_segment(self, point, seg_start, seg_end):
        """Calculate minimum distance from a point to a line segment.

        Args:
            point: (x, y) tuple
            seg_start: (x, y) tuple for segment start
            seg_end: (x, y) tuple for segment end

        Returns:
            float: Minimum distance from point to segment
        """
        px, py = point
        x1, y1 = seg_start
        x2, y2 = seg_end

        # Vector from start to end
        dx = x2 - x1
        dy = y2 - y1

        # If segment is a point
        if dx == 0 and dy == 0:
            return math.sqrt((px - x1) ** 2 + (py - y1) ** 2)

        # Parameter t for closest point on line
        t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))

        # Closest point on segment
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy

        # Distance to closest point
        return math.sqrt((px - closest_x) ** 2 + (py - closest_y) ** 2)

    def _get_bar_corners(self, body, bar_obj):
        """Get the four corner points of a bar in world coordinates.

        Args:
            body: Box2D body with current position and angle
            bar_obj: Bar object with length and thickness attributes

        Returns:
            list: Four (x, y) tuples representing the corners
        """
        angle_rad = body.angle  # Use body's current angle (already in radians)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        half_length = bar_obj.length / 2
        half_thickness = bar_obj.thickness / 2

        # Four corners in local coordinates (bar frame)
        local_corners = [
            (-half_length, -half_thickness),
            (half_length, -half_thickness),
            (half_length, half_thickness),
            (-half_length, half_thickness),
        ]

        # Transform to world coordinates
        corners = []
        for lx, ly in local_corners:
            wx = body.position.x + lx * cos_a - ly * sin_a
            wy = body.position.y + lx * sin_a + ly * cos_a
            corners.append((wx, wy))

        return corners

    def _validate_contact_distances(self):
        """Validate all tracked contacts by checking physical distances.

        This method is called once per simulation step to ensure contacts reported
        by Box2D correspond to objects that are actually close enough to be touching.
        Contacts that fail distance validation are invalidated.

        This prevents the race condition where contacts are invalidated mid-success-check,
        which can cause non-deterministic behavior. By validating all contacts once per
        step (before success checking), we ensure consistent state.
        """
        if not self.config.validate_contact_distance or self.level is None:
            return

        # Make a copy to avoid modifying set during iteration
        contacts_to_validate = list(self.contact_listener.contacts)

        for contact_pair in contacts_to_validate:
            a, b = contact_pair

            # Skip if objects don't exist
            if a not in self.level.objects or b not in self.level.objects:
                continue

            body_a = self.bodies.get(a)
            body_b = self.bodies.get(b)
            if body_a is None or body_b is None:
                continue

            # Get object sizes to determine contact threshold
            obj_a = self.level.objects[a]
            obj_b = self.level.objects[b]

            # Calculate actual distance and contact threshold based on object types
            distance = None
            contact_threshold = None

            if isinstance(obj_a, Ball) and isinstance(obj_b, Ball):
                # Ball-ball contact: distance is center-to-center
                pos_a = body_a.position
                pos_b = body_b.position
                distance = ((pos_a.x - pos_b.x) ** 2 + (pos_a.y - pos_b.y) ** 2) ** 0.5
                contact_threshold = obj_a.radius + obj_b.radius + CONTACT_DISTANCE_TOLERANCE
            elif isinstance(obj_a, Ball) and isinstance(obj_b, Basket):
                # Ball-basket contact: distance to basket fixtures
                distance = self._distance_ball_to_basket(body_a.position, body_b, obj_b)
                contact_threshold = obj_a.radius + CONTACT_DISTANCE_TOLERANCE
            elif isinstance(obj_a, Basket) and isinstance(obj_b, Ball):
                # Basket-ball contact: symmetric
                distance = self._distance_ball_to_basket(body_b.position, body_a, obj_a)
                contact_threshold = obj_b.radius + CONTACT_DISTANCE_TOLERANCE
            elif isinstance(obj_a, Ball) and isinstance(obj_b, Bar):
                # Ball-bar contact: calculate distance from ball center to bar surface
                distance = self._distance_ball_to_bar(body_a.position, obj_b)
                contact_threshold = obj_a.radius + CONTACT_DISTANCE_TOLERANCE
            elif isinstance(obj_a, Bar) and isinstance(obj_b, Ball):
                # Bar-ball contact: same as ball-bar (symmetric)
                distance = self._distance_ball_to_bar(body_b.position, obj_a)
                contact_threshold = obj_b.radius + CONTACT_DISTANCE_TOLERANCE
            elif isinstance(obj_a, Bar) and isinstance(obj_b, Bar):
                # Bar-bar contact: edge-to-edge distance with larger tolerance
                distance = self._distance_bar_to_bar(body_a, obj_a, body_b, obj_b)
                contact_threshold = 0.1
            else:
                # For other object combinations (basket, etc.), use center-to-center
                # with a conservative threshold
                pos_a = body_a.position
                pos_b = body_b.position
                distance = ((pos_a.x - pos_b.x) ** 2 + (pos_a.y - pos_b.y) ** 2) ** 0.5
                contact_threshold = 0.5  # Conservative threshold for basket and other objects

            # If objects are too far apart, invalidate the contact
            if distance is not None and distance > contact_threshold:
                self.contact_listener.invalidate_contact(contact_pair)

    def is_in_contact_for_duration(self, a, b, success_time: Optional[float] = None):
        """Check if objects are currently in unbroken contact for the required duration.

        This is a READ-ONLY method that does not modify contact state. Contact validation
        happens separately in _validate_contact_distances() to prevent race conditions.

        Args:
            a: Name of the first object
            b: Name of the second object
            success_time: Required contact duration in seconds. If None, uses config.default_success_time.

        Returns:
            bool: True if objects are in contact and have been for at least success_time seconds.

        Raises:
            ValueError: If level is not set or objects are not in the level.
        """
        if self.level is None:
            raise ValueError(
                "Level is not set. Please call reset() with a valid level before checking for contact duration."
            )
        if a not in self.level.objects or b not in self.level.objects:
            return False
        if success_time is None:
            success_time = self.config.default_success_time

        # Simply check if duration requirement is met (validation happens separately)
        return self.contact_listener.IsInContactForDuration(a, b, success_time)

    def time_update(self, dt):
        """Update the contact listener's internal time tracking.

        Args:
            dt: Time delta in seconds to add to the current simulation time.
        """
        self.contact_listener.Update(dt)

    def get_contact_duration(self, a, b):
        """Get the current unbroken contact duration between two objects.

        Args:
            a: Name of the first object
            b: Name of the second object

        Returns:
            float: Duration in seconds that objects have been in continuous contact.
                  Returns 0 if objects are not currently in contact.

        Raises:
            ValueError: If level is not set or objects are not in the level.
        """
        if self.level is None:
            raise ValueError(
                "Level is not set. Please call reset() with a valid level before checking for contact duration."
            )
        if a not in self.level.objects or b not in self.level.objects:
            return 0
        return self.contact_listener.GetContactDuration(a, b)

    def get_contact_log(self):
        """Get the full contact event log for research purposes."""
        return self.contact_listener.get_contact_log()

    def get_contact_statistics(self):
        """Get statistics about all contacts for research purposes."""
        return self.contact_listener.get_contact_statistics()
