from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import gymnasium as gym
import numpy as np

from interphyre.config import PRECISION, SimulationConfig
from interphyre.engine import Box2DEngine
from interphyre.level import Level
from interphyre.render import Renderer

if TYPE_CHECKING:
    from interphyre.interventions.state import StateSnapshot
    from interphyre.interventions.triggers import Trigger
    from interphyre.objects import PhyreObject


class InterventionContext:
    """Context manager for scoped interventions.

    Provides batched modifications and optional auto-rollback on exception.
    Use for level-structural changes or when you need transactional semantics.

    Example:
        with env.intervention_context() as ctx:
            ctx.add_object("ball", Ball(x=0, y=0, radius=0.5))
            ctx.apply_impulse("ball", impulse=(5.0, 0.0))
            ctx.modify_success_condition(lambda engine: custom_check(engine))
    """

    def __init__(self, env: "InterphyreEnv", auto_rollback: bool = False):
        """Initialize intervention context.

        Args:
            env: The InterphyreEnv instance to operate on
            auto_rollback: If True, automatically restore state on exception
        """
        self._env = env
        self._auto_rollback = auto_rollback
        self._snapshot: Optional[StateSnapshot] = None

    def __enter__(self) -> "InterventionContext":
        if self._auto_rollback:
            from interphyre.interventions.state import StateSnapshot

            self._snapshot = StateSnapshot.capture(self._env.engine)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_type is not None and self._auto_rollback and self._snapshot:
            self._snapshot.restore(self._env.engine)
            return True  # Suppress the exception
        return False

    # === Object Management ===

    def add_object(
        self,
        name: str,
        obj: "PhyreObject",
        impulse: Optional[Tuple[float, float]] = None,
    ) -> None:
        """Add a new object to the simulation."""
        self._env.add_object(name, obj, impulse=impulse)

    def remove_object(self, name: str) -> None:
        """Remove an object from the simulation."""
        self._env.remove_object(name)

    def apply_impulse(
        self,
        name: str,
        impulse: Tuple[float, float],
        point: Optional[Tuple[float, float]] = None,
    ) -> None:
        """Apply an impulse to an object."""
        self._env.apply_impulse(name, impulse, point=point)

    def apply_force(
        self,
        name: str,
        force: Tuple[float, float],
        point: Optional[Tuple[float, float]] = None,
    ) -> None:
        """Apply a force to an object."""
        self._env.apply_force(name, force, point=point)

    def set_velocity(
        self,
        name: str,
        vx: Optional[float] = None,
        vy: Optional[float] = None,
    ) -> None:
        """Set object linear velocity."""
        self._env.set_velocity(name, vx=vx, vy=vy)

    def set_position(
        self,
        name: str,
        x: Optional[float] = None,
        y: Optional[float] = None,
    ) -> None:
        """Set object position."""
        self._env.set_position(name, x=x, y=y)

    def freeze(self, name: str) -> None:
        """Freeze object by zeroing all velocities."""
        self._env.freeze(name)

    # === Level-Structural Changes (only available in context) ===

    def modify_success_condition(
        self, condition: Callable[[Box2DEngine], bool]
    ) -> None:
        """Modify the level's success condition.

        Args:
            condition: New success condition function that takes engine and returns bool
        """
        self._env._level.success_condition = condition

    def modify_metadata(self, **kwargs) -> None:
        """Modify the level's metadata.

        Args:
            **kwargs: Key-value pairs to update in metadata
        """
        if self._env._level.metadata is None:
            self._env._level.metadata = {}
        self._env._level.metadata.update(kwargs)


class InterphyreEnv(gym.Env):
    """Gymnasium environment for physics-based puzzles.

    This environment simulates physics puzzles where agents place objects to achieve
    specific goals. The environment follows a one-shot paradigm: agents provide an
    action (object placement), then the full simulation runs to completion.

    Example (standard RL usage):
        env = InterphyreEnv("catapult", seed=42, render_mode="human")
        obs, info = env.reset()
        obs, reward, term, trunc, info = env.step([(0.5, 3.0, 0.6)])

    Example (intervention/replanning):
        env = InterphyreEnv("catapult", seed=42, enable_interventions=True)
        env.place_action((0.5, 3.0, 0.6))
        snapshot, step = env.run_until(on_contact("ball", "platform"))
        if snapshot:
            env.restore(snapshot)
            env.add_object("ball2", Ball(x=0, y=2, radius=0.3))
            obs, reward, term, trunc, info = env.step_until(on_success())
    """

    metadata = {
        "render_modes": ["human", "rgb_array", "single_rgb_array"],
        "render_fps": 30,
        "name": "InterphyreEnv",
    }

    def __init__(
        self,
        level_name: str,
        seed: Optional[int] = None,
        config: Optional[SimulationConfig] = None,
        render_mode: Optional[str] = None,
        observation_type: str = "physics_state",
        action_type: str = "continuous",
        image_size: Tuple[int, int] = (600, 600),
        image_ppm: float = 60.0,
        discrete_colors: bool = False,
        enable_interventions: bool = False,
    ):
        """Initialize the Phyre environment.

        Args:
            level_name: Name of the level to load from the registry
            seed: Random seed for level variation (optional)
            config: Optional simulation configuration (uses defaults if None)
            render_mode: Rendering mode - "human" for pygame, "rgb_array" for images, None for no rendering
            observation_type: Type of observation space ("physics_state", "image", "both")
            action_type: Type of action space ("continuous", "discrete")
            image_size: Size of rendered images (width, height) for image observations
            image_ppm: Pixels per Box2D unit for image rendering
            discrete_colors: If True, use single-channel discrete colors instead of RGB
            enable_interventions: If True, enable intervention scheduling in the engine
        """
        super().__init__()

        # Load level from registry
        from interphyre.levels import load_level

        self._level = load_level(level_name, seed=seed)
        self._level_name = level_name
        self._seed = seed

        # Set up config with intervention flag
        self.config = config or SimulationConfig()
        if enable_interventions:
            self.config = SimulationConfig(
                **{
                    **self.config.__dict__,
                    "enable_interventions": True,
                }
            )

        # Set up renderer based on render_mode
        self.render_mode = render_mode
        self.renderer: Optional[Renderer] = None
        if render_mode == "human":
            from interphyre.render.pygame import PygameRenderer

            self.renderer = PygameRenderer(width=600, height=600, ppm=60)

        self.observation_type = observation_type
        self.action_type = action_type
        self.image_size = image_size
        self.image_ppm = image_ppm
        self.discrete_colors = discrete_colors

        # Initialize engine
        self.engine = Box2DEngine(config=self.config)
        self.action_placed = False
        self.current_obs = None
        self.current_state = None
        self.step_count = 0
        self.max_steps = self.config.max_steps
        self._rollout_complete = False
        self._active_interventions: List[Any] = []

        # Set up action space
        self._setup_action_space()

        # Set up observation space
        self._setup_observation_space()

        # Initialize numpy random generator
        self.np_random = np.random.default_rng()

        # Initialize state
        self.reset()

    # === Factory Methods ===

    @classmethod
    def make(
        cls,
        level_name: str,
        seed: Optional[int] = None,
        **kwargs,
    ) -> "InterphyreEnv":
        """Create a InterphyreEnv from a level name.

        This is an alias for the constructor for familiarity with gym.make().

        Args:
            level_name: Name of the level to load
            seed: Random seed for level variation
            **kwargs: Additional arguments passed to InterphyreEnv constructor

        Returns:
            InterphyreEnv instance
        """
        return cls(level_name, seed=seed, **kwargs)

    @classmethod
    def from_level(
        cls,
        level: Level,
        config: Optional[SimulationConfig] = None,
        render_mode: Optional[str] = None,
        **kwargs,
    ) -> "InterphyreEnv":
        """Create a InterphyreEnv from a custom Level object.

        Use this when you have a custom level that isn't in the registry.

        Args:
            level: Custom Level object
            config: Optional simulation configuration
            render_mode: Rendering mode
            **kwargs: Additional arguments

        Returns:
            InterphyreEnv instance
        """
        # Create instance without calling __init__ normally
        instance = object.__new__(cls)
        gym.Env.__init__(instance)

        # Set up the level directly
        instance._level = level
        instance._level_name = level.name
        instance._seed = None

        # Set up config
        enable_interventions = kwargs.pop("enable_interventions", False)
        instance.config = config or SimulationConfig()
        if enable_interventions:
            instance.config = SimulationConfig(
                **{
                    **instance.config.__dict__,
                    "enable_interventions": True,
                }
            )

        # Set up renderer
        instance.render_mode = render_mode
        instance.renderer = None
        if render_mode == "human":
            from interphyre.render.pygame import PygameRenderer

            instance.renderer = PygameRenderer(width=600, height=600, ppm=60)

        instance.observation_type = kwargs.get("observation_type", "physics_state")
        instance.action_type = kwargs.get("action_type", "continuous")
        instance.image_size = kwargs.get("image_size", (600, 600))
        instance.image_ppm = kwargs.get("image_ppm", 60.0)
        instance.discrete_colors = kwargs.get("discrete_colors", False)

        # Initialize engine
        instance.engine = Box2DEngine(config=instance.config)
        instance.action_placed = False
        instance.current_obs = None
        instance.current_state = None
        instance.step_count = 0
        instance.max_steps = instance.config.max_steps
        instance._rollout_complete = False
        instance._active_interventions = []

        # Set up spaces
        instance._setup_action_space()
        instance._setup_observation_space()

        # Initialize numpy random generator
        instance.np_random = np.random.default_rng()

        # Initialize state
        instance.reset()

        return instance

    # === Properties ===

    @property
    def level(self) -> Level:
        """Get the current level (read-only)."""
        return self._level

    @property
    def objects(self) -> Dict[str, Any]:
        """Get the level's objects dictionary (read-only view)."""
        return self._level.objects

    @property
    def success(self) -> bool:
        """Check if the current state satisfies the success condition."""
        return self._level.success_condition(self.engine)

    # === Intervention API ===

    def run_until(
        self,
        trigger: "Trigger",
        action: Optional[
            Union[Tuple[float, float, float], List[Tuple[float, float, float]]]
        ] = None,
        max_steps: int = 240,
    ) -> Tuple[Optional["StateSnapshot"], int]:
        """Run simulation until trigger fires.

        Args:
            trigger: Trigger condition to wait for
            action: Optional action to place before running. Can be:
                - Single (x, y, radius) tuple for one action object
                - List of tuples for multiple action objects
                - None if action already placed or no action objects
            max_steps: Maximum steps to simulate

        Returns:
            (snapshot, step_index) if triggered, (None, final_step) if timeout

        Example:
            snapshot, step = env.run_until(
                on_contact("ball", "platform"),
                action=(0.5, 3.0, 0.6),
                max_steps=500
            )
        """
        from interphyre.interventions.state import StateSnapshot

        # Place action if provided and not already placed
        if action is not None and not self.action_placed:
            if isinstance(action, tuple) and len(action) == 3:
                action = [action]

            validation_result = self._validate_action_with_failure(action)
            if validation_result["invalid"]:
                raise ValueError(f"Invalid action: {validation_result['error']}")

            self._place_action_objects(validation_result["action"])
            self.action_placed = True

        start = self.step_count

        for step_index in range(start, start + max_steps):
            self._step_physics()
            self.render()

            if trigger.should_fire(step_index + 1, self.engine):
                snapshot = StateSnapshot.capture(
                    self.engine,
                    metadata={"step_index": step_index + 1, "trigger": str(trigger)},
                )
                return snapshot, step_index + 1

        return None, start + max_steps

    def restore(self, snapshot: "StateSnapshot") -> None:
        """Restore simulation to a previous state.

        Args:
            snapshot: StateSnapshot to restore
        """
        snapshot.restore(self.engine)
        if snapshot.metadata and "step_index" in snapshot.metadata:
            self.step_count = snapshot.metadata["step_index"]
        self._rollout_complete = False

    def step_until(
        self,
        trigger: "Trigger",
        max_steps: int = 240,
    ) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """Continue simulation until trigger fires, returning Gym-style output.

        This is the intervention-aware equivalent of step() for continuing
        after a restore or intervention.

        Args:
            trigger: Trigger condition to wait for (e.g., on_success())
            max_steps: Maximum steps to simulate

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        snapshot, final_step = self.run_until(trigger, action=None, max_steps=max_steps)

        success = self._level.success_condition(self.engine)
        truncated = snapshot is None and not success

        obs = self._get_observation()
        reward = self._calculate_reward(success, truncated)
        info = self._get_info_dict(success, success, truncated)
        info["final_step"] = final_step

        return obs, reward, success, truncated, info

    def intervention_context(self, auto_rollback: bool = False) -> InterventionContext:
        """Create an intervention context for scoped modifications.

        Args:
            auto_rollback: If True, automatically restore state if exception occurs

        Returns:
            InterventionContext for use in a with statement
        """
        return InterventionContext(self, auto_rollback=auto_rollback)

    # === Object Management API ===

    def add_object(
        self,
        name: str,
        obj: "PhyreObject",
        impulse: Optional[Tuple[float, float]] = None,
    ) -> None:
        """Add a new object to the simulation.

        Args:
            name: Unique name for the object
            obj: PhyreObject instance (Ball, Bar, or Basket)
            impulse: Optional initial impulse (ix, iy)

        Raises:
            ValueError: If name already exists
        """
        if name in self.engine.bodies:
            raise ValueError(f"Object '{name}' already exists")

        from interphyre.objects import (
            Ball,
            Bar,
            Basket,
            create_ball,
            create_bar,
            create_basket,
        )

        # Add to level objects
        self._level.objects[name] = obj

        # Create physics body
        if isinstance(obj, Ball):
            body = create_ball(
                self.engine.world,
                obj,
                name,
                use_ccd=self.config.continuous_collision_detection,
            )
        elif isinstance(obj, Bar):
            body = create_bar(
                self.engine.world,
                obj,
                name,
                use_ccd=self.config.continuous_collision_detection,
            )
        elif isinstance(obj, Basket):
            body = create_basket(
                self.engine.world,
                obj,
                name,
                use_ccd=self.config.continuous_collision_detection,
            )
        else:
            raise TypeError(f"Unknown object type: {type(obj)}")

        self.engine.bodies[name] = body

        # Apply initial impulse if provided
        if impulse is not None:
            self.apply_impulse(name, impulse)

    def remove_object(self, name: str) -> None:
        """Remove an object from the simulation.

        Args:
            name: Name of object to remove

        Raises:
            ValueError: If object doesn't exist
        """
        if name not in self.engine.bodies:
            raise ValueError(f"Object '{name}' not found")

        # Destroy physics body
        self.engine.world.DestroyBody(self.engine.bodies[name])
        del self.engine.bodies[name]

        # Remove from level
        if name in self._level.objects:
            del self._level.objects[name]

    def apply_impulse(
        self,
        name: str,
        impulse: Tuple[float, float],
        point: Optional[Tuple[float, float]] = None,
    ) -> None:
        """Apply an impulse to an object.

        Args:
            name: Object name
            impulse: (ix, iy) impulse vector
            point: Application point (default: center of mass)
        """
        body = self._get_body(name)
        from Box2D import b2Vec2

        ix, iy = impulse
        if point is None:
            point_vec = body.worldCenter
        else:
            point_vec = b2Vec2(point[0], point[1])

        body.ApplyLinearImpulse(b2Vec2(ix, iy), point_vec, True)

    def apply_force(
        self,
        name: str,
        force: Tuple[float, float],
        point: Optional[Tuple[float, float]] = None,
    ) -> None:
        """Apply a force to an object.

        Args:
            name: Object name
            force: (fx, fy) force vector
            point: Application point (default: center of mass)
        """
        body = self._get_body(name)
        from Box2D import b2Vec2

        fx, fy = force
        if point is None:
            point_vec = body.worldCenter
        else:
            point_vec = b2Vec2(point[0], point[1])

        body.ApplyForce(b2Vec2(fx, fy), point_vec, True)

    def set_velocity(
        self,
        name: str,
        vx: Optional[float] = None,
        vy: Optional[float] = None,
    ) -> None:
        """Set object linear velocity.

        Args:
            name: Object name
            vx: X velocity (None to keep current)
            vy: Y velocity (None to keep current)
        """
        body = self._get_body(name)
        from Box2D import b2Vec2

        current = body.linearVelocity
        new_vx = vx if vx is not None else current.x
        new_vy = vy if vy is not None else current.y
        body.linearVelocity = b2Vec2(new_vx, new_vy)

    def set_position(
        self,
        name: str,
        x: Optional[float] = None,
        y: Optional[float] = None,
    ) -> None:
        """Set object position.

        Args:
            name: Object name
            x: X position (None to keep current)
            y: Y position (None to keep current)
        """
        body = self._get_body(name)
        from Box2D import b2Vec2

        current = body.position
        new_x = x if x is not None else current.x
        new_y = y if y is not None else current.y
        body.transform = (b2Vec2(new_x, new_y), body.angle)

    def freeze(self, name: str) -> None:
        """Freeze object by zeroing all velocities.

        Args:
            name: Object name
        """
        body = self._get_body(name)
        from Box2D import b2Vec2

        body.linearVelocity = b2Vec2(0, 0)
        body.angularVelocity = 0.0

    def _get_body(self, name: str):
        """Get Box2D body by name."""
        body = self.engine.bodies.get(name)
        if body is None:
            raise ValueError(f"Object '{name}' not found")
        return body

    # === Standard Gym Methods ===

    def _setup_action_space(self):
        """Set up the action space based on action_type and level configuration."""
        if self.action_type == "continuous":
            if len(self._level.action_objects) == 0:
                self.action_space = gym.spaces.Box(
                    low=np.array([]), high=np.array([]), dtype=np.float32
                )
            else:
                # Check for custom action bounds in level metadata
                if (
                    self._level.metadata is not None
                    and "action_bounds" in self._level.metadata
                ):
                    action_bounds = self._level.metadata["action_bounds"]
                else:
                    action_bounds = {
                        "x": (-5.0, 5.0),
                        "y": (-5.0, 5.0),
                        "r": (0.1, 1.5),
                    }
                x_low, x_high = action_bounds["x"]
                y_low, y_high = action_bounds["y"]
                r_low, r_high = action_bounds["r"]

                # Each action object gets (x, y, size)
                action_dim = len(self._level.action_objects) * 3
                lows = np.array(
                    [x_low, y_low, r_low] * len(self._level.action_objects),
                    dtype=np.float32,
                )
                highs = np.array(
                    [x_high, y_high, r_high] * len(self._level.action_objects),
                    dtype=np.float32,
                )
                self.action_space = gym.spaces.Box(
                    low=lows, high=highs, shape=(action_dim,), dtype=np.float32
                )
        elif self.action_type == "discrete":
            num_objects = len(self._level.action_objects)
            if num_objects == 0:
                self.action_space = gym.spaces.MultiDiscrete(
                    np.array([], dtype=np.int64)
                )
            else:
                x_y_bins = int((5.0 - (-5.0)) / 0.1 + 1)  # 101
                size_bins = int((1.5 - 0.1) / 0.1 + 1)  # 15

                self._discrete_step = 0.1
                self._discrete_bins = (x_y_bins, x_y_bins, size_bins)
                self._discrete_lows = (-5.0, -5.0, 0.1)

                nvec = np.array(list(self._discrete_bins) * num_objects, dtype=np.int64)
                self.action_space = gym.spaces.MultiDiscrete(nvec)
        else:
            raise ValueError(f"Unknown action_type: {self.action_type}")

    def _setup_observation_space(self):
        """Set up the observation space based on observation_type."""
        if self.observation_type == "physics_state":
            self.observation_space = gym.spaces.Dict(
                {
                    "objects": gym.spaces.Dict(
                        {
                            name: gym.spaces.Dict(
                                {
                                    "position": gym.spaces.Box(
                                        low=-10, high=10, shape=(2,), dtype=np.float32
                                    ),
                                    "velocity": gym.spaces.Box(
                                        low=-10, high=10, shape=(2,), dtype=np.float32
                                    ),
                                    "angle": gym.spaces.Box(
                                        low=-np.pi,
                                        high=np.pi,
                                        shape=(),
                                        dtype=np.float32,
                                    ),
                                    "angular_velocity": gym.spaces.Box(
                                        low=-10, high=10, shape=(), dtype=np.float32
                                    ),
                                    "type": gym.spaces.Text(max_length=20),
                                }
                            )
                            for name in self._level.objects.keys()
                        }
                    ),
                    "contacts": gym.spaces.Box(
                        low=0,
                        high=1,
                        shape=(len(self._level.objects), len(self._level.objects)),
                        dtype=np.int8,
                    ),
                    "step_count": gym.spaces.Discrete(self.max_steps + 1),
                }
            )
        elif self.observation_type == "image":
            width, height = self.image_size
            if self.discrete_colors:
                self.observation_space = gym.spaces.Box(
                    low=0, high=7, shape=(height, width), dtype=np.uint8
                )
            else:
                self.observation_space = gym.spaces.Box(
                    low=0, high=255, shape=(height, width, 3), dtype=np.uint8
                )
        elif self.observation_type == "both":
            self.observation_space = gym.spaces.Dict(
                {
                    "physics_state": gym.spaces.Dict(
                        {
                            "objects": gym.spaces.Dict(
                                {
                                    name: gym.spaces.Dict(
                                        {
                                            "position": gym.spaces.Box(
                                                low=-10,
                                                high=10,
                                                shape=(2,),
                                                dtype=np.float32,
                                            ),
                                            "velocity": gym.spaces.Box(
                                                low=-10,
                                                high=10,
                                                shape=(2,),
                                                dtype=np.float32,
                                            ),
                                            "angle": gym.spaces.Box(
                                                low=-np.pi,
                                                high=np.pi,
                                                shape=(),
                                                dtype=np.float32,
                                            ),
                                            "angular_velocity": gym.spaces.Box(
                                                low=-10,
                                                high=10,
                                                shape=(),
                                                dtype=np.float32,
                                            ),
                                            "type": gym.spaces.Text(max_length=20),
                                        }
                                    )
                                    for name in self._level.objects.keys()
                                }
                            ),
                            "contacts": gym.spaces.Box(
                                low=0,
                                high=1,
                                shape=(
                                    len(self._level.objects),
                                    len(self._level.objects),
                                ),
                                dtype=np.int8,
                            ),
                            "step_count": gym.spaces.Discrete(self.max_steps + 1),
                        }
                    ),
                    "image": gym.spaces.Box(
                        low=0,
                        high=255 if not self.discrete_colors else 7,
                        shape=(
                            (self.image_size[1], self.image_size[0], 3)
                            if not self.discrete_colors
                            else (self.image_size[1], self.image_size[0])
                        ),
                        dtype=np.uint8,
                    ),
                }
            )
        else:
            raise ValueError(f"Unknown observation_type: {self.observation_type}")

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """Reset the environment to initial state.

        Args:
            seed: Random seed for reproducibility (optional)
            options: Additional options for reset (e.g., interventions)

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)

        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        # Reset engine and state
        self.engine.reset(self._level)
        self.action_placed = False
        self.step_count = 0
        self._rollout_complete = False

        # Load interventions from options
        self._active_interventions = []
        if options and "interventions" in options:
            self._active_interventions = options["interventions"]

        observation = self._get_observation()

        info = {
            "level_name": self._level.name,
            "action_objects": self._level.action_objects,
            "total_objects": len(self._level.objects),
            "step_count": self.step_count,
            "action_placed": self.action_placed,
            "success": False,
            "truncated": False,
            "terminated": False,
        }

        return observation, info

    def step(
        self, action: Union[List[Tuple[float, float, float]], np.ndarray]
    ) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """Execute one episode: place objects and run full simulation to completion.

        This is a one-shot environment - step() can only be called once per episode.
        After calling step(), you must call reset() to start a new episode.

        Args:
            action: Action to execute. For continuous actions, should be:
                - List of (x, y, size) tuples for each action object
                - Numpy array of shape (n_objects * 3,) with flattened coordinates

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        if self._rollout_complete:
            raise RuntimeError(
                "Episode already complete. Call reset() to start a new episode."
            )

        validation_result = self._validate_action_with_failure(action)
        if validation_result["invalid"]:
            obs = self._get_observation()
            info = {
                "level_name": self._level.name,
                "step_count": 0,
                "action_placed": False,
                "success": False,
                "terminated": True,
                "truncated": False,
                "world_stationary": False,
                "validation_error": validation_result["error"],
                "invalid_action": True,
            }
            self._rollout_complete = True
            return obs, -1.0, True, False, info

        self._place_action_objects(validation_result["action"])
        self.action_placed = True

        obs, reward, terminated, truncated, info = self._run_simulation_rollout()
        self._rollout_complete = True
        return obs, reward, terminated, truncated, info

    def _step_physics(self):
        """Execute a single physics step (internal method)."""
        self.engine.world.Step(
            self.config.time_step,
            self.config.velocity_iters,
            self.config.position_iters,
        )

        self.engine._validate_contact_distances()
        self.engine.time_update(self.config.time_step)
        self.step_count += 1

    def _run_simulation_rollout(self):
        """Run physics simulation to completion."""
        interventions = []

        for step_index in range(self.max_steps):
            self._step_physics()
            self.render()

            success = self._level.success_condition(self.engine)
            terminated = success
            truncated = step_index >= self.max_steps - 1

            if success or truncated:
                break

        obs = self._get_observation()
        reward = self._calculate_reward(success, truncated)
        info = self._get_info_dict(success, terminated, truncated)
        info["interventions"] = interventions

        return obs, reward, terminated, truncated, info

    def _validate_action(
        self, action: Union[List[Tuple[float, float, float]], np.ndarray]
    ) -> List[Tuple[float, float, float]]:
        """Validate action format and convert to standard format."""
        if len(self._level.action_objects) == 0:
            if action != [] and not (
                isinstance(action, np.ndarray) and action.size == 0
            ):
                raise ValueError(
                    f"No action objects in level, but received action: {action}"
                )
            return []

        expected_dim = len(self._level.action_objects) * 3

        if self.action_type == "discrete":
            x_bins, y_bins, s_bins = getattr(self, "_discrete_bins", (101, 101, 15))
            x_low, y_low, s_low = getattr(self, "_discrete_lows", (-5.0, -5.0, 0.1))
            step = getattr(self, "_discrete_step", 0.1)

            if isinstance(action, np.ndarray):
                if action.shape != (expected_dim,):
                    raise ValueError(
                        f"Expected action shape ({expected_dim},), got {action.shape}"
                    )
                indices = action.astype(np.int64).tolist()
            elif isinstance(action, list):
                if len(action) != len(self._level.action_objects):
                    raise ValueError(
                        f"Expected {len(self._level.action_objects)} action tuples, got {len(action)}"
                    )
                indices = []
                for i, pos in enumerate(action):
                    if not isinstance(pos, (tuple, list)) or len(pos) != 3:
                        raise ValueError(
                            f"Action {i} must be a tuple/list of length 3 (x, y, size), got {pos}"
                        )
                    if not all(isinstance(v, (int, np.integer)) for v in pos):
                        raise ValueError(
                            f"Action {i} must contain integer indices for discrete mode, got {pos}"
                        )
                    indices.extend([int(pos[0]), int(pos[1]), int(pos[2])])
            else:
                raise ValueError(
                    f"Action must be list of tuples or numpy array, got {type(action)}"
                )

            converted_action: List[Tuple[float, float, float]] = []
            for i in range(0, expected_dim, 3):
                xi, yi, si = int(indices[i]), int(indices[i + 1]), int(indices[i + 2])
                if not (0 <= xi < x_bins and 0 <= yi < y_bins and 0 <= si < s_bins):
                    raise ValueError(
                        f"Discrete indices out of bounds at object {i // 3}: {(xi, yi, si)}"
                    )
                x = round(x_low + step * xi, PRECISION)
                y = round(y_low + step * yi, PRECISION)
                s = round(s_low + step * si, PRECISION)
                converted_action.append((float(x), float(y), float(s)))
        else:
            if isinstance(action, np.ndarray):
                if action.shape != (expected_dim,):
                    raise ValueError(
                        f"Expected action shape ({expected_dim},), got {action.shape}"
                    )
                converted_action = [
                    (action[i], action[i + 1], np.clip(action[i + 2], 0.1, 1.5))
                    for i in range(0, len(action), 3)
                ]
            elif isinstance(action, list):
                if len(action) != len(self._level.action_objects):
                    raise ValueError(
                        f"Expected {len(self._level.action_objects)} action tuples, got {len(action)}"
                    )
                for i, pos in enumerate(action):
                    if not isinstance(pos, (tuple, list)) or len(pos) != 3:
                        raise ValueError(
                            f"Action {i} must be a tuple/list of length 3 (x, y, size), got {pos}"
                        )
                    if not all(isinstance(x, (int, float)) for x in pos):
                        raise ValueError(
                            f"Action {i} coordinates must be numbers, got {pos}"
                        )
                converted_action = [
                    (x, y, np.clip(s, 0.1, 1.5)) for (x, y, s) in action
                ]
            else:
                raise ValueError(
                    f"Action must be list of tuples or numpy array, got {type(action)}"
                )

        return converted_action

    def _validate_action_with_failure(
        self, action: Union[List[Tuple[float, float, float]], np.ndarray]
    ) -> Dict[str, Any]:
        """Validate action and return failure information instead of raising exceptions."""
        try:
            converted_action = self._validate_action(action)

            for i, (x, y, radius) in enumerate(converted_action):
                if not self._is_within_bounds(x, y, radius):
                    min_coord = -5.0 + radius
                    max_coord = 5.0 - radius
                    return {
                        "invalid": True,
                        "action": None,
                        "error": (
                            f"Action object {i} at ({x:.2f}, {y:.2f}) with radius {radius:.2f} is "
                            f"OUT OF BOUNDS. Valid range: {min_coord:.2f} <= x <= {max_coord:.2f}, "
                            f"{min_coord:.2f} <= y <= {max_coord:.2f}."
                        ),
                    }
                if self._would_collide_with_objects(x, y, radius):
                    return {
                        "invalid": True,
                        "action": None,
                        "error": (
                            f"Action object {i} at ({x:.2f}, {y:.2f}) with radius {radius:.2f} "
                            f"OVERLAPS with an existing object. Choose a position that does not "
                            f"overlap with the green ball, blue ball, or other level objects."
                        ),
                    }

            return {"invalid": False, "action": converted_action, "error": None}
        except ValueError as e:
            return {"invalid": True, "action": None, "error": str(e)}

    def _is_valid_placement(self, x: float, y: float, radius: float) -> bool:
        """Check if placing an object at (x, y) with given radius is valid."""
        if not self._is_within_bounds(x, y, radius):
            return False
        if self._would_collide_with_objects(x, y, radius):
            return False
        return True

    def _is_within_bounds(self, x: float, y: float, radius: float) -> bool:
        """Check if object placement is within world boundaries."""
        min_x = -5.0 + radius
        max_x = 5.0 - radius
        min_y = -5.0 + radius
        max_y = 5.0 - radius
        return min_x <= x <= max_x and min_y <= y <= max_y

    def _would_collide_with_objects(self, x: float, y: float, radius: float) -> bool:
        """Check if object placement would collide with existing objects."""
        for name, obj in self._level.objects.items():
            if name in self._level.action_objects:
                continue

            if hasattr(obj, "radius"):
                distance = np.sqrt((x - obj.x) ** 2 + (y - obj.y) ** 2)
                if distance <= (radius + getattr(obj, "radius", 0.1)):
                    return True
            elif hasattr(obj, "length"):
                if self._circle_intersects_bar(x, y, radius, obj):
                    return True
            elif hasattr(obj, "total_width"):
                if self._circle_intersects_basket(x, y, radius, obj):
                    return True

        return False

    def _circle_intersects_bar(self, cx: float, cy: float, radius: float, bar) -> bool:
        """Check if circle intersects with rotated bar using precise geometry."""
        angle_rad = np.radians(-bar.angle)
        dx = cx - bar.x
        dy = cy - bar.y
        local_x = dx * np.cos(angle_rad) - dy * np.sin(angle_rad)
        local_y = dx * np.sin(angle_rad) + dy * np.cos(angle_rad)

        half_length = bar.length / 2
        half_thickness = bar.thickness / 2

        closest_x = np.clip(local_x, -half_length, half_length)
        closest_y = np.clip(local_y, -half_thickness, half_thickness)

        dist_sq = (local_x - closest_x) ** 2 + (local_y - closest_y) ** 2
        return dist_sq <= radius**2

    def _circle_intersects_basket(
        self, cx: float, cy: float, radius: float, basket
    ) -> bool:
        """Check if circle intersects with any basket wall (not the interior)."""
        half_width = basket.total_width / 2
        half_height = basket.total_height / 2
        wall_thickness = getattr(
            basket, "wall_thickness", 0.1 * min(basket.total_width, basket.total_height)
        )

        basket_left = basket.x - half_width
        basket_right = basket.x + half_width
        basket_bottom = basket.y - half_height
        basket_top = basket.y + half_height

        walls = [
            (basket_left, basket_bottom, basket_left + wall_thickness, basket_top),
            (basket_right - wall_thickness, basket_bottom, basket_right, basket_top),
            (basket_left, basket_bottom, basket_right, basket_bottom + wall_thickness),
            (basket_left, basket_top - wall_thickness, basket_right, basket_top),
        ]

        for wall in walls:
            if self._circle_intersects_rect(cx, cy, radius, *wall):
                return True
        return False

    def _circle_intersects_rect(
        self,
        cx: float,
        cy: float,
        radius: float,
        left: float,
        bottom: float,
        right: float,
        top: float,
    ) -> bool:
        """Check if circle intersects with axis-aligned rectangle."""
        closest_x = np.clip(cx, left, right)
        closest_y = np.clip(cy, bottom, top)
        distance = np.sqrt((cx - closest_x) ** 2 + (cy - closest_y) ** 2)
        return distance <= radius

    def _place_action_objects(self, action: List[Tuple[float, float, float]]):
        """Place action objects at the specified positions and sizes."""
        if len(action) != len(self._level.action_objects):
            raise ValueError(
                f"Expected {len(self._level.action_objects)} positions, got {len(action)}"
            )
        self.engine.place_action_objects(action)

    def _get_observation(self) -> Any:
        """Get the current observation based on observation_type."""
        if self.observation_type == "physics_state":
            return self._get_physics_state()
        elif self.observation_type == "image":
            return self._get_image_observation()
        elif self.observation_type == "both":
            return {
                "physics_state": self._get_physics_state(),
                "image": self._get_image_observation(),
            }
        else:
            raise ValueError(f"Unknown observation_type: {self.observation_type}")

    def _get_physics_state(self) -> Dict[str, Any]:
        """Get the physics state observation."""
        if self.engine.world is None:
            return {}

        objects_state = {}
        object_names = list(self._level.objects.keys())

        for name in object_names:
            if name in self.engine.bodies:
                body = self.engine.bodies[name]
                objects_state[name] = {
                    "position": np.array(
                        [body.position.x, body.position.y], dtype=np.float32
                    ),
                    "velocity": np.array(
                        [body.linearVelocity.x, body.linearVelocity.y], dtype=np.float32
                    ),
                    "angle": float(body.angle),
                    "angular_velocity": float(body.angularVelocity),
                    "type": type(self._level.objects[name]).__name__,
                }
            else:
                obj = self._level.objects[name]
                objects_state[name] = {
                    "position": np.array([obj.x, obj.y], dtype=np.float32),
                    "velocity": np.array([0.0, 0.0], dtype=np.float32),
                    "angle": float(obj.angle),
                    "angular_velocity": 0.0,
                    "type": type(obj).__name__,
                }

        contact_matrix = np.zeros(
            (len(object_names), len(object_names)), dtype=np.bool_
        )
        for i, name1 in enumerate(object_names):
            for j, name2 in enumerate(object_names):
                if i != j and self.engine.has_contact(name1, name2):
                    contact_matrix[i, j] = True

        return {
            "objects": objects_state,
            "contacts": contact_matrix,
            "step_count": self.step_count,
        }

    def _get_image_observation(self) -> np.ndarray:
        """Get image observation by rendering current simulation state."""
        from interphyre.render import OpenCVRenderer

        width, height = self.image_size

        world_size = 10.0
        target_ppm = min(width, height) / world_size
        ppm = min(target_ppm, self.image_ppm)

        renderer = OpenCVRenderer(width=width, height=height, ppm=ppm)

        if self.discrete_colors:
            image = renderer.render_discrete(self.engine)
        else:
            image = renderer.render(self.engine)

        renderer.close()
        return image

    def _calculate_reward(self, success: bool, truncated: bool) -> float:
        """Calculate the reward for the current state."""
        if success:
            return 1.0
        elif truncated:
            return -0.1
        else:
            return 0.0

    def _get_info_dict(
        self, success: bool, terminated: bool, truncated: bool
    ) -> Dict[str, Any]:
        """Get the info dictionary for the current step."""
        if terminated and truncated:
            truncated = False

        info = {
            "level_name": self._level.name,
            "step_count": self.step_count,
            "action_placed": self.action_placed,
            "success": success,
            "terminated": terminated,
            "truncated": truncated,
            "world_stationary": (
                self.engine.world_is_stationary() if self.engine.world else False
            ),
        }

        if hasattr(self.engine, "get_contact_statistics"):
            contact_stats = self.engine.get_contact_statistics()
            info["contact_statistics"] = contact_stats

        if self.config.enable_profiling:
            perf_stats = self.engine.profiler.get_stats()
            info["performance_stats"] = perf_stats

        return info

    def simulate(
        self,
        steps: Optional[int] = None,
        return_trace: bool = False,
        verbose: bool = False,
    ) -> Optional[List[Tuple[Any, float, bool, bool, Dict[str, Any]]]]:
        """Public method for debugging/profiling: run simulation with custom parameters."""
        if steps is None:
            steps = self.config.max_steps

        if self.engine.world is None:
            raise ValueError(
                "World is not initialized. Call reset() before simulating."
            )

        trace = []
        status = "running"
        terminated = False

        if self.config.enable_profiling:
            self.engine.profiler.start_step_batch()

        for i in range(steps):
            self._step_physics()

            done = self._level.success_condition(self.engine)
            if done:
                status = "success"
            elif self.engine.world_is_stationary():
                status = "world_is_stationary"
            elif i == steps - 1:
                status = "timeout"
                terminated = True

            if return_trace:
                observation = self._get_observation()
                reward = self._calculate_reward(done, terminated)
                info = self._get_info_dict(done, done, terminated)
                trace.append((observation, reward, done, terminated, info))

            self.render()

            if verbose:
                print(f"Step {i+1}/{steps}, status: {status}")
            if done or terminated:
                break

        if self.config.enable_profiling:
            self.engine.profiler.end_step_batch(steps)

        return trace if return_trace else None

    def render(self):
        """Render the current state."""
        if self.renderer:
            self.renderer.render(self.engine)

    def close(self):
        """Close the environment and clean up resources."""
        if self.renderer:
            self.renderer.close()

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics from the engine's profiler."""
        return self.engine.profiler.get_stats()

    def reset_profiler(self):
        """Reset the performance profiler."""
        self.engine.profiler.reset()

    def get_contact_log(self) -> List[Dict[str, Any]]:
        """Get the full contact event log for research purposes."""
        return self.engine.get_contact_log()

    def get_contact_statistics(self) -> Dict[str, Any]:
        """Get statistics about all contacts for research purposes."""
        return self.engine.get_contact_statistics()

    def get_level_info(self) -> Dict[str, Any]:
        """Get information about the current level."""
        return {
            "name": self._level.name,
            "action_objects": self._level.action_objects,
            "total_objects": len(self._level.objects),
            "object_types": {
                name: type(obj).__name__ for name, obj in self._level.objects.items()
            },
            "metadata": self._level.metadata,
        }
