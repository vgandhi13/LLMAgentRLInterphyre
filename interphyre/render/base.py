from abc import ABC, abstractmethod

# Color palette for rendering objects
COLORS = {
    "green": (32, 201, 162),
    "red": (235, 82, 52),
    "blue": (32, 93, 214),
    "black": (0, 0, 0),
    "gray": (200, 200, 200),
    "purple": (81, 56, 150),
    "yellow": (255, 211, 67),
    "white": (255, 255, 255),
}

# Discrete color mapping for single-channel representation
# Only includes colors actually used in Interphyre levels
# 0 = transparent/background, 1-7 = object colors
DISCRETE_COLORS = {
    0: (255, 255, 255),  # transparent/background (white)
    1: (32, 201, 162),  # green
    2: (235, 82, 52),  # red
    3: (0, 0, 0),  # black
    4: (81, 56, 150),  # purple
    5: (200, 200, 200),  # gray
    6: (32, 93, 214),  # blue
    7: (255, 0, 0),  # walls (bright red)
}

# Reverse mapping from RGB to discrete color index
RGB_TO_DISCRETE = {rgb: idx for idx, rgb in DISCRETE_COLORS.items()}


class Renderer(ABC):
    """Abstract base class for rendering physics simulations.

    This class defines the interface that all renderers must implement
    to visualize the physics simulation. Renderers handle the conversion
    from physics world coordinates to screen coordinates and drawing
    of all objects in the simulation.
    """

    @abstractmethod
    def render(self, engine) -> None:
        """Render the current state of the physics simulation.

        Args:
            engine: The Box2DEngine containing the physics world to render
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Close and clean up any resources used by the renderer.

        This method should be called when the renderer is no longer needed
        to properly clean up resources like windows, contexts, etc.
        """
        pass
