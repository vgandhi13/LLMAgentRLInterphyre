import numpy as np
from typing import cast
from interphyre.objects import Ball, Bar, PhyreObject
from interphyre.level import Level
from interphyre.levels import register_level
from interphyre.config import MAX_X, MAX_Y, MIN_X, MIN_Y, WORLD_WIDTH, WORLD_HEIGHT


def success_condition(engine):
    success_time = engine.config.default_success_time
    return engine.is_in_contact_for_duration("green_ball", "purple_floor", success_time)


@register_level
def build_level(seed=None) -> Level:
    """Build locust swarm level.

    NOTE: This level has inherent difficulty variability across seeds due to random
    chain-based obstacle generation. Some seeds may be impossible while others may
    be trivial. Seed filtering during data collection is recommended.
    """
    rng = np.random.default_rng(seed)

    ball_x = rng.uniform(MIN_X + 1, MAX_X - 1)
    ball_y = MAX_Y - 0.1 * WORLD_HEIGHT
    ball_radius = 0.5
    star_radius = 0.25

    green_ball = Ball(
        x=ball_x,
        y=ball_y,
        radius=ball_radius,
        color="green",
        dynamic=True,
    )

    red_ball_radius = rng.uniform(0.3, 0.6)
    red_ball = Ball(
        x=0.0,
        y=0.0,
        radius=red_ball_radius,
        color="red",
        dynamic=True,
    )

    top = ball_y - 5 * ball_radius
    stars = []

    def gen_chain(start_x, start_y):
        """
        Generate a chain of stars with normal random steps.
        """
        angle = rng.uniform() * 2 * np.pi
        angle_diff = rng.uniform() * 2 * np.pi / 10
        chain_stars = [(start_x, start_y)]
        line_length = 1
        n_valid = 0
        max_points = rng.integers(15, 30)
        max_step_size = 2 * ball_radius + 2 * star_radius + 0.05

        while n_valid < max_points:
            if line_length >= 3 and rng.uniform() < 0.2:
                # Branch to random existing point
                x, y = chain_stars[rng.integers(len(chain_stars))]
                line_length = 1
                angle = rng.uniform() * 2 * np.pi
                angle_diff = rng.uniform() * 2 * np.pi / 10
            else:
                line_length += 1
                # Normal random step
                step = rng.uniform(0.5, max_step_size)

                angle += angle_diff
                dx, dy = step * np.cos(angle), step * np.sin(angle)
                x, y = chain_stars[-1]
                x += dx
                y += dy

            if y >= top:
                continue

            chain_stars.append((x, y))

            # Convert to normalized coordinates for bounds check
            norm_x = (x - MIN_X) / WORLD_WIDTH
            norm_y = (y - MIN_Y) / WORLD_HEIGHT
            if 0.0 < norm_x < 1 and 0.0 < norm_y < 1:
                n_valid += 1

        return chain_stars

    # Generate two separate star chains
    for offset in [0.2, 0.7]:
        start_x = MIN_X + offset * WORLD_WIDTH
        start_y = MIN_Y + 0.5 * WORLD_HEIGHT
        stars.extend(gen_chain(start_x, start_y))

    # Create star objects
    star_objects = {}
    for i, (x, y) in enumerate(stars):
        # Convert to normalized coordinates for bounds check
        norm_x = (x - MIN_X) / WORLD_WIDTH
        norm_y = (y - MIN_Y) / WORLD_HEIGHT
        if 0 <= norm_x <= 1 and 0 <= norm_y <= 1:
            star_ball = Ball(
                x=x,
                y=y,
                radius=star_radius,
                color="black",
                dynamic=False,
            )
            star_objects[f"star_{i}"] = star_ball

    purple_floor = Bar.from_point_and_angle(
        x=0.0,
        y=-4.9,
        length=WORLD_WIDTH,
        thickness=0.2,
        angle=0.0,
        color="purple",
        dynamic=False,
    )

    objects = {
        "green_ball": green_ball,
        "red_ball": red_ball,
        "purple_floor": purple_floor,
        **star_objects,
    }

    return Level(
        name="locust_swarm",
        objects=cast(dict[str, PhyreObject], objects),
        action_objects=["red_ball"],
        success_condition=success_condition,
        metadata={
            "description": "Get the green ball to the purple floor by using the red ball to navigate through the two separate clouds of star obstacles."
        },
    )
