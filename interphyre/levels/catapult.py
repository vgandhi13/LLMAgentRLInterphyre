import numpy as np
from typing import cast
from interphyre.objects import Ball, Bar, PhyreObject, Basket
from interphyre.level import Level
from interphyre.levels import register_level
from interphyre.config import MIN_X, MAX_X, MIN_Y, MAX_Y


def success_condition(engine):
    success_time = engine.config.default_success_time
    return engine.is_in_contact_for_duration("green_ball", "blue_ball", success_time)


@register_level
def build_level(seed=None) -> Level:
    rng = np.random.default_rng(seed)

    # Parameters
    step_diff = rng.uniform(-0.3, 0.3)
    step_base = rng.uniform(0.01, 0.1)
    basket_scale = rng.uniform(0.6, 1.2)
    basket_right_ratio = rng.uniform(0.9, 0.98)
    ledge_angle = rng.uniform(-10, 10)
    left_side_ratio = rng.uniform(0.05, 0.2)

    # Calculate step heights
    if step_diff > 0:
        platform_height_ratio = step_base
        ledge_height_ratio = platform_height_ratio + step_diff
    else:
        ledge_height_ratio = step_base
        platform_height_ratio = ledge_height_ratio - step_diff

    # Left step (black platform)
    platform_length = 3.0
    platform_left_edge = MIN_X + left_side_ratio * 10
    platform_bottom = MIN_Y + platform_height_ratio * 10

    platform = Bar(
        left=platform_left_edge,
        right=platform_left_edge + platform_length,
        y=platform_bottom,
        thickness=0.2,
        color="black",
        dynamic=False,
    )

    # Pivot ball on left step
    pivot_radius = 0.75
    pivot_ball = Ball(
        x=platform.left + pivot_radius,
        y=platform.y + platform.thickness / 2 + pivot_radius,
        radius=pivot_radius,
        color="gray",
        dynamic=True,
    )

    # Catapult arm (bar)
    bar_length = 4.0
    bar_thickness = 0.2
    bar_left_edge = MIN_X + 0.2
    # Bar's bottom should rest on pivot's top
    bar_center_y = pivot_ball.y + pivot_radius + bar_thickness / 2

    catapult_bar = Bar(
        left=bar_left_edge,
        right=bar_left_edge + bar_length,
        y=bar_center_y,
        thickness=bar_thickness,
        color="gray",
        dynamic=True,
    )

    # Green ball (ball to launch)
    green_radius = 0.25
    green_left_edge = MIN_X + 0.2
    green_ball = Ball(
        x=green_left_edge + green_radius,
        y=catapult_bar.y + catapult_bar.thickness / 2 + green_radius,
        radius=green_radius,
        color="green",
        dynamic=True,
    )

    # Blocker ball at top
    blocker_radius = 0.5
    blocker_ball = Ball(
        x=pivot_ball.x,
        y=MAX_Y - blocker_radius,
        radius=blocker_radius,
        color="black",
        dynamic=False,
    )

    # Right step (ledge)
    ledge_length = 3.0
    ledge_bottom = MIN_Y + ledge_height_ratio * 10
    ledge_center_x = MAX_X - ledge_length / 2

    ledge = Bar.from_point_and_angle(
        x=ledge_center_x,
        y=ledge_bottom,
        angle=ledge_angle,
        length=ledge_length,
        thickness=0.2,
        color="black",
        dynamic=False,
    )

    # Basket on ledge
    basket_right_edge = MIN_X + basket_right_ratio * 10
    basket_dims = Basket.calculate_dimensions(basket_scale)
    basket_x = basket_right_edge - basket_dims["total_width"] / 2
    basket_y = ledge.y + ledge.thickness / 2

    basket = Basket(
        x=basket_x,
        y=basket_y,
        scale=basket_scale,
        angle=ledge_angle,
        anchor="bottom_center",
        color="gray",
        dynamic=True,
    )

    # Blue ball in basket
    blue_ball_radius = basket_scale * 0.35
    blue_ball = Ball(
        x=basket_x,
        y=basket_y + blue_ball_radius + 0.4,
        radius=blue_ball_radius,
        color="blue",
        dynamic=True,
    )

    # User-placeable red ball
    red_ball = Ball(
        x=0.0,
        y=0.0,
        radius=0.5,
        color="red",
        dynamic=True,
    )

    objects = {
        "green_ball": green_ball,
        "red_ball": red_ball,
        "blue_ball": blue_ball,
        "ledge": ledge,
        "basket": basket,
        "blocker_ball": blocker_ball,
        "platform": platform,
        "pivot_ball": pivot_ball,
        "catapult_bar": catapult_bar,
    }

    return Level(
        name="catapult",
        objects=cast(dict[str, PhyreObject], objects),
        action_objects=["red_ball"],
        success_condition=success_condition,
        metadata={"description": "Launch the green ball into the basket with the blue ball."},
    )
