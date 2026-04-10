"""Simple viewer for visualizing Interphyre levels and solutions."""

import argparse
import json
from pathlib import Path
from typing import Optional, List, Union, Tuple

from interphyre import InterphyreEnv, SimulationConfig
from interphyre.render.pygame import PygameRenderer
from interphyre.render.video import VideoRecorder, generate_video_filename


def visualize_action(
    level_name: str,
    seed: int,
    action: Union[List[float], Tuple[float, float, float]],
    pause_time: float = 2.0,
    record_video: bool = False,
    video_format: str = "mp4",
    output_dir: str = "outputs",
) -> bool:
    """View a level with a specific action.

    Args:
        level_name: Name of the level
        seed: Random seed
        action: Action as [x, y, r] or (x, y, r)
        pause_time: Pause duration after simulation (seconds)
        record_video: Whether to record video
        video_format: Video format ('mp4' or 'gif')
        output_dir: Output directory for videos

    Returns:
        True if level was solved
    """
    # Normalize action
    if isinstance(action, (list, tuple)) and len(action) == 3:
        action = tuple(action)
    else:
        raise ValueError(f"Action must be [x, y, r] or (x, y, r), got {action}")

    print(f"Viewing {level_name} (seed={seed}): {action}")

    # Setup renderer
    if record_video:
        video_path = generate_video_filename(
            level_name, seed, output_dir, video_format, label="action"
        )
        renderer = VideoRecorder(
            width=600, height=600, ppm=60,
            video_format=video_format, fps=60, output_path=video_path
        )
        print(f"Recording to: {video_path}")
    else:
        renderer = PygameRenderer(width=600, height=600, ppm=60)

    # Run simulation
    config = SimulationConfig(fps=60, time_step=1/60)
    env = InterphyreEnv(level_name, seed=seed, config=config)
    env.renderer = renderer

    try:
        env.reset()
        obs, reward, terminated, truncated, info = env.step([action])
        success = info.get("success", False)

        print(f"Result: {'SUCCESS' if success else 'FAIL'} (reward={reward})")

        if not record_video and hasattr(renderer, 'wait'):
            renderer.wait(int(pause_time * 1000))

        return success
    finally:
        renderer.close()
        env.close()


def visualize_solution_from_file(
    level_name: str,
    solutions_file: str,
    seed: Optional[int] = None,
    pause_time: float = 2.0,
    record_video: bool = False,
    video_format: str = "mp4",
    output_dir: str = "outputs",
):
    """View solutions from a JSON file.

    Args:
        level_name: Name of the level
        solutions_file: Path to JSON file with solutions
        seed: Optional seed filter (only view this seed)
        pause_time: Pause duration between solutions
        record_video: Whether to record videos
        video_format: Video format
        output_dir: Output directory
    """
    with open(solutions_file) as f:
        solutions = json.load(f)

    # Filter by level and seed
    matching = []
    for sol in solutions:
        if sol.get("level") == level_name:
            if seed is None or sol.get("seed") == seed:
                matching.append(sol)

    print(f"Found {len(matching)} solutions for {level_name}")

    successful = 0
    for i, sol in enumerate(matching, 1):
        print(f"\n[{i}/{len(matching)}]")
        success = visualize_action(
            level_name=sol.get("level", level_name),
            seed=sol["seed"],
            action=sol["action"],
            pause_time=pause_time,
            record_video=record_video,
            video_format=video_format,
            output_dir=output_dir,
        )
        if success:
            successful += 1

    print(f"\nResults: {successful}/{len(matching)} successful")


def visualize_all_solutions(
    solutions_file: str,
    pause_time: float = 2.0,
    record_video: bool = False,
    video_format: str = "mp4",
    output_dir: str = "outputs",
):
    """View all solutions from a JSON file.

    Args:
        solutions_file: Path to JSON file with solutions
        pause_time: Pause duration between solutions
        record_video: Whether to record videos
        video_format: Video format
        output_dir: Output directory
    """
    with open(solutions_file) as f:
        solutions = json.load(f)

    print(f"Visualizing {len(solutions)} solutions")

    for i, sol in enumerate(solutions, 1):
        print(f"\n[{i}/{len(solutions)}]")
        visualize_action(
            level_name=sol["level"],
            seed=sol["seed"],
            action=sol["action"],
            pause_time=pause_time,
            record_video=record_video,
            video_format=video_format,
            output_dir=output_dir,
        )


def run_random_demo(
    level_name: str,
    seed: Optional[int] = None,
    max_trials: int = 20,
    pause_time: float = 1.0,
    record_video: bool = False,
    video_format: str = "mp4",
    output_dir: str = "outputs",
):
    """Run random agent demo.

    Args:
        level_name: Name of the level
        seed: Random seed (None for random)
        max_trials: Maximum number of trials
        pause_time: Pause between trials
        record_video: Whether to record video
        video_format: Video format
        output_dir: Output directory
    """
    import numpy as np

    if seed is not None:
        np.random.seed(seed)

    print(f"Random demo: {level_name} (max {max_trials} trials)")

    config = SimulationConfig(fps=60, time_step=1/60)
    level_seed = seed or np.random.randint(0, 100000)

    # Setup renderer
    if record_video:
        video_path = generate_video_filename(
            level_name, seed or 0, output_dir, video_format, label="demo"
        )
        renderer = VideoRecorder(
            width=600, height=600, ppm=60,
            video_format=video_format, fps=60, output_path=video_path
        )
        env = InterphyreEnv(level_name, seed=level_seed, config=config)
        env.renderer = renderer
        print(f"Recording to: {video_path}")
    else:
        env = InterphyreEnv(level_name, seed=level_seed, config=config, render_mode="human")

    try:
        for trial in range(1, max_trials + 1):
            # Find a valid action
            max_action_attempts = 100
            action = None
            for attempt in range(max_action_attempts):
                env.reset()
                env.render()  # Render initial state

                candidate = env.action_space.sample()
                try:
                    # Convert numpy floats to Python floats for validation
                    action_tuple = (float(candidate[0]), float(candidate[1]), float(candidate[2]))
                    obs, reward, terminated, truncated, info = env.step([action_tuple])
                    # Check if action was invalid (reward -1.0)
                    if info.get("invalid_action", False):
                        continue
                    action = action_tuple
                    break
                except ValueError:
                    continue

            if action is None:
                print(f"\nCouldn't find valid action after {max_action_attempts} attempts")
                break

            print(f"\nTrial {trial}: {tuple(action)}")
            success = info.get("success", False)
            print(f"  {'SUCCESS' if success else 'FAIL'} (reward={reward})")

            if success:
                print(f"\nSolved in {trial} trials!")
                if not record_video and hasattr(env.renderer, 'wait'):
                    env.renderer.wait(int(pause_time * 2000))  # Pause longer on success
                break

            if not record_video and hasattr(env.renderer, 'wait'):
                env.renderer.wait(int(pause_time * 1000))
    finally:
        env.close()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Interphyre Viewer")
    parser.add_argument("level", help="Level name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--action", nargs=3, type=float, metavar=("X", "Y", "R"),
                        help="Action: x y radius")
    parser.add_argument("--solutions", help="Path to solutions JSON file")
    parser.add_argument("--demo", action="store_true", help="Run random demo")
    parser.add_argument("--trials", type=int, default=20, help="Max trials for demo")
    parser.add_argument("--pause", type=float, default=2.0, help="Pause duration (seconds)")
    parser.add_argument("--record", action="store_true", help="Record video")
    parser.add_argument("--format", default="mp4", choices=["mp4", "gif"],
                        help="Video format")
    parser.add_argument("--output-dir", default="outputs", help="Output directory")

    args = parser.parse_args()

    if args.demo:
        run_random_demo(
            args.level, args.seed, args.trials, args.pause,
            args.record, args.format, args.output_dir
        )
    elif args.solutions:
        if args.level == "all":
            visualize_all_solutions(
                args.solutions, args.pause, args.record, args.format, args.output_dir
            )
        else:
            visualize_solution_from_file(
                args.level, args.solutions, args.seed, args.pause,
                args.record, args.format, args.output_dir
            )
    elif args.action:
        visualize_action(
            args.level, args.seed, args.action, args.pause,
            args.record, args.format, args.output_dir
        )
    else:
        parser.error("Must specify --action, --solutions, or --demo")


if __name__ == "__main__":
    main()
