import argparse

import numpy as np
import math

DEFAULT_CONFIG_PATH = "saved_configuration.pkl"
DEFAULT_MODEL_PATH = "models/ppo_active_sensing"
DEFAULT_ROBOT_COUNT = 3
DEFAULT_MIN_DISTANCE = 25.0
DEFAULT_MAX_DISTANCE = 100.0
DEFAULT_GENERATION_RADIUS = 100.0
DEFAULT_WORLD_PADDING = 75.0
DEFAULT_MAX_STEPS = 100
DEFAULT_ROTATION_LIMIT = math.radians(35)
DEFAULT_ROTATION_STEP = math.radians(6)
DEFAULT_ARTIFACTS_DIR = "artifacts/cooperative_rl_union"


def parse_sensor_parameter(
    value: float | list[float] | tuple[float, ...] | np.ndarray,
    sensor_count: int,
    name: str,
) -> np.ndarray:
    if np.isscalar(value):
        return np.full(sensor_count, float(value), dtype=np.float32)  # type: ignore

    parsed = np.asarray(value, dtype=np.float32)
    if parsed.shape != (sensor_count,):
        raise ValueError(
            f"{name} must be a scalar or have exactly {sensor_count} values, got shape {parsed.shape}."
        )
    return parsed


def parse_angle_list(value: str | None) -> list[float] | None:
    if value is None:
        return None
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        return None
    return [math.radians(float(part)) for part in parts]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cooperative active sensing with PPO on top of GA-optimized sensor layouts."
    )
    parser.add_argument(
        "mode",
        choices=("train", "evaluate", "render-random"),
        help="Run PPO training, evaluate a saved model, or render a random rollout.",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="Path to saved GA configuration pickle.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_PATH,
        help="Path used for saving/loading PPO models.",
    )
    parser.add_argument(
        "--timesteps", type=int, default=50_000, help="Training timesteps for PPO."
    )
    parser.add_argument(
        "--episodes", type=int, default=3, help="Evaluation episode count."
    )
    parser.add_argument(
        "--robots",
        type=int,
        default=DEFAULT_ROBOT_COUNT,
        help="Number of robots in the team.",
    )
    parser.add_argument(
        "--max-steps", type=int, default=DEFAULT_MAX_STEPS, help="Episode horizon."
    )
    parser.add_argument(
        "--rotation-limits-deg",
        default=None,
        help="Optional comma-separated per-sensor rotation limits in degrees.",
    )
    parser.add_argument(
        "--rotation-steps-deg",
        default=None,
        help="Optional comma-separated per-sensor rotation step sizes in degrees.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic actions during evaluation.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render evaluation episodes step by step.",
    )
    parser.add_argument(
        "--tensorboard-log",
        default=None,
        help="Optional tensorboard log directory for PPO training.",
    )
    parser.add_argument(
        "--artifacts-dir",
        default=DEFAULT_ARTIFACTS_DIR,
        help="Directory where training and evaluation artifacts will be written.",
    )
    return parser
