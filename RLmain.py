import argparse
import math
from pathlib import Path

import numpy as np

from env.RLEnv import make_env
from utils.files_management import ensure_directory, write_csv, write_json
from utils.parsing import build_arg_parser, parse_angle_list
from stable_baselines3.common.vec_env import VecNormalize

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


def linear_schedule(initial_value):
    def func(progress_remaining):
        return progress_remaining * initial_value

    return func


def train(args: argparse.Namespace) -> None:
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor

    artifacts_dir = ensure_directory(args.artifacts_dir)
    train_artifacts_dir = ensure_directory(artifacts_dir / "train")

    env = make_env(
        config_path=args.config,
        robot_count=args.robots,
        max_steps=args.max_steps,
        rotation_limit=parse_angle_list(args.rotation_limits_deg)
        or DEFAULT_ROTATION_LIMIT,
        rotation_step=parse_angle_list(args.rotation_steps_deg)
        or DEFAULT_ROTATION_STEP,
        seed=args.seed,
    )

    env = Monitor(
        env,
        filename=str(train_artifacts_dir / "monitor.csv"),
        info_keywords=(
            "outward_coverage",
            "teammate_visibility",
            "overlap_penalty",
            "steering_penalty",
            "outward_sensor_ratio",
            "workspace_area",
            "mean_rotation_limit_deg",
            "mean_rotation_step_deg",
        ),
    )

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        n_steps=1024,
        batch_size=128,
        learning_rate=linear_schedule(5e-4),
        gamma=0,
        tensorboard_log=args.tensorboard_log,
    )
    model.learn(total_timesteps=args.timesteps)

    model_path = Path(args.model)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)

    write_json(
        train_artifacts_dir / "training_run.json",
        {
            "mode": "train",
            "config": args.config,
            "model_path": str(model_path),
            "timesteps": args.timesteps,
            "robots": args.robots,
            "max_steps": args.max_steps,
            "seed": args.seed,
            "tensorboard_log": args.tensorboard_log,
            "coverage_mode": "union_area",
            "rotation_limits_deg": [
                float(v)
                for v in np.degrees(env.unwrapped.rotation_limits)  # type: ignore
            ],
            "rotation_steps_deg": [
                float(v)
                for v in np.degrees(env.unwrapped.rotation_steps)  # type: ignore
            ],
            "monitor_file": str(train_artifacts_dir / "monitor.csv"),
        },
    )
    env.close()


def evaluate(args: argparse.Namespace) -> None:
    from stable_baselines3 import PPO

    artifacts_dir = ensure_directory(args.artifacts_dir)
    eval_artifacts_dir = ensure_directory(artifacts_dir / "evaluate")

    env = make_env(
        config_path=args.config,
        robot_count=args.robots,
        max_steps=args.max_steps,
        rotation_limit=parse_angle_list(args.rotation_limits_deg)
        or DEFAULT_ROTATION_LIMIT,
        rotation_step=parse_angle_list(args.rotation_steps_deg)
        or DEFAULT_ROTATION_STEP,
        seed=args.seed,
    )

    model = PPO.load(args.model, env=env)
    episode_rows: list[dict] = []
    step_rows: list[dict] = []

    for episode in range(args.episodes):
        observation, info = env.reset(
            seed=None if args.seed is None else args.seed + episode
        )
        total_reward = 0.0
        episode_seed = None if args.seed is None else args.seed + episode

        while True:
            action, _ = model.predict(observation, deterministic=args.deterministic)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_rows.append(
                {
                    "episode": episode,
                    "seed": episode_seed,
                    "step": int(info["step"]),
                    "reward": float(reward),
                    "cumulative_reward": float(total_reward),
                    "outward_coverage": float(info["outward_coverage"]),
                    "teammate_visibility": float(info["teammate_visibility"]),
                    "overlap_penalty": float(info["overlap_penalty"]),
                    "steering_penalty": float(info["steering_penalty"]),
                    "outward_sensor_ratio": float(info["outward_sensor_ratio"]),
                    "workspace_area": float(info["workspace_area"]),
                    "mean_rotation_limit_deg": float(info["mean_rotation_limit_deg"]),
                    "mean_rotation_step_deg": float(info["mean_rotation_step_deg"]),
                }
            )

            if args.render:
                env.render()

            if terminated or truncated:
                episode_rows.append(
                    {
                        "episode": episode,
                        "seed": episode_seed,
                        "total_reward": float(total_reward),
                        "outward_coverage": float(info["outward_coverage"]),
                        "teammate_visibility": float(info["teammate_visibility"]),
                        "overlap_penalty": float(info["overlap_penalty"]),
                        "steering_penalty": float(info["steering_penalty"]),
                        "outward_sensor_ratio": float(info["outward_sensor_ratio"]),
                        "workspace_area": float(info["workspace_area"]),
                        "mean_rotation_limit_deg": float(
                            info["mean_rotation_limit_deg"]
                        ),
                        "mean_rotation_step_deg": float(info["mean_rotation_step_deg"]),
                        "steps": int(info["step"]),
                        "deterministic": bool(args.deterministic),
                    }
                )
                print(
                    "Episode {} reward {:.3f} outward_coverage {:.3f} teammate_visibility {:.3f} overlap {:.3f}".format(
                        episode,
                        total_reward,
                        info["outward_coverage"],
                        info["teammate_visibility"],
                        info["overlap_penalty"],
                    )
                )
                break

    write_csv(eval_artifacts_dir / "evaluation_episodes.csv", episode_rows)
    write_csv(eval_artifacts_dir / "evaluation_steps.csv", step_rows)
    write_json(
        eval_artifacts_dir / "evaluation_summary.json",
        {
            "mode": "evaluate",
            "config": args.config,
            "model_path": args.model,
            "episodes": args.episodes,
            "robots": args.robots,
            "max_steps": args.max_steps,
            "seed": args.seed,
            "deterministic": args.deterministic,
            "coverage_mode": "union_area",
            "rotation_limits_deg": [float(v) for v in np.degrees(env.rotation_limits)],
            "rotation_steps_deg": [float(v) for v in np.degrees(env.rotation_steps)],
            "mean_total_reward": float(
                np.mean([row["total_reward"] for row in episode_rows])
            )
            if episode_rows
            else 0.0,
            "mean_outward_coverage": float(
                np.mean([row["outward_coverage"] for row in episode_rows])
            )
            if episode_rows
            else 0.0,
            "mean_teammate_visibility": float(
                np.mean([row["teammate_visibility"] for row in episode_rows])
            )
            if episode_rows
            else 0.0,
            "mean_overlap_penalty": float(
                np.mean([row["overlap_penalty"] for row in episode_rows])
            )
            if episode_rows
            else 0.0,
            "episode_csv": str(eval_artifacts_dir / "evaluation_episodes.csv"),
            "step_csv": str(eval_artifacts_dir / "evaluation_steps.csv"),
        },
    )
    env.close()


def render_random_policy(args: argparse.Namespace) -> None:
    artifacts_dir = ensure_directory(args.artifacts_dir)
    random_artifacts_dir = ensure_directory(artifacts_dir / "render_random")

    env = make_env(
        config_path=args.config,
        robot_count=args.robots,
        max_steps=args.max_steps,
        rotation_limit=parse_angle_list(args.rotation_limits_deg)
        or DEFAULT_ROTATION_LIMIT,
        rotation_step=parse_angle_list(args.rotation_steps_deg)
        or DEFAULT_ROTATION_STEP,
        seed=args.seed,
        render_mode="human",
    )
    observation, info = env.reset(seed=args.seed)
    total_reward = 0.0
    step_rows: list[dict] = []

    for _ in range(args.max_steps):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_rows.append(
            {
                "step": int(info["step"]),
                "reward": float(reward),
                "cumulative_reward": float(total_reward),
                "outward_coverage": float(info["outward_coverage"]),
                "teammate_visibility": float(info["teammate_visibility"]),
                "overlap_penalty": float(info["overlap_penalty"]),
                "steering_penalty": float(info["steering_penalty"]),
                "outward_sensor_ratio": float(info["outward_sensor_ratio"]),
                "workspace_area": float(info["workspace_area"]),
                "mean_rotation_limit_deg": float(info["mean_rotation_limit_deg"]),
                "mean_rotation_step_deg": float(info["mean_rotation_step_deg"]),
            }
        )
        # env.render()
        if terminated or truncated:
            break

    write_csv(random_artifacts_dir / "random_rollout_steps.csv", step_rows)
    write_json(
        random_artifacts_dir / "random_rollout_summary.json",
        {
            "mode": "render-random",
            "config": args.config,
            "robots": args.robots,
            "max_steps": args.max_steps,
            "seed": args.seed,
            "coverage_mode": "union_area",
            "rotation_limits_deg": [float(v) for v in np.degrees(env.rotation_limits)],
            "rotation_steps_deg": [float(v) for v in np.degrees(env.rotation_steps)],
            "total_reward": float(total_reward),
            "final_outward_coverage": float(info["outward_coverage"]),
            "final_teammate_visibility": float(info["teammate_visibility"]),
            "final_overlap_penalty": float(info["overlap_penalty"]),
            "step_csv": str(random_artifacts_dir / "random_rollout_steps.csv"),
        },
    )
    print(
        "Random rollout reward {:.3f} outward_coverage {:.3f} teammate_visibility {:.3f} overlap {:.3f}".format(
            total_reward,
            info["outward_coverage"],
            info["teammate_visibility"],
            info["overlap_penalty"],
        )
    )
    env.close()


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.mode == "train":
        train(args)
    elif args.mode == "evaluate":
        evaluate(args)
    else:
        render_random_policy(args)


if __name__ == "__main__":
    main()
