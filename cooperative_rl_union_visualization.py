import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import shapely
from matplotlib import animation
from shapely.plotting import plot_polygon

from RLmain import (
    DEFAULT_CONFIG_PATH,
    ensure_directory,
    make_env,
)


DEFAULT_MODEL_PATH = "/tmp/cooperative_rl_union_longrun/models/ppo_active_sensing"
DEFAULT_OUTPUT_PATH = "experiments/cooperative_rl_union/adaptivity.gif"
DEFAULT_FRAMES = 180
DEFAULT_FPS = 12
ROBOT_LENGTH_SCALE = 1.8
ROBOT_WIDTH_SCALE = 0.9
PAIR_CLEARANCE_MARGIN = 22.0
ROTATION_CHANGE_THRESHOLD = math.radians(0.35)
GLOBAL_FOV_MARGIN = math.radians(8.0)
VISUAL_ROTATION_LIMIT = math.radians(12.0)


def rotation_matrix(angle: float) -> np.ndarray:
    return np.array(
        [
            [math.cos(angle), -math.sin(angle)],
            [math.sin(angle), math.cos(angle)],
        ],
        dtype=np.float32,
    )


def heading_from_positions(
    current: np.ndarray, previous: np.ndarray | None, fallback: float = 0.0
) -> np.ndarray:
    if previous is None:
        return np.full(len(current), fallback, dtype=np.float32)

    headings = []
    for cur, prev in zip(current, previous):
        delta = cur - prev
        norm = float(np.linalg.norm(delta))
        if norm < 1e-6:
            headings.append(fallback)
        else:
            headings.append(math.atan2(float(delta[1]), float(delta[0])))
    return np.asarray(headings, dtype=np.float32)


def visual_robot_size(robot_size: np.ndarray) -> tuple[float, float]:
    base = float(max(robot_size))
    return base * ROBOT_LENGTH_SCALE, base * ROBOT_WIDTH_SCALE


def oriented_robot_polygon(
    position: np.ndarray, heading: float, robot_size: np.ndarray
) -> shapely.Polygon:
    length, width = visual_robot_size(robot_size)
    half_l = length / 2.0
    half_w = width / 2.0
    local = np.array(
        [
            [half_l, half_w],
            [half_l, -half_w],
            [-half_l, -half_w],
            [-half_l, half_w],
        ],
        dtype=np.float32,
    )
    rot = rotation_matrix(heading)
    points = [(position + rot @ point).tolist() for point in local]
    return shapely.Polygon(points)


def clamp_positions(
    positions: np.ndarray, world_low: np.ndarray, world_high: np.ndarray
) -> np.ndarray:
    return np.clip(positions, world_low, world_high)


def separate_robot_bodies(
    env,
    positions: np.ndarray,
    headings: np.ndarray,
    world_low: np.ndarray,
    world_high: np.ndarray,
) -> np.ndarray:
    adjusted = positions.copy()

    for _ in range(12):
        moved = False
        for i in range(len(adjusted)):
            for j in range(i + 1, len(adjusted)):
                body_i = oriented_robot_polygon(
                    adjusted[i], float(headings[i]), env.robot_size
                )
                body_j = oriented_robot_polygon(
                    adjusted[j], float(headings[j]), env.robot_size
                )
                if not body_i.intersects(body_j):
                    continue

                moved = True
                delta = adjusted[i] - adjusted[j]
                dist = float(np.linalg.norm(delta))
                if dist < 1e-6:
                    direction = np.array([1.0, 0.0], dtype=np.float32)
                else:
                    direction = delta / dist

                overlap_extent = (
                    max(body_i.intersection(body_j).area ** 0.5, 1.0)
                    + PAIR_CLEARANCE_MARGIN * 0.15
                )
                shift = 0.5 * overlap_extent * direction
                adjusted[i] += shift
                adjusted[j] -= shift

        adjusted = clamp_positions(adjusted, world_low, world_high)
        if not moved:
            break

    return adjusted


def safe_positions(
    env,
    candidate_positions: np.ndarray,
    previous_positions: np.ndarray | None = None,
) -> np.ndarray:
    headings = heading_from_positions(candidate_positions, previous_positions)
    return separate_robot_bodies(
        env, candidate_positions, headings, env.world_low, env.world_high
    )


def fov_sector(
    origin: tuple[float, float],
    radius: float,
    start_angle: float,
    end_angle: float,
    samples: int = 18,
) -> shapely.Polygon:
    if end_angle < start_angle:
        end_angle += 2.0 * math.pi
    angles = np.linspace(start_angle, end_angle, samples)
    points = [origin]
    for angle in angles:
        points.append(
            (
                origin[0] + radius * math.cos(float(angle)),
                origin[1] + radius * math.sin(float(angle)),
            )
        )
    return shapely.Polygon(points)


def local_fov_triangle(
    origin: tuple[float, float], radius: float, center_angle: float, fov_angle: float
) -> shapely.Polygon:
    return shapely.Polygon(
        [
            origin,
            (
                origin[0] + radius * math.cos(center_angle + fov_angle / 2.0),
                origin[1] + radius * math.sin(center_angle + fov_angle / 2.0),
            ),
            (
                origin[0] + radius * math.cos(center_angle - fov_angle / 2.0),
                origin[1] + radius * math.sin(center_angle - fov_angle / 2.0),
            ),
        ]
    )


def apply_visual_rotation_limit(env) -> None:
    visual_limits = np.minimum(
        env.rotation_limits,
        np.full(env.sensor_count, VISUAL_ROTATION_LIMIT, dtype=np.float32),
    )
    env.sensor_offsets = np.clip(
        env.sensor_offsets,
        -visual_limits[np.newaxis, :],
        visual_limits[np.newaxis, :],
    )


def build_visual_sensor_geometry(
    env, robot_positions: np.ndarray, robot_headings: np.ndarray
):
    bodies: list[shapely.Polygon] = []
    global_fovs: list[list[shapely.Polygon]] = []
    local_fovs: list[list[shapely.Polygon]] = []
    local_origins: list[list[tuple[float, float]]] = []

    for robot_idx in range(env.robot_count):
        heading = float(robot_headings[robot_idx])
        body = oriented_robot_polygon(
            robot_positions[robot_idx], heading, env.robot_size
        )
        bodies.append(body)

        rot = rotation_matrix(heading)
        robot_global: list[shapely.Polygon] = []
        robot_local: list[shapely.Polygon] = []
        robot_origins: list[tuple[float, float]] = []

        for sensor_idx in range(env.sensor_count):
            base_position = env.base_positions[sensor_idx]
            sensor_origin = robot_positions[robot_idx] + rot @ base_position
            sensor_origin_tuple = (float(sensor_origin[0]), float(sensor_origin[1]))
            robot_origins.append(sensor_origin_tuple)

            base_heading = float(env.base_rotations[sensor_idx] + heading)
            current_heading = float(
                base_heading + env.sensor_offsets[robot_idx, sensor_idx]
            )
            radius = float(env.sensor_ranges[sensor_idx])
            fov_angle = float(env.sensor_fov[sensor_idx])
            visual_rotation_limit = min(
                float(env.rotation_limits[sensor_idx]),
                VISUAL_ROTATION_LIMIT,
            )

            total_global_fov = min(
                fov_angle + 2.0 * GLOBAL_FOV_MARGIN,
                fov_angle + 2.0 * visual_rotation_limit,
            )
            global_polygon = fov_sector(
                sensor_origin_tuple,
                radius,
                current_heading - total_global_fov / 2.0,
                current_heading + total_global_fov / 2.0,
            )
            local_polygon = local_fov_triangle(
                sensor_origin_tuple,
                radius,
                current_heading,
                fov_angle,
            )

            robot_global.append(global_polygon)
            robot_local.append(local_polygon)

        global_fovs.append(robot_global)
        local_fovs.append(robot_local)
        local_origins.append(robot_origins)

    return bodies, global_fovs, local_fovs, local_origins


class SmoothFormationTrajectory:
    def __init__(
        self, robot_count: int, world_low: np.ndarray, world_high: np.ndarray, seed: int
    ):
        self.robot_count = robot_count
        self.world_low = world_low.astype(np.float32)
        self.world_high = world_high.astype(np.float32)
        self.rng = np.random.default_rng(seed)

        self.center_base = (self.world_low + self.world_high) / 2.0
        self.center_amp = (self.world_high - self.world_low) * np.array(
            [0.18, 0.14], dtype=np.float32
        )
        self.center_phase = self.rng.uniform(0.0, 2.0 * math.pi, size=2)

        angles = np.linspace(0.0, 2.0 * math.pi, robot_count, endpoint=False)
        self.base_offsets = np.stack(
            [
                np.cos(angles),
                np.sin(angles),
            ],
            axis=1,
        ).astype(np.float32)
        self.base_offsets *= self.rng.uniform(38.0, 60.0, size=(robot_count, 1)).astype(
            np.float32
        )

        self.radial_amp = self.rng.uniform(5.0, 14.0, size=robot_count).astype(
            np.float32
        )
        self.radial_phase = self.rng.uniform(
            0.0, 2.0 * math.pi, size=robot_count
        ).astype(np.float32)
        self.local_amp = self.rng.uniform(3.0, 8.0, size=(robot_count, 2)).astype(
            np.float32
        )
        self.local_phase = self.rng.uniform(
            0.0, 2.0 * math.pi, size=(robot_count, 2)
        ).astype(np.float32)
        self.local_freq = self.rng.uniform(0.6, 1.3, size=(robot_count, 2)).astype(
            np.float32
        )

        self.rotation_phase = float(self.rng.uniform(0.0, 2.0 * math.pi))
        self.rotation_wobble_phase = float(self.rng.uniform(0.0, 2.0 * math.pi))
        self.rotation_rate = float(self.rng.uniform(0.7, 1.0))

    def positions_at(self, t: float) -> np.ndarray:
        center = self.center_base + np.array(
            [
                self.center_amp[0] * math.sin(0.23 * t + self.center_phase[0]),
                self.center_amp[1] * math.sin(0.17 * t + self.center_phase[1]),
            ],
            dtype=np.float32,
        )

        formation_angle = (
            0.35 * math.sin(0.09 * t + self.rotation_phase)
            + self.rotation_rate * 0.03 * t
            + 0.18 * math.sin(0.05 * t + self.rotation_wobble_phase)
        )
        rot = rotation_matrix(formation_angle)

        positions = []
        for idx in range(self.robot_count):
            radial_scale = 1.0 + 0.18 * math.sin(0.11 * t + self.radial_phase[idx])
            offset = self.base_offsets[idx] * radial_scale
            local_wiggle = np.array(
                [
                    self.local_amp[idx, 0]
                    * math.sin(
                        self.local_freq[idx, 0] * 0.08 * t + self.local_phase[idx, 0]
                    ),
                    self.local_amp[idx, 1]
                    * math.cos(
                        self.local_freq[idx, 1] * 0.08 * t + self.local_phase[idx, 1]
                    ),
                ],
                dtype=np.float32,
            )
            positions.append(center + rot @ offset + local_wiggle)

        return np.asarray(positions, dtype=np.float32)


def save_metrics_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fd:
        writer = csv.DictWriter(fd, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize smooth multi-robot motion with policy-driven sensor adaptation."
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="Path to saved GA configuration pickle.",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL_PATH, help="Path to the trained PPO model."
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT_PATH, help="Output GIF path."
    )
    parser.add_argument(
        "--frames", type=int, default=DEFAULT_FRAMES, help="Animation frame count."
    )
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help="Animation FPS.")
    parser.add_argument(
        "--robots", type=int, default=3, help="Number of robots to visualize."
    )
    parser.add_argument(
        "--seed", type=int, default=7, help="Random seed for the trajectory generator."
    )
    parser.add_argument(
        "--show", action="store_true", help="Display the animation window after saving."
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    from stable_baselines3 import PPO

    output_path = Path(args.output)
    ensure_directory(output_path.parent)

    env = make_env(
        config_path=args.config,
        robot_count=args.robots,
        max_steps=args.frames,
        seed=args.seed,
    )
    model = PPO.load(args.model, env=env)

    env.reset(seed=args.seed)
    trajectory = SmoothFormationTrajectory(
        robot_count=args.robots,
        world_low=env.world_low,
        world_high=env.world_high,
        seed=args.seed,
    )

    robot_colors = plt.cm.Set2(np.linspace(0.0, 1.0, args.robots))
    fig, (ax, metrics_ax) = plt.subplots(
        1,
        2,
        figsize=(14, 8),
        gridspec_kw={"width_ratios": [2.3, 1.0]},
    )
    trails = [[] for _ in range(args.robots)]
    metrics_rows: list[dict] = []
    previous_positions: np.ndarray | None = None
    previous_offsets: np.ndarray | None = None
    initial_local_fovs = None
    frozen_positions = safe_positions(env, trajectory.positions_at(0))

    def draw_frame(frame_idx: int):
        nonlocal previous_positions, previous_offsets, initial_local_fovs
        baseline_phase = frame_idx < args.fps
        if baseline_phase:
            env.robot_positions = frozen_positions.copy()
        else:
            motion_frame = frame_idx - args.fps
            env.robot_positions = safe_positions(
                env,
                trajectory.positions_at(motion_frame),
                previous_positions=frozen_positions
                if motion_frame == 0
                else previous_positions,
            )
        env.outward_sensor_mask = env._compute_outward_sensor_mask()

        if baseline_phase:
            env.sensor_offsets.fill(0.0)
            info = env._get_info()
            reward = float(info["reward"])
        else:
            observation = env._get_observation()
            action, _ = model.predict(observation, deterministic=True)
            _, reward, _, _, info = env.step(action)
        apply_visual_rotation_limit(env)
        saved_offsets = env.sensor_offsets.copy()
        env.sensor_offsets.fill(0.0)
        fixed_info = env._get_info()
        env.sensor_offsets[:] = saved_offsets
        robot_headings = heading_from_positions(env.robot_positions, previous_positions)
        bodies, global_fovs, local_fovs, _ = build_visual_sensor_geometry(
            env, env.robot_positions, robot_headings
        )
        team_center = np.mean(env.robot_positions, axis=0)
        previous_positions = env.robot_positions.copy()
        if initial_local_fovs is None:
            initial_local_fovs = local_fovs

        if previous_offsets is None:
            offset_change = np.full_like(env.sensor_offsets, np.inf, dtype=np.float32)
        else:
            offset_change = np.abs(env.sensor_offsets - previous_offsets)
        previous_offsets = env.sensor_offsets.copy()

        ax.clear()
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.15)
        ax.set_xlim(env.world_low[0], env.world_high[0])
        ax.set_ylim(env.world_low[1], env.world_high[1])
        ax.set_facecolor("#f7f5ef")

        plot_polygon(
            env.workspace_polygon,
            ax=ax,
            add_points=False,
            color=(0.75, 0.74, 0.70, 0.22),
        )

        for robot_idx in range(args.robots):
            trails[robot_idx].append(env.robot_positions[robot_idx].copy())
            trail = np.asarray(trails[robot_idx][-25:])
            if len(trail) > 1:
                ax.plot(
                    trail[:, 0],
                    trail[:, 1],
                    color=robot_colors[robot_idx],
                    alpha=0.45,
                    linewidth=1.6,
                )

        for robot_idx in range(args.robots):
            plot_polygon(
                bodies[robot_idx],
                ax=ax,
                add_points=False,
                color=(0.08, 0.08, 0.08, 0.82),
            )

            for sensor_idx in range(env.sensor_count):
                is_changing = (
                    offset_change[robot_idx, sensor_idx] > ROTATION_CHANGE_THRESHOLD
                )
                if env.outward_sensor_mask[robot_idx, sensor_idx]:
                    base_rgb = robot_colors[robot_idx][:3]
                else:
                    base_rgb = (0.25, 0.45, 0.85)

                if baseline_phase:
                    plot_polygon(
                        initial_local_fovs[robot_idx][sensor_idx],
                        ax=ax,
                        add_points=False,
                        color=(*base_rgb, 0.32),
                        alpha=0.32,
                    )
                    continue

                if not is_changing:
                    continue

                global_alpha = 0.06
                local_alpha = 0.58
                global_polygon = global_fovs[robot_idx][sensor_idx]
                local_polygon = local_fovs[robot_idx][sensor_idx]

                plot_polygon(
                    global_polygon,
                    ax=ax,
                    add_points=False,
                    color=(*base_rgb, global_alpha),
                )
                plot_polygon(
                    local_polygon,
                    ax=ax,
                    add_points=False,
                    color=(*base_rgb, local_alpha),
                    alpha=local_alpha,
                )

            ax.scatter(
                env.robot_positions[robot_idx, 0],
                env.robot_positions[robot_idx, 1],
                color=robot_colors[robot_idx],
                s=42,
                edgecolors="black",
                linewidths=0.6,
                zorder=5,
            )
            ax.text(
                env.robot_positions[robot_idx, 0] + 2.5,
                env.robot_positions[robot_idx, 1] + 2.5,
                f"R{robot_idx}",
                fontsize=10,
                weight="bold",
            )

        ax.scatter(
            team_center[0],
            team_center[1],
            marker="x",
            s=80,
            color="black",
            linewidths=1.5,
        )
        if baseline_phase:
            ax.set_title(
                "Initial Local FOVs | frame {:03d} | baseline view before adaptive steering".format(
                    frame_idx
                )
            )
        else:
            ax.set_title(
                "Adaptive Multi-Robot Sensing | frame {:03d} | reward {:.2f} | union {:.3f} | vis {:.3f} | overlap {:.3f}".format(
                    frame_idx,
                    reward,
                    info["outward_coverage"],
                    info["teammate_visibility"],
                    info["overlap_penalty"],
                )
            )
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        metrics_rows.append(
            {
                "frame": frame_idx,
                "reward": float(reward),
                "outward_coverage": float(info["outward_coverage"]),
                "fixed_outward_coverage": float(fixed_info["outward_coverage"]),
                "teammate_visibility": float(info["teammate_visibility"]),
                "overlap_penalty": float(info["overlap_penalty"]),
                "steering_penalty": float(info["steering_penalty"]),
            }
        )

        frames = [row["frame"] for row in metrics_rows]
        coverage = [row["outward_coverage"] for row in metrics_rows]
        fixed_coverage = [row["fixed_outward_coverage"] for row in metrics_rows]
        overlap = [row["overlap_penalty"] for row in metrics_rows]

        metrics_ax.clear()
        metrics_ax.set_facecolor("#fbfaf6")
        metrics_ax.grid(True, alpha=0.2)
        metrics_ax.plot(
            frames, coverage, color="#1f7a8c", linewidth=2.4, label="Union coverage"
        )
        metrics_ax.plot(
            frames,
            fixed_coverage,
            color="#5c677d",
            linewidth=1.7,
            linestyle="--",
            label="Fixed coverage",
        )
        metrics_ax.plot(
            frames, overlap, color="#d1495b", linewidth=1.6, label="Overlap"
        )
        metrics_ax.scatter(frames[-1], coverage[-1], color="#1f7a8c", s=30, zorder=5)
        metrics_ax.scatter(
            frames[-1], fixed_coverage[-1], color="#5c677d", s=22, zorder=5
        )
        metrics_ax.set_xlim(0, max(args.frames - 1, 1))
        metrics_ax.set_ylim(0.0, 1.05)
        metrics_ax.set_title("Dynamic Coverage Trace")
        metrics_ax.set_xlabel("Frame")
        metrics_ax.set_ylabel("Metric value")
        metrics_ax.legend(loc="upper right", frameon=True)

    anim = animation.FuncAnimation(
        fig, draw_frame, frames=args.frames, interval=1000 / args.fps, repeat=False
    )
    writer = animation.PillowWriter(fps=args.fps)
    anim.save(output_path, writer=writer)

    metrics_path = output_path.with_suffix(".csv")
    save_metrics_csv(metrics_path, metrics_rows)

    if args.show:
        plt.show()
    else:
        plt.close(fig)

    env.close()
    print(f"Saved animation to {output_path}")
    print(f"Saved per-frame metrics to {metrics_path}")


if __name__ == "__main__":
    main()
