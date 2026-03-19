import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import shapely
from matplotlib import animation
from shapely import affinity, ops
from shapely.geometry import Point
from shapely.plotting import plot_polygon

from cooperative_rl_union import DEFAULT_CONFIG_PATH, ensure_directory, make_env
from cooperative_rl_union_visualization import (
    DEFAULT_FPS,
    DEFAULT_FRAMES,
    DEFAULT_MODEL_PATH,
    ROTATION_CHANGE_THRESHOLD,
    SmoothFormationTrajectory,
    apply_visual_rotation_limit,
    build_visual_sensor_geometry,
    heading_from_positions,
    oriented_robot_polygon,
)


DEFAULT_OUTPUT_PATH = "experiments/cooperative_rl_union/adaptivity_with_obstacles.gif"
CLEARANCE_MARGIN = 6.0
PAIR_CLEARANCE_MARGIN = 22.0


def save_metrics_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fd:
        writer = csv.DictWriter(fd, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize smooth multi-robot motion with obstacle-aware sensor occlusion."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to saved GA configuration pickle.")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Path to the trained PPO model.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="Output GIF path.")
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES, help="Animation frame count.")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help="Animation FPS.")
    parser.add_argument("--robots", type=int, default=3, help="Number of robots to visualize.")
    parser.add_argument("--seed", type=int, default=11, help="Random seed for trajectories and obstacle placement.")
    parser.add_argument("--obstacles", type=int, default=3, help="Number of occluding obstacles.")
    parser.add_argument("--show", action="store_true", help="Display the animation window after saving.")
    return parser


def generate_obstacles(
    world_low: np.ndarray,
    world_high: np.ndarray,
    count: int,
    seed: int,
    forbidden_region: shapely.Polygon | None = None,
) -> list[shapely.Polygon]:
    rng = np.random.default_rng(seed)
    center = (world_low + world_high) / 2.0
    span = (world_high - world_low)
    obstacles: list[shapely.Polygon] = []
    ring_centroid = None if forbidden_region is None else np.array(forbidden_region.centroid.coords[0], dtype=np.float32)
    target_radius = None
    if forbidden_region is not None:
        minx, miny, maxx, maxy = forbidden_region.bounds
        ring_radius_x = max(maxx - minx, 1.0) / 2.0
        ring_radius_y = max(maxy - miny, 1.0) / 2.0
        target_radius = np.array([ring_radius_x + 18.0, ring_radius_y + 18.0], dtype=np.float32)

    for _ in range(500):
        if len(obstacles) >= count:
            break

        width = float(rng.uniform(34.0, 58.0))
        height = float(rng.uniform(30.0, 54.0))
        if ring_centroid is not None and target_radius is not None:
            angle = float(rng.uniform(0.0, 2.0 * math.pi))
            radial_jitter = rng.uniform(-8.0, 8.0, size=2).astype(np.float32)
            anchor = ring_centroid + np.array(
                [
                    target_radius[0] * math.cos(angle),
                    target_radius[1] * math.sin(angle),
                ],
                dtype=np.float32,
            ) + radial_jitter
            anchor = np.clip(anchor, world_low + 16.0, world_high - 16.0)
            dx = float(anchor[0] - center[0])
            dy = float(anchor[1] - center[1])
        else:
            dx = float(rng.uniform(-0.34, 0.34) * span[0])
            dy = float(rng.uniform(-0.28, 0.28) * span[1])
        base = shapely.box(-width / 2.0, -height / 2.0, width / 2.0, height / 2.0)
        rotated = affinity.rotate(base, float(rng.uniform(-35.0, 35.0)), origin=(0, 0))
        translated = affinity.translate(rotated, xoff=float(center[0] + dx), yoff=float(center[1] + dy))

        if forbidden_region is not None and translated.intersects(forbidden_region):
            continue
        if any(translated.distance(other) < 8.0 for other in obstacles):
            continue

        obstacles.append(translated)

    return obstacles


def trajectory_forbidden_region(
    env,
    trajectory: SmoothFormationTrajectory,
    frames: int,
) -> shapely.Polygon:
    points: list[Point] = []

    for frame_idx in range(frames):
        positions = trajectory.positions_at(frame_idx)
        for position in positions:
            points.append(Point(float(position[0]), float(position[1])))

    point_cloud = ops.unary_union(points)
    radius = float(max(env.robot_size) / 2.0 + CLEARANCE_MARGIN + 4.0)
    return point_cloud.buffer(radius)


def clamp_positions(positions: np.ndarray, world_low: np.ndarray, world_high: np.ndarray) -> np.ndarray:
    return np.clip(positions, world_low, world_high)


def robot_body_polygon(env, position: np.ndarray) -> shapely.Polygon:
    return env._robot_polygon(position)


def push_out_of_obstacles(
    env,
    positions: np.ndarray,
    obstacles: list[shapely.Polygon],
    margin: float,
) -> np.ndarray:
    adjusted = positions.copy()
    clearance_obstacles = [obstacle.buffer(margin) for obstacle in obstacles]
    center = (env.world_low + env.world_high) / 2.0

    for robot_idx in range(len(adjusted)):
        for _ in range(12):
            body = robot_body_polygon(env, adjusted[robot_idx])
            hit = None
            for obstacle in clearance_obstacles:
                if body.intersects(obstacle):
                    hit = obstacle
                    break
            if hit is None:
                break

            obstacle_center = np.array(hit.centroid.coords[0], dtype=np.float32)
            direction = adjusted[robot_idx] - obstacle_center
            norm = float(np.linalg.norm(direction))
            if norm < 1e-6:
                direction = adjusted[robot_idx] - center
                norm = float(np.linalg.norm(direction))
            if norm < 1e-6:
                direction = np.array([1.0, 0.0], dtype=np.float32)
                norm = 1.0

            adjusted[robot_idx] += (direction / norm) * 3.0
            adjusted[robot_idx] = clamp_positions(adjusted[robot_idx], env.world_low, env.world_high)

    return adjusted


def separate_robot_bodies(
    env,
    positions: np.ndarray,
    headings: np.ndarray,
    world_low: np.ndarray,
    world_high: np.ndarray,
) -> np.ndarray:
    adjusted = positions.copy()

    for _ in range(10):
        moved = False
        for i in range(len(adjusted)):
            for j in range(i + 1, len(adjusted)):
                body_i = oriented_robot_polygon(adjusted[i], float(headings[i]), env.robot_size)
                body_j = oriented_robot_polygon(adjusted[j], float(headings[j]), env.robot_size)
                if not body_i.intersects(body_j):
                    continue

                moved = True
                delta = adjusted[i] - adjusted[j]
                dist = float(np.linalg.norm(delta))
                if dist < 1e-6:
                    direction = np.array([1.0, 0.0], dtype=np.float32)
                else:
                    direction = delta / dist

                overlap_extent = max(body_i.intersection(body_j).area ** 0.5, 1.0) + PAIR_CLEARANCE_MARGIN * 0.15
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
    obstacles: list[shapely.Polygon],
    previous_positions: np.ndarray | None = None,
) -> np.ndarray:
    adjusted = clamp_positions(candidate_positions, env.world_low, env.world_high)
    adjusted = push_out_of_obstacles(env, adjusted, obstacles, margin=CLEARANCE_MARGIN)
    headings = heading_from_positions(adjusted, previous_positions)
    adjusted = separate_robot_bodies(env, adjusted, headings, env.world_low, env.world_high)
    adjusted = push_out_of_obstacles(env, adjusted, obstacles, margin=CLEARANCE_MARGIN)
    return adjusted


def normalize_relative_angles(angles: list[float], reference: float) -> list[float]:
    normalized = []
    for angle in angles:
        delta = angle - reference
        while delta <= -math.pi:
            delta += 2.0 * math.pi
        while delta > math.pi:
            delta -= 2.0 * math.pi
        normalized.append(delta)
    return normalized


def shadow_polygon(
    origin: tuple[float, float],
    obstacle: shapely.Polygon,
    far_distance: float,
) -> shapely.Polygon | None:
    vertices = [tuple(coord) for coord in obstacle.exterior.coords[:-1]]
    if not vertices:
        return None

    centroid = obstacle.centroid
    reference = math.atan2(centroid.y - origin[1], centroid.x - origin[0])
    angles = [math.atan2(v[1] - origin[1], v[0] - origin[0]) for v in vertices]
    rel_angles = normalize_relative_angles(angles, reference)

    min_idx = int(np.argmin(rel_angles))
    max_idx = int(np.argmax(rel_angles))
    min_angle = reference + rel_angles[min_idx]
    max_angle = reference + rel_angles[max_idx]
    min_vertex = vertices[min_idx]
    max_vertex = vertices[max_idx]

    far_min = (
        origin[0] + far_distance * math.cos(min_angle),
        origin[1] + far_distance * math.sin(min_angle),
    )
    far_max = (
        origin[0] + far_distance * math.cos(max_angle),
        origin[1] + far_distance * math.sin(max_angle),
    )

    polygon = shapely.Polygon([min_vertex, far_min, far_max, max_vertex])
    if polygon.is_empty or polygon.area == 0:
        return None
    return polygon


def occluded_sensor_polygon(
    sensor_polygon: shapely.Polygon,
    sensor_origin: tuple[float, float],
    obstacles: list[shapely.Polygon],
    far_distance: float,
) -> shapely.Polygon:
    visible = sensor_polygon

    for obstacle in obstacles:
        if visible.is_empty:
            break
        if not visible.intersects(obstacle):
            continue

        shadow = shadow_polygon(sensor_origin, obstacle, far_distance)
        subtractors = [obstacle]
        if shadow is not None:
            subtractors.append(shadow)
        visible = visible.difference(ops.unary_union(subtractors))

    return visible


def visible_polygons_for_team(
    robots: list[list],
    obstacles: list[shapely.Polygon],
    far_distance: float,
) -> list[list[shapely.Polygon]]:
    visible: list[list[shapely.Polygon]] = []
    for robot_sensors in robots:
        robot_visible: list[shapely.Polygon] = []
        for sensor in robot_sensors:
            robot_visible.append(
                occluded_sensor_polygon(
                    sensor_polygon=sensor.get_polygon(),
                    sensor_origin=sensor.position,
                    obstacles=obstacles,
                    far_distance=far_distance,
                )
            )
        visible.append(robot_visible)
    return visible


def visible_pair(sensor, target_point: tuple[float, float], obstacle_union) -> bool:
    polygon = sensor.get_polygon()
    point = Point(target_point)
    if not polygon.covers(point):
        return False
    line = shapely.LineString([sensor.position, target_point])
    return not line.crosses(obstacle_union) and not line.within(obstacle_union)


def metrics_with_obstacles(env, robots, visible_polygons, obstacles) -> dict[str, float]:
    obstacle_union = ops.unary_union(obstacles) if obstacles else shapely.GeometryCollection()
    outward_polygons: list[shapely.Polygon] = []
    overlaps: list[float] = []

    for robot_idx, robot_sensors in enumerate(robots):
        for sensor_idx, _ in enumerate(robot_sensors):
            if env.outward_sensor_mask[robot_idx, sensor_idx]:
                polygon = visible_polygons[robot_idx][sensor_idx].intersection(env.workspace_polygon)
                if not polygon.is_empty:
                    outward_polygons.append(polygon)

    if outward_polygons:
        union_area = ops.unary_union(outward_polygons).area / max(env.workspace_area, 1e-6)
    else:
        union_area = 0.0

    for idx, polygon in enumerate(outward_polygons):
        for other in outward_polygons[idx + 1 :]:
            union = polygon.union(other).area
            if union > 0:
                overlaps.append(polygon.intersection(other).area / union)
    overlap_penalty = float(sum(overlaps) / len(overlaps)) if overlaps else 0.0

    visible_pairs = 0
    total_pairs = env.robot_count * max(1, env.robot_count - 1)
    for i in range(env.robot_count):
        inward_indices = [
            sensor_idx
            for sensor_idx in range(env.sensor_count)
            if not env.outward_sensor_mask[i, sensor_idx]
        ]
        for j in range(env.robot_count):
            if i == j:
                continue
            target = (float(env.robot_positions[j, 0]), float(env.robot_positions[j, 1]))
            if any(visible_pair(robots[i][sensor_idx], target, obstacle_union) for sensor_idx in inward_indices):
                visible_pairs += 1

    teammate_visibility = visible_pairs / total_pairs if total_pairs else 0.0
    outward_offsets = env.sensor_offsets[env.outward_sensor_mask]
    outward_limits = np.broadcast_to(
        env.rotation_limits[np.newaxis, :],
        env.sensor_offsets.shape,
    )[env.outward_sensor_mask]
    if outward_offsets.size == 0:
        steering_penalty = 0.0
    else:
        steering_penalty = float(
            np.mean(np.abs(outward_offsets) / np.maximum(outward_limits, 1e-6))
        )
    reward = 2.4 * union_area + 1.8 * teammate_visibility - 0.35 * overlap_penalty - 0.10 * steering_penalty

    return {
        "outward_coverage": float(union_area),
        "teammate_visibility": float(teammate_visibility),
        "overlap_penalty": float(overlap_penalty),
        "steering_penalty": float(steering_penalty),
        "reward": float(reward),
    }


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
    forbidden_region = trajectory_forbidden_region(env, trajectory, args.frames)
    obstacles = generate_obstacles(
        world_low=env.world_low,
        world_high=env.world_high,
        count=args.obstacles,
        seed=args.seed + 101,
        forbidden_region=forbidden_region,
    )
    far_distance = float(np.linalg.norm(env.world_high - env.world_low)) * 1.5

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
    frozen_positions = safe_positions(env, trajectory.positions_at(0), obstacles)

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
                obstacles,
                previous_positions=frozen_positions if motion_frame == 0 else previous_positions,
            )
        env.outward_sensor_mask = env._compute_outward_sensor_mask()

        if baseline_phase:
            env.sensor_offsets.fill(0.0)
        else:
            observation = env._get_observation()
            action, _ = model.predict(observation, deterministic=True)
            _, _, _, _, _ = env.step(action)
        apply_visual_rotation_limit(env)
        robot_headings = heading_from_positions(env.robot_positions, previous_positions)
        bodies, global_fovs, local_fovs, local_origins = build_visual_sensor_geometry(
            env, env.robot_positions, robot_headings
        )
        if initial_local_fovs is None:
            initial_local_fovs = local_fovs

        if previous_offsets is None:
            offset_change = np.full_like(env.sensor_offsets, np.inf, dtype=np.float32)
        else:
            offset_change = np.abs(env.sensor_offsets - previous_offsets)
        previous_offsets = env.sensor_offsets.copy()

        visible_polygons = []
        for robot_idx in range(env.robot_count):
            robot_visible = []
            for sensor_idx in range(env.sensor_count):
                robot_visible.append(
                    occluded_sensor_polygon(
                        local_fovs[robot_idx][sensor_idx],
                        local_origins[robot_idx][sensor_idx],
                        obstacles,
                        far_distance,
                    )
                )
            visible_polygons.append(robot_visible)

        robots = env._build_team_sensors()
        metrics = metrics_with_obstacles(env, robots, visible_polygons, obstacles)
        saved_offsets = env.sensor_offsets.copy()
        env.sensor_offsets.fill(0.0)
        fixed_robot_headings = heading_from_positions(env.robot_positions, previous_positions)
        _, _, fixed_local_fovs, fixed_local_origins = build_visual_sensor_geometry(
            env, env.robot_positions, fixed_robot_headings
        )
        fixed_visible_polygons = []
        for robot_idx in range(env.robot_count):
            robot_visible = []
            for sensor_idx in range(env.sensor_count):
                robot_visible.append(
                    occluded_sensor_polygon(
                        fixed_local_fovs[robot_idx][sensor_idx],
                        fixed_local_origins[robot_idx][sensor_idx],
                        obstacles,
                        far_distance,
                    )
                )
            fixed_visible_polygons.append(robot_visible)
        fixed_robots = env._build_team_sensors()
        fixed_metrics = metrics_with_obstacles(env, fixed_robots, fixed_visible_polygons, obstacles)
        env.sensor_offsets[:] = saved_offsets
        team_center = np.mean(env.robot_positions, axis=0)
        previous_positions = env.robot_positions.copy()

        ax.clear()
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.15)
        ax.set_xlim(env.world_low[0], env.world_high[0])
        ax.set_ylim(env.world_low[1], env.world_high[1])
        ax.set_facecolor("#f5f4ef")

        plot_polygon(
            env.workspace_polygon,
            ax=ax,
            add_points=False,
            color=(0.72, 0.71, 0.68, 0.20),
        )

        for obstacle in obstacles:
            plot_polygon(
                obstacle,
                ax=ax,
                add_points=False,
                color=(0.48, 0.22, 0.12, 0.95),
            )

        for robot_idx in range(args.robots):
            trails[robot_idx].append(env.robot_positions[robot_idx].copy())
            trail = np.asarray(trails[robot_idx][-30:])
            if len(trail) > 1:
                ax.plot(trail[:, 0], trail[:, 1], color=robot_colors[robot_idx], alpha=0.45, linewidth=1.5)

        for robot_idx in range(args.robots):
            plot_polygon(bodies[robot_idx], ax=ax, add_points=False, color=(0.05, 0.05, 0.05, 0.82))

            for sensor_idx in range(env.sensor_count):
                raw_polygon = global_fovs[robot_idx][sensor_idx]
                local_polygon = local_fovs[robot_idx][sensor_idx]
                visible_polygon = visible_polygons[robot_idx][sensor_idx]
                outward = env.outward_sensor_mask[robot_idx, sensor_idx]
                is_changing = offset_change[robot_idx, sensor_idx] > ROTATION_CHANGE_THRESHOLD

                if outward:
                    base_rgb = robot_colors[robot_idx][:3]
                else:
                    base_rgb = (0.26, 0.47, 0.86)

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

                global_alpha = 0.05
                local_alpha = 0.60

                plot_polygon(raw_polygon, ax=ax, add_points=False, color=(*base_rgb, global_alpha))
                plot_polygon(local_polygon, ax=ax, add_points=False, color=(*base_rgb, 0.12), alpha=0.12)
                if not visible_polygon.is_empty:
                    plot_polygon(
                        visible_polygon,
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

        ax.scatter(team_center[0], team_center[1], marker="x", s=80, color="black", linewidths=1.5)
        if baseline_phase:
            ax.set_title(
                "Initial Local FOVs With Obstacles | frame {:03d} | baseline view before adaptive steering".format(
                    frame_idx
                )
            )
        else:
            ax.set_title(
                "Obstacle-Aware Adaptivity | frame {:03d} | reward {:.2f} | union {:.3f} | vis {:.3f} | overlap {:.3f}".format(
                    frame_idx,
                    metrics["reward"],
                    metrics["outward_coverage"],
                    metrics["teammate_visibility"],
                    metrics["overlap_penalty"],
                )
            )
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        metrics_rows.append(
            {
                "frame": frame_idx,
                "reward": metrics["reward"],
                "outward_coverage": metrics["outward_coverage"],
                "fixed_outward_coverage": fixed_metrics["outward_coverage"],
                "teammate_visibility": metrics["teammate_visibility"],
                "overlap_penalty": metrics["overlap_penalty"],
                "steering_penalty": metrics["steering_penalty"],
            }
        )

        frames = [row["frame"] for row in metrics_rows]
        coverage = [row["outward_coverage"] for row in metrics_rows]
        fixed_coverage = [row["fixed_outward_coverage"] for row in metrics_rows]
        overlap = [row["overlap_penalty"] for row in metrics_rows]

        metrics_ax.clear()
        metrics_ax.set_facecolor("#fbfaf6")
        metrics_ax.grid(True, alpha=0.2)
        metrics_ax.plot(frames, coverage, color="#1f7a8c", linewidth=2.4, label="Union coverage")
        metrics_ax.plot(
            frames,
            fixed_coverage,
            color="#5c677d",
            linewidth=1.7,
            linestyle="--",
            label="Fixed coverage",
        )
        metrics_ax.plot(frames, overlap, color="#d1495b", linewidth=1.6, label="Overlap")
        metrics_ax.scatter(frames[-1], coverage[-1], color="#1f7a8c", s=30, zorder=5)
        metrics_ax.scatter(frames[-1], fixed_coverage[-1], color="#5c677d", s=22, zorder=5)
        metrics_ax.set_xlim(0, max(args.frames - 1, 1))
        metrics_ax.set_ylim(0.0, 1.05)
        metrics_ax.set_title("Dynamic Coverage Trace")
        metrics_ax.set_xlabel("Frame")
        metrics_ax.set_ylabel("Metric value")
        metrics_ax.legend(loc="upper right", frameon=True)

    anim = animation.FuncAnimation(fig, draw_frame, frames=args.frames, interval=1000 / args.fps, repeat=False)
    writer = animation.PillowWriter(fps=args.fps)
    anim.save(output_path, writer=writer)

    metrics_path = output_path.with_suffix(".csv")
    save_metrics_csv(metrics_path, metrics_rows)

    if args.show:
        plt.show()
    else:
        plt.close(fig)

    env.close()
    print(f"Saved obstacle-aware animation to {output_path}")
    print(f"Saved per-frame metrics to {metrics_path}")


if __name__ == "__main__":
    main()
