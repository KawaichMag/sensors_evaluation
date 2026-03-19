import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import shapely
from shapely.plotting import plot_polygon

from cooperative_rl_union import DEFAULT_CONFIG_PATH, load_configuration
from cooperative_rl_union_visualization import (
    GLOBAL_FOV_MARGIN,
    VISUAL_ROTATION_LIMIT,
    fov_sector,
    local_fov_triangle,
    oriented_robot_polygon,
    rotation_matrix,
)


DEFAULT_OUTPUT = "experiments/cooperative_rl_union/debug_fov.png"
VISUAL_RANGE_SCALE = 0.42
VISUAL_MAX_RANGE = 85.0


def visual_sensor_range(sensor_range: float) -> float:
    return min(sensor_range * VISUAL_RANGE_SCALE, VISUAL_MAX_RANGE)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Static debug view for robot bodies and global/local sensor FOV geometry."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to saved configuration pickle.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output image path.")
    parser.add_argument("--sensor", type=int, default=0, help="Sensor index to highlight.")
    parser.add_argument("--local-fov-deg", type=float, default=60.0, help="Debug local FOV angle in degrees.")
    parser.add_argument(
        "--use-config-fov",
        action="store_true",
        help="Use the sensor angle from the saved configuration instead of the debug override.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sensors, robot_size = load_configuration(args.config)
    sensor = sensors[args.sensor]

    robot_position = np.array([55.0, 35.0], dtype=np.float32)
    robot_heading = math.radians(25.0)
    sensor_offset = math.radians(10.0)
    visual_rotation_limit = VISUAL_ROTATION_LIMIT

    local_fov_angle = (
        float(sensor.angle)
        if args.use_config_fov
        else math.radians(args.local_fov_deg)
    )
    global_fov_angle = min(
        local_fov_angle + 2.0 * GLOBAL_FOV_MARGIN,
        local_fov_angle + 2.0 * visual_rotation_limit,
    )

    sensor_origin = robot_position + rotation_matrix(robot_heading) @ np.asarray(sensor.position, dtype=np.float32)
    sensor_origin_tuple = (float(sensor_origin[0]), float(sensor_origin[1]))
    center_heading = float(sensor.rotation + robot_heading + sensor_offset)
    radius = visual_sensor_range(float(sensor.distance))

    body = oriented_robot_polygon(robot_position, robot_heading, robot_size)
    global_fov = fov_sector(
        sensor_origin_tuple,
        radius,
        center_heading - global_fov_angle / 2.0,
        center_heading + global_fov_angle / 2.0,
    )
    local_fov = local_fov_triangle(
        sensor_origin_tuple,
        radius,
        center_heading,
        local_fov_angle,
    )
    workspace_polygon = shapely.box(-25.0, -25.0, 155.0, 125.0)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.2)
    ax.set_xlim(-25.0, 155.0)
    ax.set_ylim(-25.0, 125.0)
    ax.set_facecolor("#f7f5ef")

    plot_polygon(
        workspace_polygon,
        ax=ax,
        add_points=False,
        color=(0.8, 0.79, 0.74, 0.18),
    )
    base_rgb = plt.cm.Set2(0)
    plot_polygon(body, ax=ax, add_points=False, color=(0.08, 0.08, 0.08, 0.82))
    plot_polygon(global_fov, ax=ax, add_points=False, color=(*base_rgb[:3], 0.22))
    plot_polygon(local_fov, ax=ax, add_points=False, color=(*base_rgb[:3], 0.62), alpha=0.62)
    ax.scatter(robot_position[0], robot_position[1], color=base_rgb, s=44, edgecolors="black", linewidths=0.6, zorder=6)
    ax.text(robot_position[0] + 3.0, robot_position[1] + 3.0, "R0", fontsize=10, weight="bold")
    ax.scatter(sensor_origin_tuple[0], sensor_origin_tuple[1], color="black", s=22, zorder=7)
    ax.plot(
        [sensor_origin_tuple[0], sensor_origin_tuple[0] + radius * math.cos(center_heading)],
        [sensor_origin_tuple[1], sensor_origin_tuple[1] + radius * math.sin(center_heading)],
        color=(*base_rgb[:3], 0.9),
        linewidth=1.4,
    )
    ax.text(
        sensor_origin_tuple[0] + 2.0,
        sensor_origin_tuple[1] - 4.0,
        (
            f"S{args.sensor}\n"
            f"local={math.degrees(local_fov_angle):.1f} deg\n"
            f"global={math.degrees(global_fov_angle):.1f} deg\n"
            f"offset={math.degrees(sensor_offset):.1f} deg"
        ),
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.92, "edgecolor": "#444"},
    )
    ax.set_title("Debug FOV View | one robot, one sensor")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved debug visualization to {output_path}")


if __name__ == "__main__":
    main()
