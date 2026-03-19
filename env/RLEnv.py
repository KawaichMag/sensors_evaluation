from copy import deepcopy
from dataclasses import dataclass

from gymnasium import spaces
from matplotlib import pyplot as plt
import numpy as np
import math
import gymnasium as gym
import shapely
from shapely import Point, ops
from shapely.plotting import plot_polygon

from objects.Objects import Sensor
from utils.files_management import load_configuration
from utils.parsing import parse_sensor_parameter


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


@dataclass
class RewardBreakdown:
    outward_coverage: float
    teammate_visibility: float
    overlap_penalty: float
    steering_penalty: float

    @property
    def total(self) -> float:
        return (
            2.4 * self.outward_coverage
            + 1.8 * self.teammate_visibility
            - 0.35 * self.overlap_penalty
            - 0.10 * self.steering_penalty
        )


def generate_robot_positions(
    gen_radius: float,
    n_robots: int,
    min_distance: float,
    max_distance: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if n_robots < 2:
        return rng.uniform(0.0, gen_radius, size=(n_robots, 2))

    for _ in range(10_000):
        sampled = rng.uniform(0.0, gen_radius, size=(n_robots, 2))
        valid = True

        for i in range(n_robots):
            distances = [
                float(np.linalg.norm(sampled[i] - sampled[j]))
                for j in range(n_robots)
                if i != j
            ]
            if min(distances) <= min_distance or max(distances) >= max_distance:
                valid = False
                break

        if valid:
            return sampled

    raise RuntimeError("Failed to sample robot positions that satisfy distance limits.")


def clamp(value: np.ndarray, low: float, high: float) -> np.ndarray:
    return np.minimum(np.maximum(value, low), high)


class CooperativeActiveSensingEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 5}

    def __init__(
        self,
        config_path: str = DEFAULT_CONFIG_PATH,
        robot_count: int = DEFAULT_ROBOT_COUNT,
        min_robot_distance: float = DEFAULT_MIN_DISTANCE,
        max_robot_distance: float = DEFAULT_MAX_DISTANCE,
        generation_radius: float = DEFAULT_GENERATION_RADIUS,
        world_padding: float = DEFAULT_WORLD_PADDING,
        max_steps: int = DEFAULT_MAX_STEPS,
        rotation_limit: float
        | list[float]
        | tuple[float, ...]
        | np.ndarray = DEFAULT_ROTATION_LIMIT,
        rotation_step: float
        | list[float]
        | tuple[float, ...]
        | np.ndarray = DEFAULT_ROTATION_STEP,
        seed: int | None = None,
        render_mode: str | None = None,
    ):
        super().__init__()

        self.base_sensors, self.robot_size = load_configuration(config_path)
        self.robot_count = robot_count
        self.sensor_count = len(self.base_sensors)
        self.total_sensors = self.robot_count * self.sensor_count
        self.min_robot_distance = min_robot_distance
        self.max_robot_distance = max_robot_distance
        self.generation_radius = generation_radius
        self.world_padding = world_padding
        self.max_steps = max_steps
        self.render_mode = render_mode

        max_extent = self.generation_radius + self.world_padding
        self.world_low = np.array(
            [-self.world_padding, -self.world_padding], dtype=float
        )
        self.world_high = np.array([max_extent, max_extent], dtype=float)
        self.workspace_polygon = shapely.Polygon(
            [
                (self.world_low[0], self.world_high[1]),
                (self.world_high[0], self.world_high[1]),
                (self.world_high[0], self.world_low[1]),
                (self.world_low[0], self.world_low[1]),
            ]
        )
        self.workspace_area = float(self.workspace_polygon.area)

        self.base_positions = np.array(
            [
                np.asarray(sensor.position, dtype=np.float32)
                for sensor in self.base_sensors
            ],
            dtype=np.float32,
        )
        self.base_rotations = np.array(
            [float(sensor.rotation) for sensor in self.base_sensors],
            dtype=np.float32,
        )
        self.sensor_ranges = np.array(
            [float(sensor.distance) for sensor in self.base_sensors],
            dtype=np.float32,
        )
        self.sensor_fov = np.array(
            [float(sensor.angle) for sensor in self.base_sensors],
            dtype=np.float32,
        )
        self.rotation_limits = parse_sensor_parameter(
            rotation_limit,
            self.sensor_count,
            "rotation_limit",
        )
        self.rotation_steps = parse_sensor_parameter(
            rotation_step,
            self.sensor_count,
            "rotation_step",
        )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.total_sensors,),
            dtype=np.float32,
        )

        observation_size = (
            self.robot_count * 2
            + self.total_sensors
            + self.total_sensors
            + self.total_sensors * 2
            + 2
        )
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(observation_size,),
            dtype=np.float32,
        )

        self._rng = np.random.default_rng(seed)
        self.robot_positions = np.zeros((self.robot_count, 2), dtype=np.float32)
        self.sensor_offsets = np.zeros(
            (self.robot_count, self.sensor_count), dtype=np.float32
        )
        self.current_step = 0
        self.outward_sensor_mask = np.zeros(
            (self.robot_count, self.sensor_count), dtype=bool
        )
        self.last_reward = RewardBreakdown(0.0, 0.0, 0.0, 0.0)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.robot_positions = generate_robot_positions(
            gen_radius=self.generation_radius,
            n_robots=self.robot_count,
            min_distance=self.min_robot_distance,
            max_distance=self.max_robot_distance,
            rng=self._rng,
        ).astype(np.float32)
        self.sensor_offsets.fill(0.0)
        self.outward_sensor_mask = self._compute_outward_sensor_mask()
        self.current_step = 0

        observation = self._get_observation()
        info = self._get_info()
        return observation, info

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32).reshape(
            self.robot_count, self.sensor_count
        )
        action = clamp(action, -1.0, 1.0)
        action = action * self.outward_sensor_mask.astype(np.float32)

        self.sensor_offsets += action * self.rotation_steps[np.newaxis, :]
        self.sensor_offsets = np.clip(
            self.sensor_offsets,
            -self.rotation_limits[np.newaxis, :],
            self.rotation_limits[np.newaxis, :],
        )

        self.current_step += 1
        reward_breakdown = self._calculate_reward()
        self.last_reward = reward_breakdown

        observation = self._get_observation()
        terminated = False
        truncated = self.current_step >= self.max_steps
        info = self._get_info()
        return observation, reward_breakdown.total, terminated, truncated, info

    def render(self):
        robots = self._build_team_sensors()
        _, ax = plt.subplots()
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.2)
        plot_polygon(
            self.workspace_polygon,
            ax=ax,
            add_points=False,
            color=(0.6, 0.6, 0.6, 0.15),
        )

        colors = plt.cm.tab10(np.linspace(0.0, 1.0, self.robot_count))  # type: ignore

        for i, robot_sensors in enumerate(robots):
            body = self._robot_polygon(self.robot_positions[i])
            plot_polygon(body, ax=ax, add_points=False, color=(0.0, 0.0, 0.0, 0.75))

            for sensor in robot_sensors:
                plot_polygon(
                    sensor.get_polygon(),
                    ax=ax,
                    add_points=False,
                    color=colors[i],
                    alpha=0.35,
                )

            ax.text(
                self.robot_positions[i, 0],
                self.robot_positions[i, 1],
                f"R{i}",
                ha="center",
                va="center",
            )

        ax.set_title(
            "Union coverage {:.2f} | Teammate visibility {:.2f} | Overlap {:.2f}".format(
                self.last_reward.outward_coverage,
                self.last_reward.teammate_visibility,
                self.last_reward.overlap_penalty,
            )
        )
        plt.show()

    def close(self):
        plt.close("all")

    def _robot_polygon(self, position: np.ndarray) -> shapely.Polygon:
        half_w = self.robot_size[0] / 2.0
        half_h = self.robot_size[1] / 2.0
        return shapely.Polygon(
            [
                (position[0] - half_w, position[1] + half_h),
                (position[0] + half_w, position[1] + half_h),
                (position[0] + half_w, position[1] - half_h),
                (position[0] - half_w, position[1] - half_h),
            ]
        )

    def _build_team_sensors(self) -> list[list[Sensor]]:
        robots: list[list[Sensor]] = []

        for robot_idx in range(self.robot_count):
            robot_sensors = deepcopy(self.base_sensors)
            translation = self.robot_positions[robot_idx]

            for sensor_idx, sensor in enumerate(robot_sensors):
                base_position = self.base_positions[sensor_idx]
                sensor.position = (
                    float(base_position[0] + translation[0]),
                    float(base_position[1] + translation[1]),
                )
                sensor.set_rotation(
                    float(
                        self.base_rotations[sensor_idx]
                        + self.sensor_offsets[robot_idx, sensor_idx]
                    )
                )

            robots.append(robot_sensors)

        return robots

    def _compute_outward_sensor_mask(self) -> np.ndarray:
        team_center = np.mean(self.robot_positions, axis=0)
        mask = np.zeros((self.robot_count, self.sensor_count), dtype=bool)

        for robot_idx in range(self.robot_count):
            translation = self.robot_positions[robot_idx]

            for sensor_idx in range(self.sensor_count):
                sensor_position = self.base_positions[sensor_idx] + translation
                outward_vector = sensor_position - team_center

                if np.linalg.norm(outward_vector) < 1e-6:
                    outward_vector = translation - team_center

                if np.linalg.norm(outward_vector) < 1e-6:
                    outward_vector = np.array([1.0, 0.0], dtype=np.float32)

                outward_direction = outward_vector / max(
                    float(np.linalg.norm(outward_vector)), 1e-6
                )
                base_rotation = self.base_rotations[sensor_idx]
                heading = np.array(
                    [math.cos(base_rotation), math.sin(base_rotation)],
                    dtype=np.float32,
                )

                mask[robot_idx, sensor_idx] = (
                    float(np.dot(heading, outward_direction)) >= 0.0
                )

        return mask

    def _flatten_sensors(self, robots: list[list[Sensor]]) -> list[Sensor]:
        flat: list[Sensor] = []
        for robot_sensors in robots:
            flat.extend(robot_sensors)
        return flat

    def _calculate_reward(self) -> RewardBreakdown:
        robots = self._build_team_sensors()
        outward_sensors, inward_sensors = self._split_team_sensors(robots)
        outward_coverage = self._coverage_ratio(outward_sensors)
        teammate_visibility = self._visibility_ratio(inward_sensors, robots)
        overlap_penalty = (
            self._mean_overlap(outward_sensors) if outward_sensors else 0.0
        )

        outward_offsets = self.sensor_offsets[self.outward_sensor_mask]
        outward_limits = np.broadcast_to(
            self.rotation_limits[np.newaxis, :],
            self.sensor_offsets.shape,
        )[self.outward_sensor_mask]
        if outward_offsets.size == 0:
            steering_penalty = 0.0
        else:
            steering_penalty = float(
                np.mean(np.abs(outward_offsets) / np.maximum(outward_limits, 1e-6))
            )

        return RewardBreakdown(
            outward_coverage=outward_coverage,
            teammate_visibility=teammate_visibility,
            overlap_penalty=overlap_penalty,
            steering_penalty=steering_penalty,
        )

    def _split_team_sensors(
        self, robots: list[list[Sensor]]
    ) -> tuple[list[Sensor], list[Sensor]]:
        outward_sensors: list[Sensor] = []
        inward_sensors: list[Sensor] = []

        for robot_idx, robot_sensors in enumerate(robots):
            for sensor_idx, sensor in enumerate(robot_sensors):
                if self.outward_sensor_mask[robot_idx, sensor_idx]:
                    outward_sensors.append(sensor)
                else:
                    inward_sensors.append(sensor)

        return outward_sensors, inward_sensors

    def _coverage_ratio(self, sensors: list[Sensor]) -> float:
        if not sensors:
            return 0.0

        polygons = [
            sensor.get_polygon().intersection(self.workspace_polygon)
            for sensor in sensors
        ]
        polygons = [polygon for polygon in polygons if not polygon.is_empty]
        if not polygons:
            return 0.0

        covered_area = ops.unary_union(polygons).area
        return float(covered_area / max(self.workspace_area, 1e-6))

    def _visibility_ratio(
        self, inward_sensors: list[Sensor], robots: list[list[Sensor]]
    ) -> float:
        if not inward_sensors:
            return 0.0

        visible_pairs = 0
        total_pairs = self.robot_count * max(1, self.robot_count - 1)

        for i in range(self.robot_count):
            robot_inward_sensors = [
                sensor
                for sensor_idx, sensor in enumerate(robots[i])
                if not self.outward_sensor_mask[i, sensor_idx]
            ]
            for j in range(self.robot_count):
                if i == j:
                    continue

                target_point = Point(
                    float(self.robot_positions[j, 0]),
                    float(self.robot_positions[j, 1]),
                )
                if any(
                    sensor.get_polygon().covers(target_point)
                    for sensor in robot_inward_sensors
                ):
                    visible_pairs += 1

        return visible_pairs / total_pairs

    def _mean_overlap(self, sensors: list[Sensor]) -> float:
        overlaps: list[float] = []

        for i, sensor in enumerate(sensors):
            polygon = sensor.get_polygon()
            for other_sensor in sensors[i + 1 :]:
                other_polygon = other_sensor.get_polygon()
                union_area = polygon.union(other_polygon).area
                if union_area == 0:
                    continue
                overlaps.append(polygon.intersection(other_polygon).area / union_area)

        if not overlaps:
            return 0.0

        return float(sum(overlaps) / len(overlaps))

    def _get_observation(self) -> np.ndarray:
        robot_positions = self._normalize_positions(self.robot_positions).reshape(-1)
        offsets = (
            self.sensor_offsets / np.maximum(self.rotation_limits[np.newaxis, :], 1e-6)
        ).reshape(-1)
        outward_mask = self.outward_sensor_mask.astype(np.float32).reshape(-1)
        team_center = self._normalize_positions(
            np.mean(self.robot_positions, axis=0, keepdims=True)
        ).reshape(-1)

        rotations = (self.base_rotations[np.newaxis, :] + self.sensor_offsets).reshape(
            -1
        )
        rotation_features = np.stack(
            (np.sin(rotations), np.cos(rotations)), axis=1
        ).reshape(-1)

        observation = np.concatenate(
            [robot_positions, offsets, outward_mask, rotation_features, team_center]
        ).astype(np.float32)

        return observation

    def _normalize_positions(self, positions: np.ndarray) -> np.ndarray:
        center = (self.world_low + self.world_high) / 2.0
        scale = np.maximum((self.world_high - self.world_low) / 2.0, 1e-6)
        return (positions - center) / scale

    def _get_info(self) -> dict[str, float]:
        reward_breakdown = self._calculate_reward()
        self.last_reward = reward_breakdown
        return {
            "outward_coverage": reward_breakdown.outward_coverage,
            "teammate_visibility": reward_breakdown.teammate_visibility,
            "overlap_penalty": reward_breakdown.overlap_penalty,
            "steering_penalty": reward_breakdown.steering_penalty,
            "outward_sensor_ratio": float(np.mean(self.outward_sensor_mask)),
            "workspace_area": self.workspace_area,
            "mean_rotation_limit_deg": float(np.mean(np.degrees(self.rotation_limits))),
            "mean_rotation_step_deg": float(np.mean(np.degrees(self.rotation_steps))),
            "reward": reward_breakdown.total,
            "step": float(self.current_step),
        }


def make_env(
    config_path: str,
    robot_count: int,
    max_steps: int,
    rotation_limit: float
    | list[float]
    | tuple[float, ...]
    | np.ndarray = DEFAULT_ROTATION_LIMIT,
    rotation_step: float
    | list[float]
    | tuple[float, ...]
    | np.ndarray = DEFAULT_ROTATION_STEP,
    seed: int | None = None,
    render_mode: str | None = None,
) -> CooperativeActiveSensingEnv:
    return CooperativeActiveSensingEnv(
        config_path=config_path,
        robot_count=robot_count,
        max_steps=max_steps,
        rotation_limit=rotation_limit,
        rotation_step=rotation_step,
        seed=seed,
        render_mode=render_mode,
    )
