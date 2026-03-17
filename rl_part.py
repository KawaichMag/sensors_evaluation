import pickle
from objects.Objects import Sensor

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import shapely
import matplotlib.pyplot as plt

from shapely.plotting import plot_polygon
from copy import deepcopy

ROBOTS_NUMBER = 3
ROBOTS_MIN_DISTANCE = 25
ROBOTS_MAX_DISTANCE = 100

GENERATION_RADIUS = 100


def generate_robots_positions(gen_radius, n_robots, min_distance, max_distance):
    generating = True
    counter = 0

    while generating:
        counter += 1
        sampled_robot_coordinates = np.random.randint(0, gen_radius, (n_robots, 2))
        generating = False
        for robot_pos in sampled_robot_coordinates:
            distances = [
                np.linalg.norm(robot_pos - other_robot_pos)
                for other_robot_pos in sampled_robot_coordinates
                if np.linalg.norm(robot_pos - other_robot_pos) != 0
            ]
            if not (min(distances) > min_distance and max(distances) < max_distance):
                generating = True
                break

    print("Generated for {} try!".format(counter))

    return sampled_robot_coordinates


def denormalize(a, min_val, max_val):
    return min_val + (a + 1) * 0.5 * (max_val - min_val)


# class RobotsWithSensorsEnv(gym.Env):
#     def __init__(self, n_sensors: int, sensors: list[Sensor]):
#         super.__init__()

#         self.n = n_sensors
#         self.sensors = sensors

#         self.action_space = spaces.Box(
#             low=-1,
#             high=1,
#             shape=(self.n,),
#         )

#         self.observation_space = spaces.Box(
#             low=-self.max_rotation,
#             high=self.max_rotation,
#             shape=(self.n,),
#             dtype=np.float32,
#         )


def main():
    with open("saved_configuration.pkl", "rb") as fd:
        configuration = pickle.load(fd)
        sensors: list[Sensor] = configuration["sensors"]
        robot_size: tuple[float, float] = configuration["robot_size"]
        robot_size = np.array(robot_size)

    robots_positions = np.array(
        generate_robots_positions(
            GENERATION_RADIUS, ROBOTS_NUMBER, ROBOTS_MIN_DISTANCE, ROBOTS_MAX_DISTANCE
        )
    )

    for sensor in sensors:
        sensor.clear_cache()

    robots = [deepcopy(sensors) for _ in range(ROBOTS_NUMBER)]

    for i in range(ROBOTS_NUMBER):
        color = np.random.random((3,))
        robot_body = shapely.Polygon(
            [
                robots_positions[i] + [robot_size[0] / 2, robot_size[1] / 2],
                robots_positions[i] + [robot_size[0] / 2, -robot_size[1] / 2],
                robots_positions[i] + [-robot_size[0] / 2, -robot_size[1] / 2],
                robots_positions[i] + [-robot_size[0] / 2, robot_size[1] / 2],
            ]
        )

        plot_polygon(robot_body, add_points=False, color=(0, 0, 0, 0.8))

        for sensor in robots[i]:
            sensor.position = sensor.position + robots_positions[i]
            plot_polygon(sensor.get_polygon(), add_points=False, color=color)

    plt.show()


if __name__ == "__main__":
    main()
