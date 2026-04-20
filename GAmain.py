import math
import copy
import pickle

from genetic_algorithm.evolution import start_evolution
from objects.Objects import Sensor

drone_size = (130, 230)
multiplier = 50


def main():
    # Create all needed sensors with their parameters
    sensors = [
        Sensor(
            (-drone_size[0] / 2, drone_size[1] / 8), 10 * multiplier, math.radians(60)
        ),
        Sensor(
            (drone_size[0] / 2, drone_size[1] / 8), 10 * multiplier, math.radians(60)
        ),
        Sensor(
            (-drone_size[0] / 2, -drone_size[1] / 8), 10 * multiplier, math.radians(60)
        ),
        Sensor(
            (drone_size[0] / 2, -drone_size[1] / 8), 10 * multiplier, math.radians(60)
        ),
        Sensor(
            (drone_size[0] / 4, drone_size[1] / 2), 18 * multiplier, math.radians(65)
        ),
        Sensor(
            (drone_size[0] / 4, -drone_size[1] / 2), 16 * multiplier, math.radians(65)
        ),
        Sensor(
            (-drone_size[0] / 4, drone_size[1] / 2), 18 * multiplier, math.radians(65)
        ),
        Sensor(
            (-drone_size[0] / 4, -drone_size[1] / 2), 16 * multiplier, math.radians(65)
        ),
    ]

    for i in [0, 2]:
        sensors[i].rotate(math.radians(180))

    for i in [4, 6]:
        sensors[i].rotate(math.radians(90))

    for i in [5, 7]:
        sensors[i].rotate(math.radians(-90))

    # Create all zones which sensors needs to cover
    view_zones = []

    # Define population size
    population_size = 100

    # Create initial population by deepcopying
    population = [copy.deepcopy(sensors) for _ in range(population_size)]

    # Start the evolution
    population = start_evolution(
        drone_size,
        population,
        view_zones,
        population_size,
        100,
        front_gif=True,
        sensors_gif=True,
        xlabel="angel density",
        ylabel="overlapping",
    )

    chosen_one = population[0]

    keep_data = {"sensors": chosen_one, "robot_size": drone_size}

    with open("saved_configuration.pkl", "wb") as fd:
        pickle.dump(keep_data, fd)


if __name__ == "__main__":
    main()
