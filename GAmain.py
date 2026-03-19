import math
import copy
import pickle

from genetic_algorithm.evolution import start_evolution
from objects.Objects import Sensor

DRONE_SIZE = (25, 25)


def main():
    # Create all needed sensors with their parameters
    sensors = [
        Sensor((0, 0), 120, math.pi / 3),
        Sensor((0, 0), 120, math.pi / 3),
        Sensor((0, 0), 120, math.pi / 3),
    ]

    # Create all zones which sensors needs to cover
    view_zones = []

    # Define population size
    population_size = 100

    # Create initial population by deepcopying
    population = [copy.deepcopy(sensors) for _ in range(population_size)]

    # Start the evolution
    population = start_evolution(
        DRONE_SIZE,
        population,
        view_zones,
        population_size,
        5,
        front_gif=True,
        sensors_gif=True,
        xlabel="angel density",
        ylabel="overlapping",
    )

    chosen_one = population[0]

    keep_data = {"sensors": chosen_one, "robot_size": DRONE_SIZE}

    with open("saved_configuration.pkl", "wb") as fd:
        pickle.dump(keep_data, fd)


if __name__ == "__main__":
    main()
