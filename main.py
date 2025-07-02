import math
import copy

from genetic_algorithm.evolution import start_evolution
from objects.Objects import Sensor


def main():
    # Create all needed sensors with their parameters
    sensors = [
        Sensor((0, 0), 30, math.pi / 4),
        Sensor((0, 0), 30, math.pi / 4),
        Sensor((0, 0), 30, math.pi / 4),
    ]

    # Create all zones which sensors needs to cover
    view_zones = []

    # Define population size
    population_size = 100

    # Create initial population by deepcopying
    population = [copy.deepcopy(sensors) for _ in range(population_size)]

    # Start the evolution
    population = start_evolution(
        (10, 10),
        population,
        view_zones,
        population_size,
        5,
        front_gif=True,
        sensors_gif=True,
        xlabel="angel density",
        ylabel="overlapping",
    )


if __name__ == "__main__":
    main()
