import copy
from genetic_algorithm.evolution import start_evolution
from objects.Objects import Sensor
import math
from analysis.analyzation import write_results, process_string
import pandas as pd

multiplier = 50

drone_size = (670, 810)  # In mm
sensors = [
    Sensor((-670 / 2, 810 / 4), 40 * 100, math.radians(75)),
    Sensor((670 / 2, 810 / 4), 40 * 100, math.radians(75)),
    Sensor((-670 / 2, -810 / 4), 40 * 100, math.radians(75)),
    Sensor((670 / 2, -810 / 4), 40 * 100, math.radians(75)),
    Sensor((670 / 4, 810 / 2), 40 * 100, math.radians(65)),
    Sensor((670 / 4, -810 / 2), 40 * 100, math.radians(65)),
    Sensor((-670 / 4, 810 / 2), 40 * 100, math.radians(65)),
    Sensor((-670 / 4, -810 / 2), 40 * 100, math.radians(65)),
]

for i in [0, 2]:
    sensors[i].rotate(math.radians(180))

for i in [4, 6]:
    sensors[i].rotate(math.radians(90))

for i in [5, 7]:
    sensors[i].rotate(math.radians(-90))


def main():
    # Define population size
    population_size = 300

    for index in range(1):
        print("Starting {} run for analysis".format(index))

        # Create initial population by deepcopying
        population = [copy.deepcopy(sensors) for _ in range(population_size)]

        # Start the glorious evolution
        baseline, population = start_evolution(
            drone_size,
            population,
            [],
            population_size,
            150,
            front_gif=True,
            sensors_gif=False,
            xlabel="angel density",
            ylabel="overlapping",
        )

        data = pd.read_csv("analysis/frontiers.csv")

        for column in data.columns[1:]:
            data[column] = data[column].apply(process_string)

        write_results(baseline, data, "abstract3", index)


if __name__ == "__main__":
    main()
