import copy
from genetic_algorithm.evolution import start_evolution
from objects.Objects import Sensor
import math
from analysis.analyzation import write_results, process_string
import pandas as pd

multiplier = 50

drone_size = (150, 150)  # In mm
sensors = [
    Sensor((0, drone_size[1] / 3), 18 * multiplier, math.radians(65)),
    Sensor((drone_size[0] / 3, 0), 16 * multiplier, math.radians(65)),
    Sensor((0, -drone_size[1] / 3), 18 * multiplier, math.radians(65)),
    Sensor((-drone_size[0] / 3, 0), 16 * multiplier, math.radians(65)),
]

print([sensor.position for sensor in sensors])

for i in range(4):
    sensors[i].rotate(math.radians(90 * i))


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
            front_gif=False,
            sensors_gif=False,
            xlabel="angel density",
            ylabel="overlapping",
        )

        data = pd.read_csv("analysis/frontiers.csv")

        for column in data.columns[1:]:
            data[column] = data[column].apply(process_string)

        write_results(baseline, data, "abstract2", index)


if __name__ == "__main__":
    main()
