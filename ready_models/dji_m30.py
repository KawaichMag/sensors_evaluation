import copy
from genetic_algorithm.evolution import start_evolution
from objects.Objects import Sensor
import math

multiplier = 50

drone_size = (215, 365)  # In mm
sensors = [
    Sensor(
        (-drone_size[0] / 2, drone_size[1] / 2 * 3 / 4),
        33 * multiplier,
        math.radians(65),
    ),
    Sensor(
        (drone_size[0] / 2, drone_size[1] / 2 * 3 / 4),
        33 * multiplier,
        math.radians(65),
    ),
    Sensor(
        (-drone_size[0] / 2, drone_size[1] / 2 * 1.5 / 4),
        33 * multiplier,
        math.radians(65),
    ),
    Sensor(
        (drone_size[0] / 2, drone_size[1] / 2 * 1.5 / 4),
        33 * multiplier,
        math.radians(65),
    ),
    Sensor((drone_size[0] / 4, drone_size[1] / 2), 38 * multiplier, math.radians(65)),
    Sensor((drone_size[0] / 4, -drone_size[1] / 2), 33 * multiplier, math.radians(65)),
    Sensor((-drone_size[0] / 4, drone_size[1] / 2), 38 * multiplier, math.radians(65)),
    Sensor((-drone_size[0] / 4, -drone_size[1] / 2), 33 * multiplier, math.radians(65)),
]

for i in [0, 2]:
    sensors[i].rotate(math.radians(180))

for i in [4, 6]:
    sensors[i].rotate(math.radians(90))

for i in [5, 7]:
    sensors[i].rotate(math.radians(-90))

# draw_plan(drone_size, sensors)


def main():
    # Define population size
    population_size = 300

    # Create initial population by deepcopying
    population = [copy.deepcopy(sensors) for _ in range(population_size)]

    # Start the glorious evolution
    population = start_evolution(
        drone_size,
        population,
        [],
        population_size,
        150,
        front_gif=True,
        xlabel="angel density",
        ylabel="overlapping",
    )


if __name__ == "__main__":
    # draw_plan(drone_size, sensors)
    main()
