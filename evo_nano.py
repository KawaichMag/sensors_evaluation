import copy
from genetic_algorithm.evolution import start_evolution
from objects.Objects import Sensor
import math

multiplier = 50

drone_size = (94, 142)  # In mm
sensors = [
    Sensor((-drone_size[0] / 4, drone_size[1] / 2), 10 * multiplier, math.radians(40)),
    Sensor((drone_size[0] / 4, drone_size[1] / 2), 10 * multiplier, math.radians(40)),
    Sensor((-drone_size[0] / 4, -drone_size[1] / 2), 10 * multiplier, math.radians(40)),
    Sensor((drone_size[0] / 4, -drone_size[1] / 2), 10 * multiplier, math.radians(40)),
]

# for i in [0, 2]:
#     sensors[i].rotate(math.radians(180))

for i in [0, 1]:
    sensors[i].rotate(math.radians(90))

for i in [2, 3]:
    sensors[i].rotate(math.radians(-90))


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
