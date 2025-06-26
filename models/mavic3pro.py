import copy
from Objects import Sensor, draw_plan, start_evolution
import math

MULTIPLIER = 20

drone_size = (98, 230) # In mm

sensors = [
    Sensor((98/2, 230/2), 25*MULTIPLIER, math.radians(90)),
    Sensor((98/2, -230/2), 25*MULTIPLIER, math.radians(90)),
    Sensor((-98/2, 230/2), 25*MULTIPLIER, math.radians(90)),
    Sensor((-98/2, -230/2), 25*MULTIPLIER, math.radians(90)),

    Sensor((98/2, 230/2), 25*MULTIPLIER, math.radians(90)),
    Sensor((98/2, -230/2), 25*MULTIPLIER, math.radians(90)),
    Sensor((-98/2, 230/2), 25*MULTIPLIER, math.radians(90)),
    Sensor((-98/2, -230/2), 25*MULTIPLIER, math.radians(90))
]

for i in [2, 3]:
    sensors[i].rotate(math.radians(180))

for i in [5, 7]:
    sensors[i].rotate(math.radians(-90))

for i in [4, 6]:
    sensors[i].rotate(math.radians(90))

def main():
    # Define population size
    population_size = 100

    # Create initial population by deepcopying
    population = [copy.deepcopy(sensors) for _ in range(population_size)]

    # Start the evolution
    population = start_evolution(
                            drone_size,
                            population, 
                            [], 
                            population_size, 
                            100,
                            front_gif=True,
                            xlabel="angel density",
                            ylabel="overlapping"
                            )

if __name__ == "__main__":
    main()