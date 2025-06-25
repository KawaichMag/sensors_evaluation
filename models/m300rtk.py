import copy
from Objects import Sensor, draw_plan, start_evolution
import math

drone_size = (670, 810) # In mm
sensors = [
    Sensor((-670/2, 810/4), 40*100, math.radians(75)),
    Sensor((670/2, 810/4), 40*100, math.radians(75)),
    Sensor((-670/2, -810/4), 40*100, math.radians(75)),
    Sensor((670/2, -810/4), 40*100, math.radians(75)),

    Sensor((670/4, 810/2), 40*100, math.radians(65)),
    Sensor((670/4, -810/2), 40*100, math.radians(65)),
    Sensor((-670/4, 810/2), 40*100, math.radians(65)),
    Sensor((-670/4, -810/2), 40*100, math.radians(65)),
]

for i in [0, 2]:
    sensors[i].rotate(math.radians(180))

for i in [4, 6]:
    sensors[i].rotate(math.radians(90))

for i in [5, 7]:
    sensors[i].rotate(math.radians(-90))

draw_plan(drone_size, sensors)

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
                            sensors_gif=True)

if __name__ == "__main__":
    main()