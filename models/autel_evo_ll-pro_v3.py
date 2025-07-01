import copy
from Objects import Sensor, draw_plan, start_evolution
import math

multiplier = 50

drone_size = (130, 230) # In mm
sensors = [
    Sensor((-drone_size[0]/2, drone_size[1]/8), 10*multiplier, math.radians(60)),
    Sensor((drone_size[0]/2, drone_size[1]/8), 10*multiplier, math.radians(60)),
    Sensor((-drone_size[0]/2, -drone_size[1]/8), 10*multiplier, math.radians(60)),
    Sensor((drone_size[0]/2, -drone_size[1]/8), 10*multiplier, math.radians(60)),

    Sensor((drone_size[0]/4, drone_size[1]/2), 18*multiplier, math.radians(65)),
    Sensor((drone_size[0]/4, -drone_size[1]/2), 16*multiplier, math.radians(65)),
    Sensor((-drone_size[0]/4, drone_size[1]/2), 18*multiplier, math.radians(65)),
    Sensor((-drone_size[0]/4, -drone_size[1]/2), 16*multiplier, math.radians(65)),
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
                            ylabel="overlapping"
                            )

if __name__ == "__main__":
    main()