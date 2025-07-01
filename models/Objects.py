import math
import random
import string
import time
import copy
import matplotlib.pyplot as plt
import shapely
from shapely.plotting import plot_polygon
import os
from PIL import Image
import numpy as np
import shutil

type Genotype = list[Sensor]
type Population = list[Genotype]

type Front = list[int]
type Fronts = list[Front]

type Fitness = tuple[float, float]
type IndexedFitness = tuple[Fitness, int]

class Sensor:
    def __init__(self, position: tuple[float, float], distance: float, angle: float):
        self.position = position
        self.distance = distance
        self.angle = angle
        self.observation = None
        self.polygon = None
        self.rotation = 0.0
        
    def get_observation(self):
        if self.observation is None:
            main_point = self.position

            corner_point1 = (self.distance * math.cos(self.rotation + self.angle/2) + main_point[0], 
                            self.distance * math.sin(self.rotation + self.angle/2) + main_point[1])
            
            corner_point2 = (self.distance * math.cos(self.rotation - self.angle/2) + main_point[0], 
                            self.distance * math.sin(self.rotation - self.angle/2) + main_point[1])
            
            self.observation = [main_point, corner_point1, corner_point2]

        return self.observation
    
    def rotate(self, angle: float):
        self.rotation += angle
        self.observation = None
        self.polygon = None

    def set_rotation(self, rotation: float):
        self.rotation = rotation
        self.observation = None
        self.polygon = None

    def move(self, dx: float, dy: float, constrains: tuple[float, float] = (0, 0)):
        if constrains == (0, 0):
            self.position = (self.position[0] + dx, self.position[1] + dy)
            self.observation = None
            self.polygon = None
        else:
            sign_x = 1 if self.position[0] + dx > 0 else -1
            sign_y = 1 if self.position[1] + dy > 0 else -1

            new_x = sign_x * min(abs(self.position[0] + dx), constrains[0])
            new_y = sign_y * min(abs(self.position[1] + dy), constrains[1])

            self.position = (new_x, new_y)

            self.observation = None
            self.polygon = None

    def get_polygon(self):
        if self.polygon is None:
            if self.observation is None:
                self.get_observation()

            self.polygon = shapely.Polygon(self.observation)

        return self.polygon
    
class ViewZone:
    def __init__(self, path: list[tuple[float, float]]):
        self.path = path
        self.polygon = None

    def get_polygon(self):
        if self.polygon is None:
            self.polygon = shapely.Polygon(self.path)
            
        return self.polygon

    def move(self, dx: float, dy: float):
        self.path = [(x + dx, y + dy) for x, y in self.path]
        self.polygon = None

    def __str__(self):
        return str(self.path)

def get_square_view_zone(center: tuple[float, float], side_length: float) -> ViewZone:
    return ViewZone([(center[0] - side_length/2, center[1] - side_length/2), (center[0] + side_length/2, center[1] - side_length/2), (center[0] + side_length/2, center[1] + side_length/2), (center[0] - side_length/2, center[1] + side_length/2)])

def get_coverage(sensors: list[Sensor], view_zones: list[ViewZone]) -> float:
        coverage_area = 0

        for sensor in sensors:
            for view_zone in view_zones:
                if sensor.get_polygon().intersects(view_zone.get_polygon()):
                    coverage_area += sensor.get_polygon().intersection(view_zone.get_polygon()).area

        return coverage_area

def draw_experiment(drone_size: tuple[float, float],
                    population: Population, 
                    view_zones: list[ViewZone], 
                    show: bool = True, 
                    save: bool = False, 
                    subfolder: str = "",
                    filename: str = "evolution",
                    primary_solution_color: str = 'green',
                    other_solutions_color: str = 'red',
                    view_zones_color: str = 'blue',
                    drone_color: str = 'black',
                    first_label: str = "None",
                    second_label: str = "None"):

    os.makedirs(subfolder, exist_ok=True)

    

    # Setup the plot
    fig, ax = plt.subplots()
    ax.grid(True)
    ax.set_aspect('equal', adjustable='box')

    # Plot the drone
    plot_polygon(shapely.Polygon([(-drone_size[0]/2, drone_size[1]/2), (drone_size[0]/2, drone_size[1]/2), 
                                 (drone_size[0]/2, -drone_size[1]/2), (-drone_size[0]/2, -drone_size[1]/2)]), 
                                 ax=ax, color=drone_color, add_points=False)

    # Plot the primary solution
    for sensor in population[0]:
        plot_polygon(sensor.get_polygon(), ax=ax, color=primary_solution_color, add_points=False)
        fitness = fitness_function(population[0], view_zones)
        plt.title(f"{first_label} = {fitness[0]:.4f}, {second_label} = {fitness[1]:.4f}")

    # Plot the other solutions
    for sensors in population[1:]:
        for sensor in sensors:
            plot_polygon(sensor.get_polygon(), ax=ax, color=other_solutions_color, add_points=False, alpha=0.1)
    
    # Plot the view zones
    for view_zone in view_zones:
        plot_polygon(view_zone.get_polygon(), ax=ax, color=view_zones_color, add_points=False)
    
    if save:
        plt.savefig(f"{subfolder}/{filename}.png")
        plt.close()
    
    if show:
        plt.show()

    return fig, ax
    
def mutate_sensors(sensors: Genotype, constrains: tuple[float, float] = (0, 0)):
    # Mutate the sensors by rotating and moving them on a normal distribution
    if constrains == (0, 0):
        for sensor in sensors:
            sensor.rotate(random.normalvariate(0, math.pi/6))
            sensor.move(random.normalvariate(0, 1), random.normalvariate(0, 1))
    else:
        for sensor in sensors:
            sensor.rotate(random.normalvariate(0, math.pi/6))
            sensor.move(random.normalvariate(0, 1), random.normalvariate(0, 1), constrains)

def crossover(sensors1: Genotype, sensors2: Genotype) -> tuple[Genotype, Genotype]:
    # Crossover the sensors by swapping them with a 50% chance
    new_sensors_1 = []
    new_sensors_2 = []

    for i in range(len(sensors1)):
        if random.random() < 0.5:
            new_sensors_1.append(copy.deepcopy(sensors1[i]))
            new_sensors_2.append(copy.deepcopy(sensors2[i]))
        else:
            new_sensors_1.append(copy.deepcopy(sensors2[i]))
            new_sensors_2.append(copy.deepcopy(sensors1[i]))

    return new_sensors_1, new_sensors_2

def get_overlap(sensors: Genotype) -> float:
    # Get the overlap of the sensors
    intersection_areas = []

    for i, sensor in enumerate(sensors):
        for other_sensor in sensors[i+1:]:
            intersection_areas.append(
                -sensor.get_polygon().intersection(other_sensor.get_polygon()).area / sensor.get_polygon().union(other_sensor.get_polygon()).area)

    return sum(intersection_areas) / len(intersection_areas)

def get_angle_density(sensors: Genotype) -> float:
    # Get the angle density of the sensors
    angels = []
    positions = []

    middle_point = sensors[0].position

    for sensor in sensors:
        middle_point= (sensor.position[0] + middle_point[0]) / 2, (sensor.position[1] + middle_point[1]) / 2

    for i, sensor in enumerate(sensors):
        angel = sensor.rotation % (2 * math.pi)
        if angel < 0:
            angel += math.pi * 2
        mean_angel = (middle_point[0] - sensor.position[0], middle_point[1] - sensor.position[1])
        angel_vector = math.atan2(mean_angel[1], mean_angel[0])
        if angel_vector < 0:
            angel_vector = math.pi * 2 + angel_vector
        coef = -abs(abs(angel_vector - angel) - math.pi) / math.pi
        angels.append(coef)
                
    return sum(angels) / len(angels)

def get_horizontal_symmetry(sensors: Genotype) -> float:
    left_side_sensors = list(map(lambda x: x.position[0] < 0, sensors))
    right_side_sensors = list(map(lambda x: x.position[0] >= 0, sensors))

    
    return 1

def fitness_function(sensors: Genotype, view_zones: list[ViewZone]) -> tuple[float, float]:
    return get_angle_density(sensors), get_overlap(sensors)

def create_gif(folder_path: str, filename: str):
    # Create a gif from the images in the folder
    images = [img for img in os.listdir(folder_path) if img.endswith(".png") and img.startswith(filename)]
    images.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
    images = [Image.open(os.path.join(folder_path, img)) for img in images]
    images[0].save(os.path.join(folder_path, f"{filename}.gif"), save_all=True, append_images=images[1:], duration=100, loop=0)

def get_fronts(population_fitness: list[IndexedFitness], 
                  dominance: dict[int, list[int]], 
                  dominated: dict[int, int]):
    front = list(filter(lambda x: dominated[x[1]] == 0, population_fitness))
    others = list(filter(lambda x: dominated[x[1]] > 0, population_fitness))

    if len(others) == 0:
        return [individual[1] for individual in front]

    front_indexes = [individual[1] for individual in front]

    for individual in front:
        for dominated_individual in dominance[individual[1]]:
            dominated[dominated_individual] -= 1
        dominance.pop(individual[1])
        dominated.pop(individual[1])

    return front_indexes, get_fronts(others, dominance, dominated)

def decode_fronts(encoded_fronts) -> list[Front]:
    # Decode the fronts to a list of lists of sensors
    decoded_fronts = []

    while True:
        if type(encoded_fronts[1]) == tuple:
            decoded_fronts.append(encoded_fronts[0])
            encoded_fronts = encoded_fronts[1]
        else:
            decoded_fronts.append(encoded_fronts[0])
            decoded_fronts.append(encoded_fronts[1])
            break
            
    return decoded_fronts

def non_dominated_sorting(population: Population, view_zones: list[ViewZone]) -> list[Front]:
    population_fitness = [(fitness_function(sensors, view_zones), i) for i, sensors in enumerate(population)]

    dominance = {i: [] for i in range(len(population_fitness))}
    dominated = {i: 0 for i in range(len(population_fitness))}

    for individual in population_fitness:
        for other_individual in population_fitness:
            if (individual[0][0] >= other_individual[0][0] and individual[0][1] >= other_individual[0][1]) \
                and (individual[0][0] > other_individual[0][0] or individual[0][1] > other_individual[0][1]):
                dominance[individual[1]].append(other_individual[1])
                dominated[other_individual[1]] += 1

    return decode_fronts(get_fronts(population_fitness, dominance, dominated))

def draw_plan(drone_size: tuple[float, float], sensors: list[Sensor]) -> None:
    drone_polygon = shapely.Polygon([(-drone_size[0]/2, drone_size[1]/2), (drone_size[0]/2, drone_size[1]/2), 
                                     (drone_size[0]/2, -drone_size[1]/2), (-drone_size[0]/2, -drone_size[1]/2)])

    _, ax = plt.subplots()
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True)

    plot_polygon(drone_polygon, ax=ax, color='black', add_points=False)

    for sensor in sensors:
        plot_polygon(sensor.get_polygon(), ax=ax, color='green', add_points=False)

    plt.show()

def draw_fronts(population: Population, 
                view_zones: list[ViewZone],
                show: bool = True, 
                save: bool = False, 
                subfolder: str = "",
                filename: str = "pareto_frontier",
                xlabel: str = "None",
                ylabel: str = "None"):

    os.makedirs(subfolder, exist_ok=True)

    pop_fitness = [fitness_function(population[i], view_zones) for i in range(len(population))]
    fronts_indexes = non_dominated_sorting(population, view_zones)

    _, ax = plt.subplots()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    colors = ['red', 'blue', 'green', 'purple', 'orange', 'yellow', 'pink', 'brown', 'gray', 'black']

    for k, front in enumerate(fronts_indexes):
        color = colors[k % len(colors)]
        for index in front:
            ax.plot(pop_fitness[index][0], pop_fitness[index][1], 'o', color=color)

    if save:
        plt.savefig(f"{subfolder}/{filename}.png")
        plt.close()

    if show:
        plt.show()

def save_first_front(population: Population, fronts_indexes: Fronts, drone_size, first_label: str, second_label: str):
    sorted_individuals: list[Genotype] = []
    
    for index in fronts_indexes[0]:
        sorted_individuals.append(population[index])

    sorted_individuals.sort(key=lambda x: fitness_function(x, [])[0])

    for i, individual in enumerate(sorted_individuals):
        draw_experiment(drone_size, [individual], [], save=True, subfolder="first_front", filename=f"individual_{i}", first_label=first_label, second_label=second_label)

def start_evolution(drone_size: tuple[float, float],
                    population: Population, 
                    view_zones: list[ViewZone], 
                    population_size: int, 
                    gen_num: int, 
                    sensors_gif: bool = False,
                    front_gif: bool = False,
                    sensor_save: bool = True,
                    front_save: bool = True,
                    xlabel: str = "None",
                    ylabel: str = "None"
                    ):
    """
    This function starts the evolution by using concepts of NSGA-II genetic algorithm.
    It takes the following parameters:
    - population: list of lists of sensors
    - view_zones: list of view zones
    - population_size: size of the population
    - gen_num: number of generations before stopping

    - sensors_gif: if True, a gif of the sensors will be saved
    - front_gif: if True, a gif of the pareto frontier will be saved

    - sensor_save: if True, a png of the sensors at final generation will be saved
    - front_save: if True, a png of the pareto frontier at final generation will be saved
    """
    shutil.rmtree("evolution")
    shutil.rmtree("first_front")
    shutil.rmtree("pareto_frontier")

    draw_experiment(drone_size, [population[0]], view_zones, show=False, save=True, subfolder="evolution", filename=f"evolution_-1", first_label=xlabel, second_label=ylabel)


    for i in range(gen_num):
        # Crossover and mutation
        for k in range(population_size // 2):
            new_sensors_1, new_sensors_2 = crossover(population[k], population[k + 1])
            mutate_sensors(new_sensors_1, (drone_size[0]/2, drone_size[1]/2))
            mutate_sensors(new_sensors_2, (drone_size[0]/2, drone_size[1]/2))

            population.append(new_sensors_1)
            population.append(new_sensors_2)

        # Non-dominated sorting
        fronts_indexes = non_dominated_sorting(population, view_zones)
        
        new_population = []

        # Forming new population
        while len(new_population) < population_size:
            for front in fronts_indexes:
                for index in front:
                    new_population.append(population[index])

                    if len(new_population) >= population_size:
                        break

                if len(new_population) >= population_size:
                    break

            if len(new_population) >= population_size:
                break

        population = new_population

        fronts_indexes = non_dominated_sorting(population, view_zones)

        # Print the fitness of one of individual from first front
        print(f"Generation {i}: {fitness_function(population[0], view_zones)}")

        if sensors_gif:
            draw_experiment(drone_size, population[:5], view_zones, show=False, save=True, subfolder="evolution", filename=f"evolution_{i}", first_label=xlabel, second_label=ylabel)
            
        if front_gif:
            draw_fronts(population, view_zones, show=False, save=True, subfolder="pareto_frontier", filename=f"pareto_frontier_{i}", xlabel=xlabel, ylabel=ylabel)

    i += 1

    if sensor_save:
        draw_experiment(drone_size, population[:5], view_zones, show=True, save=True, subfolder="evolution", filename=f"evolution_{i}", first_label=xlabel, second_label=ylabel)
    else:
        draw_experiment(drone_size, population[:5], view_zones, show=True, save=False, subfolder="evolution", filename=f"evolution_{i}", first_label=xlabel, second_label=ylabel)

    if front_save:
        draw_fronts(population, view_zones, show=True, save=True, subfolder="pareto_frontier", filename=f"pareto_frontier_{i}", xlabel=xlabel, ylabel=ylabel)
    else:
        draw_fronts(population, view_zones, show=True, save=False, subfolder="pareto_frontier", filename=f"pareto_frontier_{i}", xlabel=xlabel, ylabel=ylabel)

    if sensors_gif:
        create_gif("evolution", "evolution")

    if front_gif:
        create_gif("pareto_frontier", "pareto_frontier")

    save_first_front(population, fronts_indexes, drone_size, first_label = xlabel, second_label = ylabel)
    
    return population

def main():
    # Create all needed sensors with their parameters
    sensors = [
        Sensor((0, 0), 30, math.pi/4),
        Sensor((0, 0), 30, math.pi/4),
        Sensor((0, 0), 30, math.pi/4)
    ]
    
    # Create all zones which sensors needs to cover
    view_zones = [get_square_view_zone((0, 10), 3), 
                  get_square_view_zone((-7, 6), 3), 
                  get_square_view_zone((7, 6), 3)]

    # Define population size
    population_size = 100

    # Create initial population by deepcopying
    population = [copy.deepcopy(sensors) for _ in range(population_size)]

    # Start the evolution
    population = start_evolution((10, 10), 
                                 population, 
                                 view_zones, 
                                 population_size, 
                                 5,
                                 front_gif=True,
                                 sensors_gif=True,
                                 xlabel="angel density",
                                 ylabel="overlapping")

if __name__ == "__main__":
    main()
    