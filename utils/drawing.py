import os
import matplotlib.pyplot as plt
import shapely
from shapely.plotting import plot_polygon
from PIL import Image

from objects.Objects import Sensor, ViewZone
from objects.Types import Genotype, Population, Front, Fronts
from genetic_algorithm.metrics import fitness_function
from genetic_algorithm.fronts import non_dominated_sorting

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

def create_gif(folder_path: str, filename: str):
    # Create a gif from the images in the folder
    images = [img for img in os.listdir(folder_path) if img.endswith(".png") and img.startswith(filename)]
    images.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
    images = [Image.open(os.path.join(folder_path, img)) for img in images]
    images[0].save(os.path.join(folder_path, f"{filename}.gif"), save_all=True, append_images=images[1:], duration=100, loop=0)

def save_first_front(population: Population, fronts_indexes: Fronts, drone_size, first_label: str, second_label: str):
    sorted_individuals: list[Genotype] = []
    
    for index in fronts_indexes[0]:
        sorted_individuals.append(population[index])

    sorted_individuals.sort(key=lambda x: fitness_function(x, [])[0])

    for i, individual in enumerate(sorted_individuals):
        draw_experiment(drone_size, [individual], [], save=True, subfolder="first_front", filename=f"individual_{i}", first_label=first_label, second_label=second_label)
