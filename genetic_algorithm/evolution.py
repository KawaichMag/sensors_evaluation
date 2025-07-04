import random
import math
import copy
import shutil
import os

from objects.Types import Genotype, Population
from objects.Objects import ViewZone
from .metrics import fitness_function
from .fronts import non_dominated_sorting
from utils.drawing import draw_experiment, draw_fronts, create_gif, save_first_front


def mutate_sensors(sensors: Genotype, constrains: tuple[float, float] = (0, 0)):
    # Mutate the sensors by rotating and moving them on a normal distribution
    if constrains == (0, 0):
        for sensor in sensors:
            sensor.rotate(random.normalvariate(0, math.pi / 6))
            sensor.move(random.normalvariate(0, 1), random.normalvariate(0, 1))
    else:
        for sensor in sensors:
            sensor.rotate(random.normalvariate(0, math.pi / 6))
            sensor.move(
                random.normalvariate(0, 1), random.normalvariate(0, 1), constrains
            )


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


def start_evolution(
    drone_size: tuple[float, float],
    population: Population,
    view_zones: list[ViewZone],
    population_size: int,
    gen_num: int,
    sensors_gif: bool = False,
    front_gif: bool = False,
    sensor_save: bool = True,
    front_save: bool = True,
    xlabel: str = "None",
    ylabel: str = "None",
):
    if os.path.exists("experiments/evolution"):
        shutil.rmtree("experiments/evolution")
    if os.path.exists("experiments/first_front"):
        shutil.rmtree("experiments/first_front")
    if os.path.exists("experiments/pareto_frontier"):
        shutil.rmtree("experiments/pareto_frontier")

    draw_experiment(
        drone_size,
        [population[0]],
        view_zones,
        show=False,
        save=True,
        subfolder="experiments",
        filename="evolution_first",
        first_label=xlabel,
        second_label=ylabel,
    )

    for i in range(gen_num):
        # Crossover and mutation
        for k in range(population_size // 2):
            new_sensors_1, new_sensors_2 = crossover(population[k], population[k + 1])
            mutate_sensors(new_sensors_1, (drone_size[0] / 2, drone_size[1] / 2))
            mutate_sensors(new_sensors_2, (drone_size[0] / 2, drone_size[1] / 2))

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
            draw_experiment(
                drone_size,
                population[:5],
                view_zones,
                show=False,
                save=True,
                subfolder="experiments/evolution",
                filename=f"evolution_{i}",
                first_label=xlabel,
                second_label=ylabel,
            )

        if front_gif:
            draw_fronts(
                population,
                view_zones,
                show=False,
                save=True,
                subfolder="experiments/pareto_frontier",
                filename=f"pareto_frontier_{i}",
                xlabel=xlabel,
                ylabel=ylabel,
            )

    i += 1

    if sensor_save:
        draw_experiment(
            drone_size,
            population[:5],
            view_zones,
            show=True,
            save=True,
            subfolder="experiments/evolution",
            filename=f"evolution_{i}",
            first_label=xlabel,
            second_label=ylabel,
        )
    else:
        draw_experiment(
            drone_size,
            population[:5],
            view_zones,
            show=True,
            save=False,
            subfolder="experiments/evolution",
            filename=f"evolution_{i}",
            first_label=xlabel,
            second_label=ylabel,
        )

    if front_save:
        draw_fronts(
            population,
            view_zones,
            show=True,
            save=True,
            subfolder="experiments/pareto_frontier",
            filename=f"pareto_frontier_{i}",
            xlabel=xlabel,
            ylabel=ylabel,
        )
    else:
        draw_fronts(
            population,
            view_zones,
            show=True,
            save=False,
            subfolder="experiments/pareto_frontier",
            filename=f"pareto_frontier_{i}",
            xlabel=xlabel,
            ylabel=ylabel,
        )

    if sensors_gif:
        create_gif("experiments/evolution", "evolution")

    if front_gif:
        create_gif("experiments/pareto_frontier", "pareto_frontier")

    save_first_front(
        population, fronts_indexes, drone_size, first_label=xlabel, second_label=ylabel
    )

    return population
