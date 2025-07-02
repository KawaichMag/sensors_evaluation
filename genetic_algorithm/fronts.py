from objects.Types import IndexedFitness, Front, Population
from objects.Objects import ViewZone
from .metrics import fitness_function


def get_fronts(
    population_fitness: list[IndexedFitness],
    dominance: dict[int, list[int]],
    dominated: dict[int, int],
):
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
        if isinstance(encoded_fronts[1], tuple):
            decoded_fronts.append(encoded_fronts[0])
            encoded_fronts = encoded_fronts[1]
        else:
            decoded_fronts.append(encoded_fronts[0])
            decoded_fronts.append(encoded_fronts[1])
            break

    return decoded_fronts


def non_dominated_sorting(
    population: Population, view_zones: list[ViewZone]
) -> list[Front]:
    population_fitness = [
        (fitness_function(sensors, view_zones), i)
        for i, sensors in enumerate(population)
    ]

    dominance = {i: [] for i in range(len(population_fitness))}
    dominated = {i: 0 for i in range(len(population_fitness))}

    for individual in population_fitness:
        for other_individual in population_fitness:
            if (
                individual[0][0] >= other_individual[0][0]
                and individual[0][1] >= other_individual[0][1]
            ) and (
                individual[0][0] > other_individual[0][0]
                or individual[0][1] > other_individual[0][1]
            ):
                dominance[individual[1]].append(other_individual[1])
                dominated[other_individual[1]] += 1

    return decode_fronts(get_fronts(population_fitness, dominance, dominated))
