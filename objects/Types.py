from .Objects import Sensor

Genotype = list[Sensor]
Population = list[Genotype]

Front = list[int]
Fronts = list[Front]

Fitness = tuple[float, float]
IndexedFitness = tuple[Fitness, int]
