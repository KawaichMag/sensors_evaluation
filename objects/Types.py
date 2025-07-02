from .Objects import Sensor

type Genotype = list[Sensor]
type Population = list[Genotype]

type Front = list[int]
type Fronts = list[Front]

type Fitness = tuple[float, float]
type IndexedFitness = tuple[Fitness, int]