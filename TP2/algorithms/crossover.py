from random import sample, random
from typing import Tuple

from utils.chromosome_factory import Chromosome, ChromosomeFactory
from utils.config import CrossoverMethodConfig


def simple_crossover(
        couple: Tuple[Chromosome, Chromosome],
        chromosome_factory: ChromosomeFactory,
        config: CrossoverMethodConfig = None
) -> Tuple[Chromosome, Chromosome]:
    return multiple_crossover(couple, chromosome_factory, CrossoverMethodConfig(simple_crossover, n=1))


def multiple_crossover(
        couple: Tuple[Chromosome, Chromosome],
        chromosome_factory: ChromosomeFactory,
        config: CrossoverMethodConfig = None
) -> Tuple[Chromosome, Chromosome]:
    crossover_points = sample(range(1, len(couple[0].genes) - 1), config.n)
    crossover_points.append(len(couple[0].genes))
    crossover_points.sort()
    s1 = list(couple[0].genes)
    s2 = list(couple[1].genes)

    for i in range(len(crossover_points)):
        if i % 2 != 0:
            _swap_elements(crossover_points[i - 1], crossover_points[i], s1, s2)

    return chromosome_factory.generate(tuple(s1)), chromosome_factory.generate(tuple(s2))


UNIFORM_PROBABILITY = 0.5


def uniform_crossover(
        couple: Tuple[Chromosome, Chromosome],
        chromosome_factory: ChromosomeFactory,
        config: CrossoverMethodConfig = None
) -> Tuple[Chromosome, Chromosome]:
    s1 = list(couple[0].genes)
    s2 = list(couple[1].genes)

    for i in range(len(s1)):
        if random() < UNIFORM_PROBABILITY:
            s1[i], s2[i] = s2[i], s1[i]

    return chromosome_factory.generate(tuple(s1)), chromosome_factory.generate(tuple(s2))


def _swap_elements(start_point: int, end_point: int, list1: list, list2: list):
    list1[start_point:end_point], list2[start_point:end_point] = \
        list2[start_point:end_point], list1[start_point:end_point]


CROSSOVER_METHODS = {
    'simple_crossover': simple_crossover,
    'multiple_crossover': multiple_crossover,
    'uniform_crossover': uniform_crossover
}
