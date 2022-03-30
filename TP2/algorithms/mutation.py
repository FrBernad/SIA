from random import random

from utils.knapsack import Chromosome


def random_mutation(
        chromosome: Chromosome,
        probability: float
) -> Chromosome:
    return tuple(map(lambda allele: not allele if random() < probability else allele, list(chromosome)))
