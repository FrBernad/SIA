from random import random

from utils.chromosome_factory import Chromosome, ChromosomeFactory


def random_mutation(
        chromosome: Chromosome,
        chromosome_factory: ChromosomeFactory,
        probability: float
) -> Chromosome:
    return chromosome_factory.generate(
        tuple(map(lambda allele: not allele if random() < probability else allele, list(chromosome.genes)))
    )
