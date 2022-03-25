from random import randint, random
from typing import Set, Tuple

Chromosome = Tuple[bool, ...]
Population = Set[Chromosome]

DEFAULT_POPULATION_SIZE = 1000
DEFAULT_POPULATION_PROBABILITY = 0.005


class Backpack(object):

    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Backpack, cls).__new__(cls)
        return cls.instance

    def __init__(self, max_capacity, max_weight, elements=None):
        if elements is None:
            elements = []
        self.max_weight = max_weight
        self.max_capacity = max_capacity
        self.elements = elements

    def _calculate_weight(self, chromosome: Chromosome):
        total_weight = 0
        for p, e in zip(chromosome, self.elements):
            if p:
                total_weight += e.weight
        return total_weight

    def _calculate_benefits(self, chromosome: Chromosome):
        total_benefits = 0
        for p, e in zip(chromosome, self.elements):
            if p:
                total_benefits += e.benefit
        return total_benefits

    def calculate_fitness(self, chromosome: Chromosome):
        return 0 if self._calculate_weight(chromosome) > self.max_capacity else self._calculate_benefits(chromosome)


class Element:
    def __init__(self, benefit, weight):
        self.weight = weight
        self.benefit = benefit


def generate_random_population(
        backpack_capacity: int,
        size: int = DEFAULT_POPULATION_SIZE,
        probability: float = DEFAULT_POPULATION_PROBABILITY
) -> Population:
    population: Population = set()

    while len(population) < size:
        chromosome = [False] * backpack_capacity
        for i in range(backpack_capacity):
            if random() < probability:
                chromosome[i] = True
        population.add(tuple(chromosome))

    return population
