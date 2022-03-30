from random import randint
from typing import Tuple, List

Chromosome = Tuple[bool, ...]
Population = List[Chromosome]

DEFAULT_POPULATION_SIZE = 20
DEFAULT_POPULATION_PROBABILITY = 0.005


class Backpack(object):

    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Backpack, cls).__new__(cls)
        return cls.instance

    def __init__(self, max_capacity, max_weight, fitness_function, elements=None):
        if elements is None:
            elements = []
        self.max_weight = max_weight
        self.max_capacity = max_capacity
        self.fitness_function = fitness_function
        self.elements = elements

    def calculate_weight(self, chromosome: Chromosome):
        total_weight = 0
        for p, e in zip(chromosome, self.elements):
            if p:
                total_weight += e.weight
        return total_weight

    def calculate_benefits(self, chromosome: Chromosome):
        total_benefits = 0
        for p, e in zip(chromosome, self.elements):
            if p:
                total_benefits += e.benefit
        return total_benefits

    def calculate_fitness(self, chromosome: Chromosome):
        return self.fitness_function(self, chromosome)


class Element:
    def __init__(self, benefit, weight):
        self.weight = weight
        self.benefit = benefit


def generate_random_population(
        backpack: Backpack,
        size: int = DEFAULT_POPULATION_SIZE,
) -> Population:
    population = set()

    while len(population) < size:
        chromosome = [False] * backpack.max_capacity
        generated = False

        while not generated:
            random_index = randint(0, backpack.max_capacity - 1)

            chromosome[random_index] = True
            if backpack.calculate_weight(chromosome) > backpack.max_weight:
                generated = True
                chromosome[random_index] = False

        population.add(tuple(chromosome))

    return list(population)
