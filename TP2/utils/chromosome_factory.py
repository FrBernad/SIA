from random import randint
from typing import List, Callable, Tuple

from utils.knapsack import Knapsack

Population = List['Chromosome']
Genes = Tuple[bool, ...]


class Chromosome:
    def __init__(self, genes: Genes, weight: int, benefit: int):
        self.fitness = 0
        self.benefit = benefit
        self.weight = weight
        self.genes = genes


class ChromosomeFactory:
    def __init__(self, knapsack: Knapsack, fitness_function: Callable):
        self.fitness_function = fitness_function
        self.knapsack = knapsack

    def generate(self, genes: Genes) -> Chromosome:
        c = Chromosome(genes,
                       self._calculate_weight(genes),
                       self._calculate_benefit(genes)
                       )

        c.fitness = self._calculate_fitness(c)

        return c

    def generate_random_population(self, size: int) -> Population:
        population = set()

        while len(population) < size:
            chromosome = [False] * self.knapsack.max_capacity
            generated = False

            while not generated:
                random_index = randint(0, self.knapsack.max_capacity - 1)

                chromosome[random_index] = True
                if self._calculate_weight(chromosome) > self.knapsack.max_weight:
                    generated = True
                    chromosome[random_index] = False

            population.add(self.generate(tuple(chromosome)))

        return list(population)

    def _calculate_weight(self, genes: Genes):
        total_weight = 0
        for p, e in zip(genes, self.knapsack.elements):
            if p:
                total_weight += e.weight
        return total_weight

    def _calculate_benefit(self, genes: Genes):
        total_benefits = 0
        for p, e in zip(genes, self.knapsack.elements):
            if p:
                total_benefits += e.benefit
        return total_benefits

    def _calculate_fitness(self, chromosome: Chromosome):
        return self.fitness_function(self.knapsack, chromosome)
