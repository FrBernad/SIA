from typing import Callable

from utils.backpack import Population, Backpack


def genetic_algorithm(
        generation_zero: Population,
        backpack: Backpack,
        couple_selection: Callable,
        crossover: Callable,
        mutation: Callable,
        selection: Callable
):
    generation_fitness = list(map(lambda chromosome: backpack.calculate_fitness(chromosome), generation_zero))
    new_generation: Population = set()

    current_generation = generation_zero

    while True:
        for i in range(len(current_generation)):
            selected_couple = couple_selection(current_generation)
            selected_couple = crossover(selected_couple)
            first_chromosome = mutation(selected_couple[0])
            second_chromosome = mutation(selected_couple[1])
            new_generation.add(first_chromosome)
            new_generation.add(second_chromosome)

        current_generation = selection(new_generation, 1000)
        print(current_generation)

