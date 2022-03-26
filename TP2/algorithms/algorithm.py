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

    current_generation = generation_zero

    current_generation_population_size = len(current_generation)

    generation_children = set()

    print(list(map(lambda chr: backpack.calculate_fitness(chr), current_generation)))

    for j in range(0, 10000):
        for i in range(current_generation_population_size):
            selected_couple = couple_selection(current_generation)
            selected_couple = crossover(selected_couple, 10)
            first_chromosome = mutation(selected_couple[0])
            second_chromosome = mutation(selected_couple[1])
            generation_children.add(first_chromosome)
            generation_children.add(second_chromosome)

        current_generation = selection(list(generation_children), backpack, current_generation_population_size)
        current_generation_population_size = len(current_generation)
        generation_children = set()

        if j % 100 == 0:
            print(list(map(lambda chr: backpack.calculate_fitness(chr), current_generation)))
