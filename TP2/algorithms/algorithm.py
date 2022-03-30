from typing import Callable

from algorithms.end_conditions import init_end_conditions, check_end_conditions, update_end_conditions, Stats
from utils.backpack import Population, Backpack
from utils.config import Config


def genetic_algorithm(
        generation_zero: Population,
        backpack: Backpack,
        couple_selection: Callable,
        crossover: Callable,
        mutation: Callable,
        selection: Callable,
        config: Config
) -> Stats:
    current_generation = generation_zero
    generation_children = set()

    stats = Stats(config.endConditionConfig, current_generation, backpack)
    init_end_conditions(config.endConditionConfig, current_generation)

    j = 0
    while not check_end_conditions(config.endConditionConfig):

        if j % 100 == 0:
            print(f'Generation {j}')
            print("Fitness-Benefit-Weight")
            print(list(map(lambda chr: backpack.calculate_fitness(chr), current_generation)))
            print(list(map(lambda chr: backpack.calculate_benefit(chr), current_generation)))
            print(list(map(lambda chr: backpack.calculate_weight(chr), current_generation)))
            print('\n')

        j += 1

        while len(generation_children) < config.initial_population_size:
            selected_couple = couple_selection(current_generation, backpack)
            selected_couple = crossover(selected_couple, config=config.crossover_method_config)
            first_chromosome = mutation(selected_couple[0], config.mutation_method_config.probability)
            second_chromosome = mutation(selected_couple[1], config.mutation_method_config.probability)
            if first_chromosome not in current_generation:
                generation_children.add(first_chromosome)
            if len(generation_children) < config.initial_population_size and second_chromosome not in current_generation:
                generation_children.add(second_chromosome)

        current_generation = selection(list(generation_children) + current_generation, backpack,
                                       config.initial_population_size, config.selection_method_config)

        generation_children = set()
        update_end_conditions(config.endConditionConfig, current_generation, backpack)
        stats.update(config.endConditionConfig, current_generation)

    sol = sorted(current_generation,
                 key=lambda chromosome: backpack.calculate_weight(chromosome),
                 reverse=True
                 )

    _print_solution(sol, j, backpack, config)

    return stats


def _print_solution(sol, j, backpack, config):
    print(f'Generation {j}')
    print("Fitness-Benefit-Weight")

    print(list(map(lambda chr: backpack.calculate_fitness(chr), sol)))
    print(list(map(lambda chr: backpack.calculate_benefit(chr), sol)))
    print(list(map(lambda chr: backpack.calculate_weight(chr), sol)))
    print('\n')

