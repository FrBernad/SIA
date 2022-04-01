from typing import Callable

from algorithms.end_conditions import init_end_conditions, check_end_conditions, update_end_conditions, Stats
from utils.knapsack import Population, Knapsack
from utils.config import Config


def genetic_algorithm(
        generation_zero: Population,
        knapsack: Knapsack,
        couple_selection: Callable,
        crossover: Callable,
        mutation: Callable,
        selection: Callable,
        config: Config
) -> Stats:
    current_generation = generation_zero
    generation_children = set()

    stats = Stats(config.end_condition_config, current_generation, knapsack)
    init_end_conditions(config.end_condition_config, current_generation)

    j = 0
    while not check_end_conditions(config.end_condition_config):

        # if j % 100 == 0:
        #     print(f'Generation {j}')
        #     print("Fitness-Benefit-Weight")
        #     print(list(map(lambda chr: knapsack.calculate_fitness(chr), current_generation)))
        #     print(list(map(lambda chr: knapsack.calculate_benefit(chr), current_generation)))
        #     print(list(map(lambda chr: knapsack.calculate_weight(chr), current_generation)))
        #     print('\n')

        j += 1

        while len(generation_children) < config.initial_population_size:
            selected_couple = couple_selection(current_generation, knapsack)
            selected_couple = crossover(selected_couple, config=config.crossover_method_config)
            first_chromosome = mutation(selected_couple[0], config.mutation_method_config.probability)
            second_chromosome = mutation(selected_couple[1], config.mutation_method_config.probability)
            if first_chromosome not in current_generation:
                generation_children.add(first_chromosome)
            if len(generation_children) < config.initial_population_size and second_chromosome not in current_generation:
                generation_children.add(second_chromosome)

        current_generation = selection(list(generation_children) + current_generation,
                                       config.end_condition_config.stats.generations_count,
                                       knapsack,
                                       config.initial_population_size, config.selection_method_config
                                       )

        generation_children.clear()
        update_end_conditions(config.end_condition_config, current_generation, knapsack)
        stats.update(config.end_condition_config, current_generation)

    sol = sorted(current_generation,
                 key=lambda chromosome: knapsack.calculate_weight(chromosome),
                 reverse=True
                 )

    _print_solution(sol, j, knapsack, config)

    return stats


def _print_solution(sol, j, knapsack, config):
    print(f'Generation {j}')
    print("Fitness-Benefit-Weight")

    print(list(map(lambda chr: knapsack.calculate_fitness(chr), sol)))
    print(list(map(lambda chr: knapsack.calculate_benefit(chr), sol)))
    print(list(map(lambda chr: knapsack.calculate_weight(chr), sol)))
    print('\n')
