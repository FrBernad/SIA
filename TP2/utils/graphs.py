import plotly.graph_objs as go

from algorithms.algorithm import genetic_algorithm
from algorithms.couple_selection import rand_couple_selection
from algorithms.crossover import uniform_crossover
from algorithms.fitness_functions import benefit_weight_ratio
from algorithms.mutation import random_mutation
from algorithms.selection import elitism_selection
from knapsack_solver import _get_knapsack_data
from utils.argument_parser import DEFAULT_DATA_FILE
from utils.knapsack import generate_random_population
from utils.config import Config, EndConditionConfig, CrossoverMethodConfig, MutationMethodConfig, SelectionMethodConfig

if __name__ == '__main__':
    initial_population_size = 10
    end_condition_config = EndConditionConfig(
        generations_count=5000, percentage=0.9,
        time=5 * 60, fitness_consecutive_generations=50,
        structure_consecutive_generations=50,
        acceptable_solution_generation_count=5000,
        fitness_min_generations=5000,
        structure_min_generations=5000
    )
    fitness_function = benefit_weight_ratio
    couple_selection_method = rand_couple_selection
    crossover_method_config = CrossoverMethodConfig(uniform_crossover)
    mutation_method_config = MutationMethodConfig(random_mutation, probability=0.05)
    selection_method_config = SelectionMethodConfig(elitism_selection)

    config = Config({}, initial_population_size, end_condition_config,
                    fitness_function, couple_selection_method,
                    crossover_method_config, mutation_method_config,
                    selection_method_config)

    knapsack = _get_knapsack_data("../" + DEFAULT_DATA_FILE, config.fitness_function)

    first_generation = generate_random_population(knapsack, config.initial_population_size)

    stats = genetic_algorithm(first_generation, knapsack, config.couple_selection_method,
                              config.crossover_method_config.method, config.mutation_method_config.method,
                              config.selection_method_config.method, config)

    x = range(0, config.end_condition_config.generations_count + 100, 100)

    fig = go.Figure([
        go.Scatter(
            name='fitness',
            x=list(x),
            y=list(map(lambda s: s['fitness'], stats.get_best_solutions_stats())),
        ),
        go.Scatter(
            name='weight',
            x=list(x),
            y=list(map(lambda s: s['weight'], stats.get_best_solutions_stats())),
        )
    ]
    )

    fig.update_layout(
        title="Generations",
        xaxis_title="Fitness and Weight",
        yaxis_title="Value",
        legend_title="Legend Title"
    )

    fig.show()
