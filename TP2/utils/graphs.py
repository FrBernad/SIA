import plotly.graph_objs as go

from algorithms.algorithm import genetic_algorithm
from algorithms.couple_selection import COUPLE_SELECTION_METHODS
from algorithms.crossover import CROSSOVER_METHODS
from algorithms.fitness_functions import benefit_weight_ratio
from algorithms.mutation import random_mutation
from algorithms.selection import SELECTION_METHODS
from knapsack_solver import _get_knapsack_data
from utils.argument_parser import DEFAULT_DATA_FILE
from utils.chromosome_factory import ChromosomeFactory
from utils.config import Config, EndConditionConfig, CrossoverMethodConfig, MutationMethodConfig, SelectionMethodConfig


def _generate_selection_graph(
        first_generation,
        couple_selection_method,
        selection_method,
        crossover_method,
        initial_population_size,
        mutation_probability,
        end_condition_config
):
    config = Config({}, initial_population_size, end_condition_config,
                    benefit_weight_ratio, couple_selection_method[1],
                    CrossoverMethodConfig(crossover_method[1], n=4),
                    MutationMethodConfig(random_mutation, probability=mutation_probability),
                    SelectionMethodConfig(selection_method[1], threshold=0.7, k=0.5, T0=200, Tc=10,
                                          truncation_size=50))

    stats = genetic_algorithm(first_generation, chromosome_factory, config.couple_selection_method,
                              config.crossover_method_config.method, config.mutation_method_config.method,
                              config.selection_method_config.method, config)

    x = range(0, config.end_condition_config.generations_count)

    fig = go.Figure([
        go.Scatter(
            name='best fitness',
            x=list(x),
            y=list(map(lambda c: c.fitness, stats.best_solutions)),
        ),
        go.Scatter(
            name='average fitness',
            x=list(x),
            y=list(stats.avg_fitness),
        ),
        go.Scatter(
            name='worst fitness',
            x=list(x),
            y=list(map(lambda c: c.fitness, stats.worst_fitness)),
        )
    ],
        {
            'title': f'{selection_method[0].replace("_", " ").title()} - Population Size {initial_population_size} - '
                     f'Mutation Probability {mutation_probability} - {crossover_method[0].replace("_", " ").title()} - '
                     f'{couple_selection_method[0].replace("_", " ").title()}',
            'xaxis_title': "Generation",
            'yaxis_title': "Fitness"
        }
    )

    fig.show()


# def _generate_overlapped_selections_graph(
#         first_generation,
#         initial_population_size,
#         mutation_probability,
#         end_condition_config
# ):
#     config = Config({}, initial_population_size, end_condition_config,
#                     benefit_weight_ratio, couple_selection_method[1],
#                     CrossoverMethodConfig(crossover_method[1], n=4),
#                     MutationMethodConfig(random_mutation, probability=mutation_probability),
#                     SelectionMethodConfig(selection_method[1], threshold=0.7, k=0.3, T0=100, Tc=50,
#                                           truncation_size=initial_population_size - 2))
#
#     stats = genetic_algorithm(first_generation, chromosome_factory, config.couple_selection_method,
#                               config.crossover_method_config.method, config.mutation_method_config.method,
#                               config.selection_method_config.method, config)
#
#     x = range(0, config.end_condition_config.generations_count)
#
#     figures = []
#
#     fig = go.Figure(
#         figures,
#         {
#             'title': f'{selection_method[0].replace("_", " ").title()} - Population Size {initial_population_size} - '
#                      f'Mutation Probability {mutation_probability} - {crossover_method[0].replace("_", " ").title()} - '
#                      f'{couple_selection_method[0].replace("_", " ").title()}',
#             'xaxis_title': "Generation",
#             'yaxis_title': "Fitness"
# }
# )
#
# fig.show()
#

if __name__ == '__main__':
    INITIAL_POPULATION_SIZES = [100]
    MUTATION_PROBABILITY = [0.005]

    END_CONDITION_CONFIG = EndConditionConfig(
        generations_count=1100, percentage=0.9,
        time=5 * 60, fitness_consecutive_generations=50,
        structure_consecutive_generations=50,
        acceptable_solution_generation_count=1000,
        fitness_min_generations=1000,
        structure_min_generations=1000
    )

    chromosome_factory = ChromosomeFactory(_get_knapsack_data("../" + DEFAULT_DATA_FILE), benefit_weight_ratio)

    for initial_population_size in INITIAL_POPULATION_SIZES:
        first_generation = chromosome_factory.generate_random_population(initial_population_size)
        for mutation_probability in MUTATION_PROBABILITY:
            for selection_method in SELECTION_METHODS.items():
                for crossover_method in CROSSOVER_METHODS.items():
                    for couple_selection_method in COUPLE_SELECTION_METHODS.items():
                        _generate_selection_graph(
                            first_generation,
                            couple_selection_method,
                            selection_method,
                            crossover_method,
                            initial_population_size,
                            mutation_probability,
                            END_CONDITION_CONFIG
                        )
