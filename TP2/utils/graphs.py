import plotly.graph_objs as go

from algorithms.algorithm import genetic_algorithm
from algorithms.couple_selection import COUPLE_SELECTION_METHODS, rand_couple_selection
from algorithms.crossover import CROSSOVER_METHODS, multiple_crossover, simple_crossover
from algorithms.fitness_functions import benefit_weight_ratio
from algorithms.mutation import random_mutation
from algorithms.selection import SELECTION_METHODS, boltzmann_selection
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
):
    end_condition_config = EndConditionConfig(
        generations_count=2000,
        time=5 * 60,
        acceptable_solution_generation_count=1000,
        structure_consecutive_generations=30,
        percentage=0.7,
        fitness_consecutive_generations=30,
        fitness_min_generations=1000,
        structure_min_generations=1000
    )

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
            y=list(map(lambda c: c.fitness, stats.worst_solutions)),
        )
    ],
        {
            # 'title': f'{selection_method[0].replace("_", " ").title()} - Population Size {initial_population_size} - '
            #          f'Mutation Probability {mutation_probability} - {crossover_method[0].replace("_", " ").title()} - '
            #          f'{couple_selection_method[0].replace("_", " ").title()}',
            'title': {
                'text': f'{selection_method[0].replace("_", " ").title()} - {crossover_method[0].replace("_", " ").title()}',
                'x': 0.5
            },
            'xaxis_title': "Generation",
            'yaxis_title': "Fitness",
            'width': 800,
            'height': 800,
            'legend': dict(
                bgcolor='rgba(0,0,0,0)',
                y=0,
                x=0.8)
        }
    )
    fig.update_xaxes(type="log")
    fig.show()


def _generate_boltzman_plots(
        chromosome_factory,
        initial_population_size,
        mutation_probability
):
    crossover_method = ['simple_crossover', simple_crossover]
    selection_method = ['boltzmann_selection', boltzmann_selection]
    couple_selection_method = ['rand_couple_selection', rand_couple_selection]
    first_generation = chromosome_factory.generate_random_population(initial_population_size)

    settings = [
        dict(k=0.1, T0=100, Tc=10),
        dict(k=0.5, T0=100, Tc=10),
        dict(k=100, T0=100, Tc=10),
    ]

    for s in settings:
        end_condition_config = EndConditionConfig(
            generations_count=2000,
            time=5 * 60,
            acceptable_solution_generation_count=1000,
            structure_consecutive_generations=30,
            percentage=0.7,
            fitness_consecutive_generations=30,
            fitness_min_generations=1000,
            structure_min_generations=1000
        )

        config = Config({}, initial_population_size, end_condition_config,
                        benefit_weight_ratio, couple_selection_method[1],
                        CrossoverMethodConfig(crossover_method[1], n=4),
                        MutationMethodConfig(random_mutation, probability=mutation_probability),
                        SelectionMethodConfig(selection_method[1], k=s['k'], T0=s['T0'], Tc=s['Tc']))

        print(f'Plotting k={s["k"]}, T0={s["T0"]}, Tc={s["Tc"]}')

        stats = genetic_algorithm(first_generation, chromosome_factory, config.couple_selection_method,
                                  config.crossover_method_config.method, config.mutation_method_config.method,
                                  config.selection_method_config.method, config)

        x = range(0, config.end_condition_config.generations_count)

        fig = go.Figure(
            go.Scatter(
                x=list(x),
                y=list(map(lambda c: c.fitness, stats.best_solutions)),
            ),
            {
                'title': {
                    'text': f'k={s["k"]}, T0={s["T0"]}, Tc={s["Tc"]}',
                    'x': 0.5
                },
                'xaxis_title': "Generation",
                'yaxis_title': "Fitness",
                'width': 800,
                'height': 800
            }
        )
        fig.update_xaxes(type="log")
        fig.show()


def _generate_overlapped_plots(
        chromosome_factory,
        initial_population_size,
        mutation_probability
):
    crossover_method = ['simple_crossover', simple_crossover]
    couple_selection_method = ['rand_couple_selection', rand_couple_selection]
    first_generation = chromosome_factory.generate_random_population(initial_population_size)

    figures = []

    for selection_method in SELECTION_METHODS.items():
        print(f'\tGenerating {selection_method[0]} plot')

        end_condition_config = EndConditionConfig(
            generations_count=2000,
            time=5 * 60,
            acceptable_solution_generation_count=1000,
            structure_consecutive_generations=30,
            percentage=0.7,
            fitness_consecutive_generations=30,
            fitness_min_generations=1000,
            structure_min_generations=1000
        )

        config = Config({}, initial_population_size, end_condition_config,
                        benefit_weight_ratio, couple_selection_method[1],
                        CrossoverMethodConfig(crossover_method[1], n=4),
                        MutationMethodConfig(random_mutation, probability=mutation_probability),
                        SelectionMethodConfig(selection_method[1], threshold=0.7, k=0.5, T0=200, Tc=10,
                                              truncation_size=initial_population_size - 2))

        stats = genetic_algorithm(first_generation, chromosome_factory, config.couple_selection_method,
                                  config.crossover_method_config.method, config.mutation_method_config.method,
                                  config.selection_method_config.method, config)

        x = range(0, config.end_condition_config.generations_count)
        figures.append(
            go.Scatter(
                name=f'{selection_method[0].replace("_", " ").title()}',
                x=list(x),
                y=list(map(lambda c: c.fitness, stats.best_solutions)),
            ),
        )

    fig = go.Figure(
        figures,
        {
            # 'title': f'Population Size {initial_population_size} - '
            #          f'Mutation Probability {mutation_probability} - {crossover_method[0].replace("_", " ").title()} - '
            #          f'{couple_selection_method[0].replace("_", " ").title()}',
            'xaxis_title': "Generation",
            'yaxis_title': "Fitness",
            'width': 800,
            'height': 800,
            'legend': dict(
                bgcolor='rgba(0,0,0,0)',
                y=0,
                x=0.7
            )
        }
    )

    fig.show()


def _generate_individual_plots(chromosome_factory, initial_population_sizes, mutation_probabilities):
    for initial_population_size in initial_population_sizes:
        first_generation = chromosome_factory.generate_random_population(initial_population_size)
        for mutation_probability in mutation_probabilities:
            for selection_method in SELECTION_METHODS.items():
                for crossover_method in CROSSOVER_METHODS.items():
                    # for couple_selection_method in COUPLE_SELECTION_METHODS.items(): FIXME: quizas probar estos
                    _generate_selection_graph(
                        first_generation,
                        ['rand_couple_selection', rand_couple_selection],
                        selection_method,
                        crossover_method,
                        initial_population_size,
                        mutation_probability
                    )


if __name__ == '__main__':
    INITIAL_POPULATION_SIZES = [100]
    MUTATION_PROBABILITY = [0.005]

    chromosome_factory = ChromosomeFactory(_get_knapsack_data("../" + DEFAULT_DATA_FILE), benefit_weight_ratio)

    # print('Generating individual plots')
    # _generate_individual_plots(chromosome_factory, INITIAL_POPULATION_SIZES, MUTATION_PROBABILITY)

    print('Generating Boltzmann plots')
    _generate_boltzman_plots(chromosome_factory, 100, 0.005)

    # print('Generating overlapped plots')
    # _generate_overlapped_plots(chromosome_factory, 100, 0.005)
