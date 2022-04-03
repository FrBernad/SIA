import plotly.graph_objs as go

from algorithms.algorithm import genetic_algorithm
from algorithms.couple_selection import COUPLE_SELECTION_METHODS, rand_couple_selection
from algorithms.crossover import CROSSOVER_METHODS, multiple_crossover, simple_crossover, uniform_crossover
from algorithms.fitness_functions import benefit_weight_ratio
from algorithms.mutation import random_mutation
from algorithms.selection import SELECTION_METHODS, boltzmann_selection, truncated_selection, tournament_selection
from knapsack_solver import _get_knapsack_data
from utils.argument_parser import DEFAULT_DATA_FILE
from utils.chromosome_factory import ChromosomeFactory
from utils.config import Config, EndConditionConfig, CrossoverMethodConfig, MutationMethodConfig, SelectionMethodConfig


def _generate_boltzmann_plots(
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
        dict(k=0.8, T0=100, Tc=10),
        dict(k=500, T0=100, Tc=10),
    ]

    figures = []

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

        x = range(0, config.end_condition_config.acceptable_solution_generation_count + 20)

        figures.append(
            go.Scatter(
                name=f'k={s["k"]}, T0={s["T0"]}, Tc={s["Tc"]}',
                x=list(x),
                y=list(map(lambda c: c.fitness, stats.best_solutions)),
            ),
        )

    fig = go.Figure(
        figures,
        {
            'title': f'Boltzmann Variations',
            'xaxis_title': "Generation",
            'yaxis_title': "Fitness",
            'legend': dict(
                bgcolor='rgba(0,0,0,0)',
                y=0,
                x=0.8
            )
        }
    )

    fig.update_xaxes(type="log")
    fig.show()


def _generate_truncated_plots(
        chromosome_factory,
        initial_population_size,
        mutation_probability
):
    crossover_method = ['simple_crossover', simple_crossover]
    selection_method = ['truncated_selection', truncated_selection]
    couple_selection_method = ['rand_couple_selection', rand_couple_selection]
    first_generation = chromosome_factory.generate_random_population(initial_population_size)

    settings = [25, 50, 100]

    figures = []

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
                        SelectionMethodConfig(selection_method[1], truncation_size=s))

        print(f'Plotting truncation size {s}')

        stats = genetic_algorithm(first_generation, chromosome_factory, config.couple_selection_method,
                                  config.crossover_method_config.method, config.mutation_method_config.method,
                                  config.selection_method_config.method, config)

        x = range(0, config.end_condition_config.acceptable_solution_generation_count + 20)

        figures.append(
            go.Scatter(
                name=f'{s}',
                x=list(x),
                y=list(map(lambda c: c.fitness, stats.best_solutions)),
            ),
        )

    fig = go.Figure(
        figures,
        {
            'title': f'Truncated Variations',
            'xaxis_title': "Generation",
            'yaxis_title': "Fitness",
            'legend': dict(
                bgcolor='rgba(0,0,0,0)',
                y=0,
                x=0
            )
        }
    )

    fig.update_xaxes(type="log")
    fig.show()


def _generate_tournament_plots(
        chromosome_factory,
        initial_population_size,
        mutation_probability
):
    crossover_method = ['simple_crossover', simple_crossover]
    selection_method = ['tournament_selection', tournament_selection]
    couple_selection_method = ['rand_couple_selection', rand_couple_selection]
    first_generation = chromosome_factory.generate_random_population(initial_population_size)

    settings = [0.5, 0.7, 0.9]

    figures = []

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
                        SelectionMethodConfig(selection_method[1], threshold=s))

        print(f'Plotting threshold {s}')

        stats = genetic_algorithm(first_generation, chromosome_factory, config.couple_selection_method,
                                  config.crossover_method_config.method, config.mutation_method_config.method,
                                  config.selection_method_config.method, config)

        x = range(0, config.end_condition_config.acceptable_solution_generation_count + 20)

        figures.append(
            go.Scatter(
                name=f'{s}',
                x=list(x),
                y=list(map(lambda c: c.fitness, stats.best_solutions)),
            ),
        )

    fig = go.Figure(
        figures,
        {
            'title': f'Tournament Variations',
            'xaxis_title': "Generation",
            'yaxis_title': "Fitness",
            'legend': dict(
                bgcolor='rgba(0,0,0,0)',
                y=0,
                x=0.8
            )
        }
    )

    fig.update_xaxes(type="log")
    fig.show()


def _generate_overlapped_plots(
        chromosome_factory,
        initial_population_size,
        mutation_probability,
        crossover_method
):
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

        x = range(0, config.end_condition_config.acceptable_solution_generation_count + 20)
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
            'title': f'{crossover_method[0].replace("_", " ").title()}',
            'xaxis_title': "Generation",
            'yaxis_title': "Fitness",
            'legend': dict(
                bgcolor='rgba(0,0,0,0)',
                y=0,
                x=0.8
            )
        }
    )

    fig.update_xaxes(type="log")
    fig.show()


def _generate_mutation_plots(chromosome_factory):
    initial_population_size = 100

    first_generation = chromosome_factory.generate_random_population(initial_population_size)

    couple_selection_method = ['rand_couple_selection', rand_couple_selection]
    crossover_method = ["simple_crossover", simple_crossover]

    for selection_method in SELECTION_METHODS.items():
        figures = []
        for mutation_probability in [0.005, 0.05, 0.5]:
            print(f'\tGenerating {selection_method[0]} {mutation_probability} plot')

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

            x = range(0, config.end_condition_config.acceptable_solution_generation_count + 20)
            figures.append(
                go.Scatter(
                    name=f'{mutation_probability}',
                    x=list(x),
                    y=list(map(lambda c: c.fitness, stats.best_solutions)),
                ),
            )

        fig = go.Figure(
            figures,
            {
                'title': f'{selection_method[0].replace("_", " ").title()}',
                'xaxis_title': "Generation",
                'yaxis_title': "Fitness",
                'legend': dict(
                    bgcolor='rgba(0,0,0,0)',
                    y=0,
                    x=0.8
                )
            }
        )

        fig.update_xaxes(type="log")
        fig.show()


def _generate_population_plots(chromosome_factory):
    initial_population_sizes = [10, 50, 100]

    couple_selection_method = ['rand_couple_selection', rand_couple_selection]
    crossover_method = ["simple_crossover", simple_crossover]

    for selection_method in SELECTION_METHODS.items():
        figures = []
        for initial_population in initial_population_sizes:
            print(f'\tGenerating {selection_method[0]} {initial_population} plot')

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

            config = Config({}, initial_population, end_condition_config,
                            benefit_weight_ratio, couple_selection_method[1],
                            CrossoverMethodConfig(crossover_method[1], n=4),
                            MutationMethodConfig(random_mutation, probability=MUTATION_PROBABILITY[0]),
                            SelectionMethodConfig(selection_method[1], threshold=0.7, k=0.5, T0=200, Tc=10,
                                                  truncation_size=initial_population - 2))

            first_generation = chromosome_factory.generate_random_population(initial_population)

            stats = genetic_algorithm(first_generation, chromosome_factory, config.couple_selection_method,
                                      config.crossover_method_config.method, config.mutation_method_config.method,
                                      config.selection_method_config.method, config)

            x = range(0, config.end_condition_config.acceptable_solution_generation_count + 20)
            figures.append(
                go.Scatter(
                    name=f'{initial_population}',
                    x=list(x),
                    y=list(map(lambda c: c.fitness, stats.best_solutions)),
                ),
            )

        fig = go.Figure(
                figures,
                {
                    'title': f'{selection_method[0].replace("_", " ").title()}',
                    'xaxis_title': "Generation",
                    'yaxis_title': "Fitness",
                    'legend': dict(
                        bgcolor='rgba(0,0,0,0)',
                        y=0,
                        x=0.8
                    )
                }
            )

        fig.update_xaxes(type="log")
        fig.show()


def _generate_selection_plots(chromosome_factory):
    initial_population_size = 100

    first_generation = chromosome_factory.generate_random_population(initial_population_size)

    couple_selection_method = ['rand_couple_selection', rand_couple_selection]

    for selection_method in SELECTION_METHODS.items():
        figures = []
        for crossover_method in CROSSOVER_METHODS.items():
            print(f'\tGenerating {selection_method[0]} {crossover_method[0]} plot')

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
                            MutationMethodConfig(random_mutation, probability=0.005),
                            SelectionMethodConfig(selection_method[1], threshold=0.7, k=0.5, T0=200, Tc=10,
                                                  truncation_size=initial_population_size - 2))

            stats = genetic_algorithm(first_generation, chromosome_factory, config.couple_selection_method,
                                      config.crossover_method_config.method, config.mutation_method_config.method,
                                      config.selection_method_config.method, config)

            x = range(0, config.end_condition_config.acceptable_solution_generation_count + 20)
            figures.append(
                go.Scatter(
                    name=f'{crossover_method[0].replace("_", " ").title()}',
                    x=list(x),
                    y=list(map(lambda c: c.fitness, stats.best_solutions)),
                ),
            )

        fig = go.Figure(
            figures,
            {
                'title': f'{selection_method[0].replace("_", " ").title()}',
                'xaxis_title': "Generation",
                'yaxis_title': "Fitness",
                'legend': dict(
                    bgcolor='rgba(0,0,0,0)',
                    y=0,
                    x=0.8
                )
            }
        )

        fig.update_xaxes(type="log")
        fig.show()


if __name__ == '__main__':
    INITIAL_POPULATION_SIZES = [100]
    MUTATION_PROBABILITY = [0.005]

    chromosome_factory = ChromosomeFactory(_get_knapsack_data("../" + DEFAULT_DATA_FILE), benefit_weight_ratio)

    # print('Generating selection plots')
    # _generate_selection_plots(chromosome_factory)
    #
    # print('Generating Boltzmann plots')
    # _generate_boltzmann_plots(chromosome_factory, 100, 0.005)
    #
    # print('Generating Population Plots')
    # _generate_population_plots(chromosome_factory)
    #
    # print('Generating Tournament plots')
    # _generate_tournament_plots(chromosome_factory, 100, 0.005)
    #
    # print('Generating overlapped plots')
    # _generate_overlapped_plots(chromosome_factory, 100, 0.005, ['simple_crossover', simple_crossover])
    # _generate_overlapped_plots(chromosome_factory, 100, 0.005, ['multiple_crossover', multiple_crossover])
    # _generate_overlapped_plots(chromosome_factory, 100, 0.005, ['uniform_crossover', uniform_crossover])

    #print('Generating Population plots')
    #_generate_population_plots(chromosome_factory)

    #
    # print('Generating Population plots')
    # _generate_population_plots(chromosome_factory, [10, 50, 100], 0.005)

    # print('Generating Couple Selection plots')
    # _generate_population_plots(chromosome_factory, 100, 0.005)
