from algorithms.end_conditions import Stats
from utils.config import Config


def generate_solution_yaml(stats: Stats, config: Config):
    sol = stats.best_solutions[-1]

    solution = {
        'initial_population': config.initial_population_size,
        'end_condition': stats.end_condition.value,
        'fitness_function': config.config_dict['fitness_function'],
        'couple_selection': config.config_dict['couple_selection'],
        'crossover': config.config_dict['crossover']['type'],
        'mutation_probability': config.mutation_method_config.probability,
        'selection': config.config_dict['selection']['type'],
        'solution': {
            'genes': ''.join(list(map(lambda gen: '1' if gen else '0', sol.genes))),
            'fitness': sol.fitness,
            'weight': sol.weight,
            'benefit': sol.benefit
        }
    }

    return solution
