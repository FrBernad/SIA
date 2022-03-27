from typing import Dict
from utils.backpack import Population, Backpack


class EndConditionStats:
    def __init__(self):
        self.generations_count = 0
        self.fitness = {
            'consecutive': 0,
            'best': 0
        }
        self.start_time = 0
        self.structure = {
            'consecutive': 0
        }
        self.is_acceptable = False


from utils.config import EndConditionConfig


def check_end_condition(config: EndConditionConfig) -> bool:
    return _generations_count_condition(config) or _time_condition(config) \
           or _structure_condition(config) or _fitness_condition(config)


def update_end_condition(config: EndConditionConfig, current_generation: Population, backpack: Backpack):
    config.stats.generations_count += 1
    config.stats.is_acceptable = _check_acceptable_solution(backpack, current_generation)
    _update_best_fitness(config.stats.fitness, backpack, current_generation)


def _generations_count_condition(config: EndConditionConfig) -> bool:
    return config.stats.generations_count < config.generations_count


def _time_condition(config: EndConditionConfig) -> bool:
    pass


def _structure_condition(config: EndConditionConfig) -> bool:
    pass


def _fitness_condition(config: EndConditionConfig) -> bool:
    return config.stats.fitness['consecutive'] == config.stats.fitness_consecutive_generations


def _update_best_fitness(fitness: Dict, backpack: Backpack, generation: Population):
    best_fitness = backpack.calculate_fitness(generation[0])

    for chromosome in generation:
        aux = backpack.calculate_fitness(chromosome)
        if aux > best_fitness:
            best_fitness = aux

    current_best = fitness['best']
    if best_fitness != current_best:
        fitness['best'] = best_fitness
        fitness['consecutive'] = 0
    else:
        fitness['consecutive'] += 1


def _check_acceptable_solution(backpack: Backpack, generation: Population) -> bool:
    for chromosome in generation:
        if backpack.calculate_weight(chromosome) < backpack.max_weight:
            return True
    return False


def _acceptable_solution_condition(config: EndConditionConfig) -> bool:
    return config.stats.is_acceptable
