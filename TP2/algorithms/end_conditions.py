from typing import Dict
from utils.backpack import Population, Backpack


class EndConditionStats:
    def __init__(self):
        self.generation: Population
        self.is_acceptable = False
        self.generations_count = 0
        # FIXME: SACAR
        self.a = []
        self.start_time = 0
        self.fitness = {
            'consecutive': 0,
            'best': 0
        }
        self.structure = {
            'consecutive': 0
        }


from utils.config import EndConditionConfig


def check_end_condition(config: EndConditionConfig) -> bool:
    return _generations_count_condition(config) or _time_condition(config) or \
           _structure_condition(config) or _fitness_condition(config) or _acceptable_solution_condition(config)


def update_end_condition(config: EndConditionConfig, current_generation: Population, backpack: Backpack):
    config.stats.generation = current_generation
    config.stats.generations_count += 1
    _update_best_fitness(config, backpack, current_generation)
    config.stats.a.append(config.stats.fitness['best'])
    config.stats.is_acceptable = _check_acceptable_solution(config, backpack, current_generation)


def _generations_count_condition(config: EndConditionConfig) -> bool:
    condition = config.stats.generations_count >= config.generations_count
    if condition:
        print('Generations Count Condition')
    return condition


def _time_condition(config: EndConditionConfig) -> bool:
    pass


def _structure_condition(config: EndConditionConfig) -> bool:
    pass


def _fitness_condition(config: EndConditionConfig) -> bool:
    if config.stats.generations_count < config.fitness_min_generations:
        return False

    condition = config.stats.fitness['consecutive'] >= config.fitness_consecutive_generations
    if condition:
        print('Fitness Condition')
    return condition


def _update_best_fitness(config: EndConditionConfig, backpack: Backpack, generation: Population):
    if config.stats.generations_count < config.fitness_min_generations:
        return False

    best_fitness = 0

    for chromosome in generation:
        aux = backpack.calculate_fitness(chromosome)
        if aux > best_fitness:
            best_fitness = aux

    current_best = config.stats.fitness['best']
    if best_fitness != current_best:
        config.stats.fitness['best'] = best_fitness
        config.stats.fitness['consecutive'] = 0
    else:
        config.stats.fitness['consecutive'] += 1


def _check_acceptable_solution(config: EndConditionConfig, backpack: Backpack, generation: Population) -> bool:
    if config.stats.generations_count < config.acceptable_solution_generation_count:
        return False
    for chromosome in generation:
        if backpack.calculate_weight(chromosome) < backpack.max_weight:
            print("Acceptable Condition")
            return True

    return False


def _acceptable_solution_condition(config: EndConditionConfig) -> bool:
    return config.stats.is_acceptable
