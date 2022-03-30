import time

from utils.backpack import Population, Backpack


class EndConditionStats:
    def __init__(self):
        self.is_acceptable = False
        self.generations_count = 0
        self.start_time = 0
        self.current_time = 0
        self.a = []
        # FIXME: SACAR
        self.fitness = {
            'consecutive': 0,
            'best': 0
        }
        self.structure = {
            'last': [],
            'current': [],
            'consecutive': 0
        }


from utils.config import EndConditionConfig


def check_end_conditions(config: EndConditionConfig) -> bool:
    return _generations_count_condition(config) or _time_condition(config) or \
           _structure_condition(config) or _fitness_condition(config) or _acceptable_solution_condition(config)


def _acceptable_solution_condition(config: EndConditionConfig) -> bool:
    return config.stats.is_acceptable


def _fitness_condition(config: EndConditionConfig) -> bool:
    if config.stats.generations_count < config.fitness_min_generations:
        return False

    condition = config.stats.fitness['consecutive'] >= config.fitness_consecutive_generations
    if condition:
        print('Fitness Condition')
    return condition


def _structure_condition(config: EndConditionConfig) -> bool:
    if config.stats.generations_count < config.structure_min_generations:
        return False

    condition = config.stats.structure['consecutive'] >= config.structure_consecutive_generations
    if condition:
        print('Structure Condition')
    return condition


def _time_condition(config: EndConditionConfig) -> bool:
    condition = (config.stats.current_time - config.stats.start_time) > config.time
    if condition:
        print('Time Condition')
    return condition


def _generations_count_condition(config: EndConditionConfig) -> bool:
    condition = config.stats.generations_count >= config.generations_count
    if condition:
        print('Generations Count Condition')
    return condition


def init_end_conditions(config: EndConditionConfig, current_generation: Population):
    config.stats.start_time = time.time()
    config.stats.current_time = config.stats.start_time
    config.stats.structure['current'] = current_generation


def update_end_conditions(config: EndConditionConfig, current_generation: Population, backpack: Backpack):
    config.stats.generation = current_generation
    config.stats.generations_count += 1
    _update_best_fitness(config, backpack, current_generation)
    _update_structure(config, current_generation)
    config.stats.current_time = time.time()
    config.stats.is_acceptable = _check_acceptable_solution(config, backpack, current_generation)
    config.stats.a.append(config.stats.fitness['best'])


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


def _update_structure(config: EndConditionConfig, current_population: Population):
    if config.stats.generations_count < config.structure_min_generations:
        return

    config.stats.structure['last'] = config.stats.structure['current']

    last_population = config.stats.structure['last']
    config.stats.structure['current'] = current_population

    similar_count = 0
    for chr in current_population:
        if chr in last_population:
            similar_count += 1

    similar_percentage = similar_count / len(current_population)

    if similar_percentage > config.percentage:
        config.stats.structure['consecutive'] += 1
    else:
        config.stats.structure['consecutive'] = 0
