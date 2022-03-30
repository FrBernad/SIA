from typing import Dict, Callable

from algorithms.end_conditions import EndConditionStats
from algorithms.mutation import random_mutation

DEFAULT_GENERATIONS_COUNT = 500
DEFAULT_TIME = 5 * 60
DEFAULT_FITNESS_CONSECUTIVE_GENERATIONS = 10
DEFAULT_PERCENTAGE = 0.8
DEFAULT_STRUCTURE_CONSECUTIVE_GENERATIONS = 10


class EndConditionConfig:
    def __init__(self, **config):
        self.generations_count = config.get('generations_count')
        self.time = config.get('time')
        self.percentage = config.get('percentage')
        self.acceptable_solution_generation_count = config.get('acceptable_solution_generation_count')
        self.fitness_consecutive_generations = config.get('fitness_consecutive_generations')
        self.fitness_min_generations = config.get('fitness_min_generations')
        self.structure_consecutive_generations = config.get('structure_consecutive_generations')
        self.structure_min_generations = config.get('structure_min_generations')
        self.stats = EndConditionStats()


class SelectionMethodConfig:
    def __init__(self, method, **config):
        self.method = method
        self.truncation_size = config.get('truncation_size')
        self.k = config.get('k')
        self.T0 = config.get('T0')
        self.Tc = config.get('Tc')


class MutationsMethodConfig:
    def __init__(self, method, **config):
        self.method = method
        self.probability = config.get('probability')


class CrossoverMethodConfig:
    def __init__(self, method, **config):
        self.method = method
        self.n = config.get('n')


from algorithms.couple_selection import COUPLE_SELECTION_METHODS
from algorithms.crossover import CROSSOVER_METHODS
from algorithms.fitness_functions import FITNESS_FUNCTIONS
from algorithms.selection import SELECTION_METHODS


class Config:
    def __init__(self, initial_population_size, endConditionConfig: EndConditionConfig, fitness_function: Callable,
                 couple_selection_method: Callable, crossover_method_config: CrossoverMethodConfig,
                 mutation_method_config: MutationsMethodConfig, selection_method_config: SelectionMethodConfig):

        self.initial_population_size = initial_population_size
        self.endConditionConfig = endConditionConfig
        self.fitness_function = fitness_function
        self.couple_selection_method = couple_selection_method
        self.crossover_method_config = crossover_method_config
        self.mutation_method_config = mutation_method_config
        self.selection_method_config = selection_method_config

    @staticmethod
    def generate(config_dict: Dict) -> 'Config':
        init_population = Config._get_initial_population_size(config_dict['initial_population'])
        return Config(
            init_population,
            Config._get_end_condition_config(config_dict['end_condition']),
            Config._get_fitness_function(config_dict['fitness_function']),
            Config._get_couple_selection_method(config_dict['couple_selection']),
            Config._get_crossover_method_config(config_dict['crossover'], init_population),
            Config._get_mutation_method_config(config_dict['mutation_probability']),
            Config._get_selection_method_config(config_dict['selection'])
        )

    @staticmethod
    def _get_end_condition_config(end_condition: Dict) -> EndConditionConfig:
        generations_count = DEFAULT_GENERATIONS_COUNT
        try:
            generations_count = int(end_condition.get('generations_count'))
            if generations_count < 500:
                raise InvalidEndCondition()

        except ValueError:
            pass

        acceptable_solution_generation_count = DEFAULT_GENERATIONS_COUNT
        try:
            acceptable_solution_generation_count = int(end_condition.get('acceptable_solution_generation_count'))
            if acceptable_solution_generation_count < 500:
                raise InvalidEndCondition()

        except ValueError:
            pass

        time = DEFAULT_TIME
        try:
            time = int(end_condition.get('time'))
            if time <= 0:
                raise InvalidEndCondition()

        except ValueError:
            pass

        fitness_consecutive_generations = DEFAULT_FITNESS_CONSECUTIVE_GENERATIONS
        try:
            fitness_consecutive_generations = int(end_condition.get('fitness').get('generations'))
            if fitness_consecutive_generations <= 0:
                raise InvalidEndCondition()

        except ValueError:
            pass

        fitness_min_generations = DEFAULT_GENERATIONS_COUNT
        try:
            fitness_min_generations = int(end_condition.get('fitness').get('min_generations'))
            if fitness_min_generations < 500:
                raise InvalidEndCondition()

        except ValueError:
            pass

        percentage = DEFAULT_PERCENTAGE
        try:
            percentage = float(end_condition.get('structure').get('percentage'))
            if percentage <= 0 or percentage >= 1:
                raise InvalidEndCondition()

        except (ValueError, TypeError):
            pass

        structure_consecutive_generations = DEFAULT_STRUCTURE_CONSECUTIVE_GENERATIONS
        try:
            structure_consecutive_generations = int(end_condition.get('structure').get('generations'))
            if structure_consecutive_generations <= 0:
                raise InvalidEndCondition()

        except (ValueError, TypeError):
            pass

        structure_min_generations = DEFAULT_GENERATIONS_COUNT
        try:
            structure_min_generations = int(end_condition.get('structure').get('min_generations'))
            if structure_min_generations < 500:
                raise InvalidEndCondition()

        except (ValueError, TypeError):
            pass

        return EndConditionConfig(generations_count=generations_count, percentage=percentage,
                                  time=time, fitness_consecutive_generations=fitness_consecutive_generations,
                                  structure_consecutive_generations=structure_consecutive_generations,
                                  acceptable_solution_generation_count=acceptable_solution_generation_count,
                                  fitness_min_generations=fitness_min_generations,
                                  structure_min_generations=structure_min_generations)

    @staticmethod
    def _get_fitness_function(fitness_function: str) -> Callable:
        if not fitness_function or fitness_function not in FITNESS_FUNCTIONS.keys():
            raise InvalidFitnessFunction()

        return FITNESS_FUNCTIONS[fitness_function]

    @staticmethod
    def _get_initial_population_size(initial_population: str) -> int:
        try:
            size = int(initial_population)
            if size <= 0:
                raise InvalidEndCondition()

            return size
        except (ValueError, TypeError):
            raise InvalidInitialPopulationSize()

    @staticmethod
    def _get_couple_selection_method(couple_selection: str) -> Callable:
        if not couple_selection or couple_selection not in COUPLE_SELECTION_METHODS.keys():
            raise InvalidCoupleSelectionMethod()

        return COUPLE_SELECTION_METHODS[couple_selection]

    @staticmethod
    def _get_crossover_method_config(crossover: Dict, initial_population: int) -> CrossoverMethodConfig:
        crossover_type = crossover.get('type')

        if not crossover_type or crossover_type not in CROSSOVER_METHODS.keys():
            raise InvalidCrossoverMethod()

        if crossover_type == 'multiple_crossover':
            try:
                n = int(crossover.get(crossover_type).get('n'))
                if n <= 0 or n >= initial_population:
                    raise InvalidCrossoverMethod()

                return CrossoverMethodConfig(
                    CROSSOVER_METHODS[crossover_type],
                    n=n,
                )

            except (ValueError, TypeError):
                raise InvalidCrossoverMethod()

        else:
            return CrossoverMethodConfig(
                CROSSOVER_METHODS[crossover_type]
            )

    @staticmethod
    def _get_mutation_method_config(mutation_probability: str) -> MutationsMethodConfig:
        if not mutation_probability:
            raise InvalidCrossoverMethod()

        try:
            probability = float(mutation_probability)

            if probability <= 0 or probability >= 1:
                raise InvalidCrossoverMethod()

            return MutationsMethodConfig(random_mutation, probability=probability)

        except (ValueError, TypeError):
            raise InvalidCrossoverMethod()

    @staticmethod
    def _get_selection_method_config(selection: Dict) -> SelectionMethodConfig:
        selection_method_type = selection.get('type')
        if not selection_method_type or selection_method_type not in SELECTION_METHODS.keys():
            raise InvalidEndCondition()

        if selection_method_type == 'truncated_selection':
            try:
                truncation_size = int(selection.get(selection_method_type).get('truncation_size'))

                if truncation_size <= 0:
                    raise InvalidSelectionMethod()

                return SelectionMethodConfig(
                    SELECTION_METHODS.get(selection_method_type),
                    truncation_size=truncation_size
                )

            except (ValueError, TypeError):
                raise InvalidCrossoverMethod()

        elif selection_method_type == 'boltzmann_selection':
            try:
                T0 = float(selection.get(selection_method_type).get('T0'))
                Tc = float(selection.get(selection_method_type).get('Tc'))
                k = int(selection.get(selection_method_type).get('k'))

                if T0 < Tc or Tc <= 0 or Tc > T0 or k <= 0:
                    raise InvalidSelectionMethod()

                return SelectionMethodConfig(
                    SELECTION_METHODS.get(selection_method_type),
                    T0=T0,
                    Tc=Tc,
                    k=k
                )
            except (ValueError, TypeError):
                raise InvalidCrossoverMethod()

        else:
            return SelectionMethodConfig(
                SELECTION_METHODS.get(selection_method_type),
            )


class ConfigException(Exception):
    def __init__(self, message):
        super().__init__(message)


class InvalidEndCondition(ConfigException):

    def __init__(self, message='Invalid end condition config'):
        self.message = message
        super().__init__(self.message)


class InvalidFitnessFunction(ConfigException):

    def __init__(self, message='Invalid fitness function'):
        self.message = message
        super().__init__(self.message)


class InvalidInitialPopulationSize(ConfigException):

    def __init__(self, message='Invalid initial population size'):
        self.message = message
        super().__init__(self.message)


class InvalidCoupleSelectionMethod(ConfigException):

    def __init__(self, message='Invalid couple selection method config'):
        self.message = message
        super().__init__(self.message)


class InvalidCrossoverMethod(ConfigException):

    def __init__(self, message='Invalid crossover method config'):
        self.message = message
        super().__init__(self.message)


class InvalidSelectionMethod(ConfigException):

    def __init__(self, message='Invalid selection method config'):
        self.message = message
        super().__init__(self.message)
