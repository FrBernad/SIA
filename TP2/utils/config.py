from typing import Dict, Callable

from algorithms.mutation import random_mutation


class EndConditionConfig:
    def __init__(self, condition, **config):
        self.condition = condition
        self.generations_count = config.get('generations_count')
        self.time = config.get('time')
        self.percentage = config.get('percentage')
        self.generations = config.get('generations')


class SelectionMethodConfig:
    def __init__(self, method, **config):
        self.method = method
        self.k = config.get('k')


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
from algorithms.end_conditions import END_CONDITIONS
from algorithms.fitness_functions import FITNESS_FUNCTIONS
from algorithms.selection import SELECTION_METHODS


class Config:
    def __init__(self, endConditionConfig: EndConditionConfig, fitness_function: Callable,
                 couple_selection_method: Callable, crossover_method_config: CrossoverMethodConfig,
                 mutation_method_config: MutationsMethodConfig, selection_method_config: SelectionMethodConfig):

        self.endConditionConfig = endConditionConfig
        self.fitness_function = fitness_function
        self.couple_selection_method = couple_selection_method
        self.crossover_method_config = crossover_method_config
        self.mutation_method_config = mutation_method_config
        self.selection_method_config = selection_method_config

    @staticmethod
    def generate(config_dict: Dict) -> 'Config':
        return Config(
            Config._get_end_condition_config(config_dict['end_condition']),
            Config._get_fitness_function(config_dict['fitness_function']),
            Config._get_couple_selection_method(config_dict['couple_selection']),
            Config._get_crossover_method_config(config_dict['crossover']),
            Config._get_mutation_method_config(config_dict['mutation_probability']),
            Config._get_selection_method_config(config_dict['selection'])
        )

    @staticmethod
    def _get_end_condition_config(end_condition: Dict) -> EndConditionConfig:
        end_cond_type = end_condition.get("type")
        if not end_cond_type or end_cond_type not in END_CONDITIONS.keys():
            raise InvalidEndCondition()

        elif end_cond_type == 'generations_count_condition':
            try:
                generations_count = int(end_condition.get("generations_count"))
                if generations_count <= 0:
                    raise InvalidEndCondition()

                return EndConditionConfig(
                    END_CONDITIONS.get(end_cond_type),
                    generations_count=generations_count,
                )

            except ValueError:
                raise InvalidEndCondition()

        elif end_cond_type == 'time_condition':
            try:
                time = int(end_condition.get("time"))
                if time <= 0:
                    raise InvalidEndCondition()

                return EndConditionConfig(
                    END_CONDITIONS.get(end_cond_type),
                    time=time,
                )

            except ValueError:
                raise InvalidEndCondition()

        else:
            try:
                percentage = float(end_condition.get(end_cond_type).get("percentage"))
                if percentage <= 0 or percentage >= 1:
                    raise InvalidEndCondition()

                generations = int(end_condition.get(end_cond_type).get("generations"))
                if generations <= 0:
                    raise InvalidEndCondition()

                return EndConditionConfig(
                    END_CONDITIONS.get(end_cond_type),
                    percentage=percentage,
                    generations=generations
                )

            except ValueError:
                raise InvalidEndCondition()

    @staticmethod
    def _get_fitness_function(fitness_function: str) -> Callable:
        if not fitness_function or fitness_function not in FITNESS_FUNCTIONS.keys():
            raise InvalidFitnessFunction()

        return FITNESS_FUNCTIONS[fitness_function]

    @staticmethod
    def _get_couple_selection_method(couple_selection: str) -> Callable:
        if not couple_selection or couple_selection not in COUPLE_SELECTION_METHODS.keys():
            raise InvalidCoupleSelectionMethod()

        return COUPLE_SELECTION_METHODS[couple_selection]

    @staticmethod
    def _get_crossover_method_config(crossover: Dict) -> CrossoverMethodConfig:
        crossover_type = crossover.get("type")

        if not crossover_type or crossover_type not in CROSSOVER_METHODS.keys():
            raise InvalidCrossoverMethod()

        if crossover_type == 'multiple_crossover':
            try:
                n = int(crossover.get(crossover_type).get('n'))
                if n <= 0:
                    raise InvalidCrossoverMethod()

                return CrossoverMethodConfig(
                    CROSSOVER_METHODS[crossover_type],
                    n=n,
                )

            except ValueError:
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

        except ValueError:
            raise InvalidCrossoverMethod()

    @staticmethod
    def _get_selection_method_config(selection: Dict) -> SelectionMethodConfig:
        selection_method_type = selection.get("type")
        if not selection_method_type or selection_method_type not in SELECTION_METHODS.keys():
            raise InvalidEndCondition()

        if selection_method_type == 'truncated_selection':
            try:
                k = int(selection.get(selection_method_type).get('k'))

                if k <= 0:
                    raise InvalidSelectionMethod()

                return SelectionMethodConfig(
                    SELECTION_METHODS.get(selection_method_type),
                    k=k,
                )

            except ValueError:
                raise InvalidCrossoverMethod()

        else:
            return SelectionMethodConfig(
                SELECTION_METHODS.get(selection_method_type),
            )


class ConfigException(Exception):
    def __init__(self, message):
        super().__init__(message)


class InvalidEndCondition(ConfigException):

    def __init__(self, message="Invalid end condition"):
        self.message = message
        super().__init__(self.message)


class InvalidFitnessFunction(ConfigException):

    def __init__(self, message="Invalid fitness function"):
        self.message = message
        super().__init__(self.message)


class InvalidCoupleSelectionMethod(ConfigException):

    def __init__(self, message="Invalid couple selection method"):
        self.message = message
        super().__init__(self.message)


class InvalidCrossoverMethod(ConfigException):

    def __init__(self, message="Invalid crossover method"):
        self.message = message
        super().__init__(self.message)


class InvalidSelectionMethod(ConfigException):

    def __init__(self, message="Invalid selection method"):
        self.message = message
        super().__init__(self.message)
