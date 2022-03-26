from utils.config import EndConditionConfig


def generations_count_condition(config: EndConditionConfig) -> bool:
    pass


def time_condition(config: EndConditionConfig) -> bool:
    pass


def structure_condition(config: EndConditionConfig) -> bool:
    pass


def fitness_condition(config: EndConditionConfig) -> bool:
    pass


END_CONDITIONS = {
    'generations_count_condition': generations_count_condition,
    'time_condition': time_condition,
    'structure_condition': structure_condition,
    'fitness_condition': fitness_condition
}
