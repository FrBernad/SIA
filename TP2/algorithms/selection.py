import itertools

from utils.backpack import Population


def elite_selection(population: Population, selection_size: int) -> Population:
    #falta implementar para que el sorted sortee por fitness
    return set(itertools.islice(sorted(population), selection_size))
