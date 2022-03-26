from itertools import islice

from utils.backpack import Population


def elite_selection(population: Population, selection_size: int) -> Population:
    # falta implementar para que el sorted sortee por fitness
    return set(sorted(population)[0:selection_size])
