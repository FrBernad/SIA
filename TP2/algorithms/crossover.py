import random
from typing import Tuple

from utils.backpack import Chromosome


def simple_crossover(couple: Tuple[Chromosome, Chromosome]) -> Tuple[Chromosome, Chromosome]:
    return multiple_crossover(couple, 1)


def multiple_crossover(couple: Tuple[Chromosome, Chromosome], crossover_amount: int) -> Tuple[Chromosome, Chromosome]:

    try:
        crossover_points = random.sample(range(0, len(couple[0]) - 1), crossover_amount)
    except ValueError:
        print('Sample size exceeded chromosome size!')
        return couple

    crossover_points.sort()

    s1 = list(couple[0])
    s2 = list(couple[1])

    for i in range(crossover_amount):

        if i % 2 != 0:
            last_point = crossover_points[i]
            swap_elements(crossover_points[i - 1], last_point, s1, s2)
        elif i + 1 >= crossover_amount and crossover_amount % 2 != 0:
            swap_elements(crossover_points[i], len(couple[0]) - 1, s1, s2)

    return tuple(s1), tuple(s2)


def swap_elements(first_point: int, last_point: int, list1: list, list2: list):
    for i in range(first_point + 1, last_point + 1):
        list1[i], list2[i] = list2[i], list1[i]
