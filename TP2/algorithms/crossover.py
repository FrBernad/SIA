from random import sample
from typing import Tuple

from utils.backpack import Chromosome


def simple_crossover(couple: Tuple[Chromosome, Chromosome]) -> Tuple[Chromosome, Chromosome]:
    return multiple_crossover(couple, 1)


def multiple_crossover(
        couple: Tuple[Chromosome, Chromosome],
        crossover_amount: int
) -> Tuple[Chromosome, Chromosome]:
    crossover_points = sample(range(1, len(couple[0]) - 1), crossover_amount)
    crossover_points.append(len(couple[0]))
    crossover_points.sort()
    s1 = list(couple[0])
    s2 = list(couple[1])

    for i in range(len(crossover_points)):
        if i % 2 != 0:
            swap_elements(crossover_points[i - 1], crossover_points[i], s1, s2)

    return tuple(s1), tuple(s2)


def swap_elements(start_point: int, end_point: int, list1: list, list2: list):
    list1[start_point:end_point], list2[start_point:end_point] = \
        list2[start_point:end_point], list1[start_point:end_point]
