from random import randint
from typing import Tuple

import random

from utils.backpack import Chromosome




def main():
    chromosomes = (1, 0, 1, 0, 1, 0), (0, 1, 0, 1, 0, 1)
    multiple_crossover(chromosomes, 2)


def simple_crossover(couple: Tuple[Chromosome, Chromosome]) -> Tuple[Chromosome, Chromosome]:
    random_point = randint(1, len(couple[0]) - 2)

    start1 = couple[0][0:random_point]
    start2 = couple[1][0:random_point]
    end1 = couple[0][random_point:len(couple[0])]
    end2 = couple[1][random_point:len(couple[1])]

    s1 = tuple([*start1, *end2])
    s2 = tuple([*start2, *end1])

    return s1, s2


def multiple_crossover(couple: Tuple[Chromosome, Chromosome], crossover_amount: int):
    try:
        crossover_points = random.sample(range(1, len(couple[0]) - 2), crossover_amount)
    except ValueError:
        print('Sample size exceeded population size')

    crossover_points.append(0)

    crossover_points.sort()
    print(crossover_points)

    s1 = list(couple[0])
    s2 = list(couple[1])

    for i in range(crossover_amount + 1):

        if i % 2 != 0:

            first_point = crossover_points[i]

            if i + 1 >= crossover_amount:
                last_point = len(couple[0])
            else:
                last_point = crossover_points[i + 1]

            swap_elements(first_point, last_point, s1, s2)

    print(s1)
    print(s2)
    return tuple(s1, s2)


def swap_elements(first_point: int, last_point: int, list1: list, list2: list):
    i = first_point

    while i < last_point:
        list1[i], list2[i] = list2[i], list1[i]
        i += 1


if __name__ == '__main__':
    main()
