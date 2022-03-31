from random import randint
from typing import Tuple, List


class Knapsack(object):

    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Knapsack, cls).__new__(cls)
        return cls.instance

    def __init__(self, max_capacity, max_weight, elements=None):
        if elements is None:
            elements = []
        self.max_weight = max_weight
        self.max_capacity = max_capacity
        self.elements = elements


class Element:
    def __init__(self, benefit, weight):
        self.weight = weight
        self.benefit = benefit
