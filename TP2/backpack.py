class Backpack:
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
