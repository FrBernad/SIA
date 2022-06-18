from numpy import array
from numpy.random import choice, uniform


def generate_noise(font: dict, percentage: float):
    characters = font.get('array').copy().astype(float)
    letters = array(font.get('letters'))

    applied_noise = round(len(characters[0]) * percentage)

    for c in characters:
        bits_to_change = choice(list(range(len(c))), replace=False, size=applied_noise)
        noise = uniform(low=-0.3, high=0.3, size=applied_noise)
        c[bits_to_change] += noise

    return dict(array=characters, letters=letters)
