from numpy import array, zeros
from numpy.random import choice
from numpy.typing import NDArray
from data.fonts import FONTS_ARRAY


def parse_font(font_number: int, selection_amount: int) -> NDArray:
    characters = FONTS_ARRAY[font_number - 1]
    bin_characters = []
    for c in characters:
        bin_array = zeros((7, 5), dtype=int)
        for row in range(0, 7):
            current_row = c[row]
            for col in range(0, 5):
                bin_array[row][4 - col] = current_row & 1
                current_row >>= 1
        bin_characters.append(bin_array.flatten())
    return array(bin_characters)[choice(list(range(len(characters))), replace=False, size=selection_amount)]
