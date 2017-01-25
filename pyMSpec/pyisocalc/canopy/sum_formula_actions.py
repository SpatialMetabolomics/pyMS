from ..periodic_table import periodic_table
from collections import Counter

class InvalidFormulaError(Exception):
    def __init__(self, desc):
        self.value = desc

    def __str__(self):
        return repr(self.value)

class Actions(object):
    def make_number(self, input, start, end, elements):
        return int(input[start:end])

    def make_element(self, input, start, end, elements):
        element = input[start:end]
        if element not in periodic_table:
            raise InvalidFormulaError("element {} is not in periodic table".format(element))
        return element

    def multiply_by_sign(self, input, start, end, elements):
        sign, complex = elements
        if sign.text == '+':
            return complex
        counts = Counter()
        for el in complex:
            counts[el] = -complex[el]
        return counts

    def make_simple_fragment(self, input, start, end, elements):
        n = 1
        if isinstance(elements[1], int):
            n = elements[1]
        counts = Counter({elements[0]: n})
        return counts

    def combine_fragments(self, input, start, end, elements):
        counts = Counter()
        for el in elements:
            counts.update(el)
        return counts

    def expand_fragment(self, input, start, end, elements):
        fragment = elements[1]
        num = elements[3]
        if not isinstance(num, int):
            num = 1
        counts = Counter()
        for el in fragment:
            counts[el] = fragment[el] * num
        return counts

    def expand_complex(self, input, start, end, elements):
        num = elements[0]
        if not isinstance(num, int):
            num = 1
        counts = Counter()
        for el in elements[1]:
            for k in el:
                counts[k] += el[k] * num
        return counts

    def make_formula(self, input, start, end, elements):
        counts = Counter()
        counts.update(elements[0])
        for child in elements[1]:
            counts.update(child.complex)
        for child in elements[2]:
            counts.update(child)
        counts = {k: counts[k] for k in counts if counts[k] != 0}

        if not counts:
            raise InvalidFormulaError("the formula is empty")
        for k in counts:
            if counts[k] < 0:
                raise InvalidFormulaError("element {} occurs less than zero times".format(k))

        return counts
