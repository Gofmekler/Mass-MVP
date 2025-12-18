import random
from model.ion import Ion
from model.element_db import ELEMENTS


class UnknownSample:
    def __init__(self, mixture=False, specific_element=None):
        self.mixture = mixture
        self.specific_element = specific_element
        self.components = self._generate()

    def _generate(self):
        if self.specific_element:
            # Используем заданный элемент
            return [(Ion(ELEMENTS[self.specific_element]), 1.0)]

        keys = list(ELEMENTS.keys())
        if not self.mixture:
            k = random.choice(keys)
            return [(Ion(ELEMENTS[k]), 1.0)]

        picks = random.sample(keys, 2)
        return [
            (Ion(ELEMENTS[picks[0]]), 0.6),
            (Ion(ELEMENTS[picks[1]]), 0.4)
        ]

    def ions(self):
        return self.components