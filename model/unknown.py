import random
from model.ion import Ion
from model.element_db import ELEMENTS


class UnknownSample:
    def __init__(self, mixture=False):
        self.mixture = mixture
        self.components = self._generate()

    def _generate(self):
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
