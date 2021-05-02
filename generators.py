import math

import numpy as np

from main import open_file

e_numbers = np.array(list(open_file()))

class Generator1:

    def __init__(self, seed=0, nb_digits=10):
        self.seed = seed
        self.index = seed % len(e_numbers)
        self.nb_digits = nb_digits

    def random(self):
        digits = []
        for i in range(self.nb_digits):
            digit = e_numbers[self.index]
            digits.append(digit)
            self.index = (self.index + 1) % len(e_numbers)
        return float("0." + "".join(map(lambda x: str(x), digits)))


class Generator2:

    def __init__(self, seed=0, nb_digits=10):
        self.seed = seed
        self.index = seed % len(e_numbers)
        self.nb_digits = nb_digits

    def random(self):
        xyz = []
        for j in range(3):
            digits = []
            for k in range(self.nb_digits):
                digit = e_numbers[self.index]
                digits.append(digit)
                self.index = (self.index + 1) % len(e_numbers)
            xyz.append(float("0." + "".join(map(lambda x: str(x), digits))))
        return math.sqrt(sum([x ** 2 for x in xyz])) / math.sqrt(3)


class Generator3:

    def __init__(self, seed=0, precision=57):
        self.seed = seed
        self.index = seed % len(e_numbers)
        self.precision = precision  # IEE754 double has 57 bits of precision

    def random(self):
        bits = 0
        generated = 0
        while bits < self.precision:
            rn = e_numbers[self.index]

            if rn > 7:
                self.index = (self.index + 1) % len(e_numbers)
                continue

            three_bits = (
                rn & 1,
                (rn & 2) // 2,
                (rn & 4) // 4
            )

            for i in range(3):
                if bits >= self.precision:
                    break
                generated = (generated << 1) | three_bits[i]
                bits += 1

            self.index = (self.index + 1) % len(e_numbers)
        return abs(generated / 2 ** self.precision)
