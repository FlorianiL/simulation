import math

import numpy as np


def kolmogorov_smirnov(numbers):
    np.sort(numbers)
    n = len(numbers)
    distance = np.max(np.array([np.abs((i/n)-numbers[i]) for i in range(len(numbers))]))
    critical = (1.358/math.sqrt(n))
    return distance < critical, distance, critical
