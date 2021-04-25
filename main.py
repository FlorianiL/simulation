from collections import Counter
from functools import lru_cache
import matplotlib.pyplot as plot
import scipy.stats as stat
import math
import numpy as np
from numpy import cumsum
import random


@lru_cache(5)
def count_numbers(data_):
    cnt = Counter(data_)
    keys = list(sorted(cnt.keys()))
    vals = [cnt[key] for key in keys]
    return keys, vals


def e_numbers():
    with open("exp.txt", "r") as e:
        for line in e:
            line = line.strip()
            if "." in line:
                line = line.split(".")[1]

            for c in line:
                yield int(c)


def construct_histo(keys, vals):
    fig, ax = plot.subplots()
    ax.bar(keys, vals)
    plot.show()


def chi_squared(data_):
    pi = 0.1
    N = sum(data_)
    kr = sum([((ni - N * pi) / (math.sqrt(N * pi))) ** 2 for ni in data_])
    crit = stat.chi2.ppf(q=0.95, df=9)
    return kr <= crit, kr, crit


def kolmogorov_smirnov(data_, labels_):
    assert len(data_) == len(labels_)
    N = sum(data_)
    real_repartition_fn: list = [sum(data_[:i + 1]) / N for i in labels_]
    th_repartition_fn: list = [0.1 * (i + 1) for i in labels_]
    max_ = max(map(lambda t: abs(t[0] - t[1]), zip(real_repartition_fn, th_repartition_fn)))
    crit = 1.358 / math.sqrt(N / len(labels_))
    return max_ < crit, max_, crit


def gap_test(number_sequence, a: int = 0, b: int = 5, total_numbers: int = 10):
    """
    :param number_sequence:
    :param a:
    :param b: b not included
    :return:
    """
    assert a < b
    probability = (b - a) / total_numbers  # => proba 1/2 d'être marqué
    intervals = []
    length_series_not_in_proba = 0
    total_for_I_0 = 0
    sequence_length = 0
    for nb in number_sequence:
        if a <= nb < b:
            # nb in interval, with probability `probability`
            intervals.append(length_series_not_in_proba)
            length_series_not_in_proba = 1
        else:
            # nb not in interval
            length_series_not_in_proba += 1
            total_for_I_0 += 1
        sequence_length += 1
    cnt = Counter(intervals)
    keys = list(sorted(cnt.keys()))
    observed = [total_for_I_0] + [cnt[key] for key in range(1, max(keys) + 1)]
    keys = [0] + keys
    observed = np.array(observed) / sequence_length

    # compare the observerd distribution to the theorical expected distribution
    expected = np.array([probability ** (n + 1) for n in range(max(keys) + 1)])
    expected = cumsum(expected)  # cumulative distribution function
    observed = cumsum(observed)
    kr = sum(((observed - expected) ** 2) / expected)
    crit = stat.chi2.ppf(q=0.05, df=len(observed) - 1)
    return kr <= crit, kr, crit


def generator(n: int = 1):
    numbers = list(e_numbers())
    for i in range(n):
        seed = numbers[i]
        a = len(numbers) - seed
        c = numbers[i + seed]
        x = (a * seed + c) % 10
        a = len(numbers) - x
        c = numbers[i + x]
        y = (a * seed + c) % 10
        a = len(numbers) - y
        c = numbers[i + y]
        z = (a * seed + c) % 10
        res = math.sqrt(x ** 2 + y ** 2 + z ** 2)
        while res > 1:
            res = res/10

        yield res


if __name__ == '__main__':
    # construct_histo(count_numbers())
    labels, data = count_numbers(e_numbers())
    print(e_numbers())
    effect_th = sum(data) / 10
    print(f"Test Chi, Carré notre popotte magique : {chi_squared(data)}")
    print(f"Test Chi Carré de Scipy : {stat.chisquare(data, [effect_th for _ in range(10)], ddof=9)}")
    print(f"Test KS, notre popotte magique : {kolmogorov_smirnov(data, labels)}")
    print(f"Test KS de Scipy : {stat.kstest(list(map(lambda x: x / sum(data), data)), stat.uniform.cdf)}")
    print(f"Test Gap, notre popotte magique : {gap_test(e_numbers())}")
    print(f"Test Gap, notre popotte magique : {gap_test([1 for _ in range(2000000)])}")
    print(f"Génération avec Python : {[random.random() for n in range(1000)]}")
    print(f"Notre générateur : {list(generator(1000))}")
