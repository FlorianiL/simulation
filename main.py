from collections import Counter
from functools import lru_cache
import matplotlib.pyplot as plot
import scipy.stats as stat
import math


@lru_cache(5)
def count_numbers():
    cnt = Counter(e_numbers())
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


def construct_histo():
    keys, vals = count_numbers()
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
    print(f"Real : {real_repartition_fn}")
    print(f"Th : {th_repartition_fn}")
    max_ = max(map(lambda t: abs(t[0] - t[1]), zip(real_repartition_fn, th_repartition_fn)))
    crit = 1.358 / math.sqrt(N / len(labels_))
    return max_ < crit, max_, crit


if __name__ == '__main__':
    # construct_histo()
    labels, data = count_numbers()
    effect_th = sum(data) / 10
    print(f"Test Chi, Carré notre popotte magique : {chi_squared(data)}")
    print(f"Test Chi Carré de Numpy : {stat.chisquare(data, [effect_th for _ in range(10)], ddof=9)}")
    print(f"Test KS, notre popotte magique : {kolmogorov_smirnov(data, labels)}")
    print(list(map(lambda x: x / sum(data), data)))
    print(f"Test KS de Numpy : {stat.kstest(list(map(lambda x: x / sum(data), data)), stat.uniform.cdf)}")



