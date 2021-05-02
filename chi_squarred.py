import scipy.stats as stat

import numpy as np


def chi_squarred(observed, theorical):
    kr = np.sum(((observed - theorical) ** 2) / theorical)
    critical = stat.chi2.ppf(q=0.95, df=len(observed) - 1)
    return kr <= critical, kr, critical


def chi_squared_uniform(observed):
    proba = 1 / len(observed)
    N = sum(observed)
    theorical = np.array([N * proba for _ in range(len(observed))])
    return chi_squarred(observed, theorical)


def chi_squarred_poker(observed, N):  # P(all_diff) > P(one_pair) > P(triplet) > P(two_pair) > P(quadr)
    probas = np.array([0.504, 0.432, 0.036, 0.027, 0.001])
    theorical = probas * N
    return chi_squarred(observed, theorical)


def split_in_groups(data):
    data = np.sort(data)
    i = 0
    count = 0
    res = []
    for d in data:
        if i <= d < (i + 0.1):
            count += 1
        else:
            i += 0.1
            res.append(count)
            count = 0
    if count != 0 and len(res) == 0:
        res.append(count)
    return np.array(res)


def chi_squared_continuous(data):
    observed = split_in_groups(data)
    N = len(data)
    proba = 1 / len(observed)
    theorical = np.array([N * proba for _ in range(len(observed))])
    return chi_squarred(observed, theorical)
