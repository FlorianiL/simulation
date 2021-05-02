import numpy as np

from chi_squarred import chi_squarred


def gap_test(data, proba, a, b):
    intervals = []
    length_series_not_in = 1
    total_gaps = 0
    sequence_length = 0
    for nb in data:
        if a <= nb < b:
            # nb in interval, with probability `probability`
            intervals.append(length_series_not_in)
            length_series_not_in = 1
        else:
            # nb not in interval
            length_series_not_in += 1
            total_gaps += 1
        sequence_length += 1
    labels, observed = np.unique(np.array(intervals), return_counts=True)
    observed = complete_labels(labels, observed)
    theorical = np.array([(1 - proba) ** (n + 1) for n in range(max(labels))]) * sum(observed)
    return chi_squarred(observed, theorical)


def gap_test_discrete(data, a: int = 0, b: int = 5, total_numbers: int = 10):
    assert a < b
    proba = (b - a) / total_numbers  # => proba 1/2 d'être marqué
    return gap_test(data, proba, a, b)


def complete_labels(labels, observed):
    res_labels = []
    res_observed = []
    n = 0
    j = 0
    for expected in range(1, np.max(labels) + 1):
        label = labels[j - n]
        if expected != label:
            res_labels.append(expected)
            res_observed.append(0)
            n += 1
        else:
            res_labels.append(label)
            res_observed.append(observed[j - n])
            j += 1
    return res_observed


def gap_test_continue(data, a = 0.0, b = 0.5):
    assert a < b
    proba = b - a  # => proba 1/2 d'être marqué
    return gap_test(data, proba, a, b)
