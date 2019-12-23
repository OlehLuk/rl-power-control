import numpy as np


def get_bins(lower_bound, upper_bound, n_bins):
    """
    Given bounds for environment state variable splits it into n_bins number of bins,
    taking into account possible values outside the bounds.

    :param lower_bound: lower bound for variable describing state space
    :param upper_bound: upper bound for variable describing state space
    :param n_bins: number of bins to receive
    :return: n_bins-1 values separating bins. I.e. the most left bin is opened from the left,
    the most right bin is open from the right.
    """
    return np.linspace(lower_bound, upper_bound, n_bins + 1)[1:-1]


def to_bin(value, bins):
    """
    Transforms actual state variable value into discretized one,
    by choosing the bin in variable space, where it belongs to.

    :param value: variable value
    :param bins: sequence of values separating variable space
    :return: number of bin variable belongs to. If it is smaller than lower_bound - 0.
    If it is bigger than the upper bound
    """
    return np.digitize(x=[value], bins=bins)[0]


def get_state_index_old(state_bins):
    """
    Transforms discretized environment state (represented as sequence of bin indexes) into an integer value.
    Value is composed by concatenating string representations of a state_bins.
    Received string is a valid integer, so it is converted to int.

    :param state_bins: sequence of integers that represents discretized environment state.
    Each integer is an index of bin, where corresponding variable belongs.
    :return: integer value corresponding to the environment state
    """
    state = int("".join(map(lambda state_bin: str(state_bin), state_bins)))
    return state


def get_state_index(state_bins, n_bins=10):
    """
    Transforms discretized environment state (represented as sequence of bin indexes) into an integer value.
    Value is composed by concatenating string representations of a state_bins.
    Received string is a valid integer, so it is converted to int.

    :param state_bins: sequence of integers that represents discretized environment state.
    Each integer is an index of bin, where corresponding variable belongs.
    :return: integer value corresponding to the environment state
    """
    state = sum([s_bin*n_bins**i for i, s_bin in enumerate(state_bins)])
    # state = int("".join(map(lambda state_bin: str(state_bin), state_bins)))
    return state