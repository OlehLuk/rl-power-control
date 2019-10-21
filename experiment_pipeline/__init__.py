from .ps_env import *
from .pipeline import *


def mse(a, b):
    return np.mean(np.power(np.array(a)-np.array(b), 2))


