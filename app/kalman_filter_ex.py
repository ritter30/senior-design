import numpy as np

def kalman_filter(x, p, z, r, init=True):
    """
    Initialization step of Kalman Filter.

    Params:
        x_0: The Initial System State
        p_0: The Initial State Variance

    Returns: ...
    """

    if init:
        x_0 = x[0]
        p_0 = p[0]
