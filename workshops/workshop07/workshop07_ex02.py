"""
Workshop 7: Overlapping generations (OLG)
Exercise 2: Transition dynamics for general CRRA preferences

This module implements the solution for transition dynamics of the OLG model
with arbitrary CRRA parameters using root-finding in each period.
"""

from pathlib import Path
import numpy as np
from scipy.optimize import root_scalar
from workshop07_ex01 import (
    Parameters,
    SteadyState,
    compute_prices,
    initialize_sim,
    compute_steady_state,
    print_steady_state,
    plot_simulation,
)


def euler_err(s, w, z_next, par: Parameters):
    """
    Compute the euler equation error for a given savings rate.

    Parameters
    ----------
    s : float
        Guess for the savings rate
    w : float
        Current wage rate
    z_next : float
        Next period's TFP
    par : Parameters
        Parameters for the given problem

    Returns
    -------
    float
        Euler equation error
    """

    # TODO: Implement the algorithm to compute the Euler equation error 
    # outlined in the exercise.


def simulate_olg_crra(z_series, eq: SteadyState):
    """
    Simulate the transition dynamics of the OLG model for arbitrary RRA values.

    Uses root-finding in each period to determine the optimal savings rate.

    Parameters
    ----------
    z_series : np.ndarray
        Time series of TFP values (length T+2)
    eq : SteadyState
        Initial steady-state equilibrium

    Returns
    -------
    Simulation
        A dataclass containing the simulated transition path of key variables.
    """

    # TODO: Implement simulation similar to the one in exercise 1, but use 
    # root-finding in each period to solve for the optimal savings rate.


if __name__ == '__main__':
    """
    Main function to run the simulation for the OLG model with general CRRA preferences.
    """
    
    # TODO: Run and plot transition path for gamma = 1
    # TODO: Run and plot transition path for gamma = 5