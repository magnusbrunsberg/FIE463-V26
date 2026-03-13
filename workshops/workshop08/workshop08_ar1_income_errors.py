"""
Workshop 8: Code for wealth dynamics with AR(1) income. 

THIS CODE CONTAINS BUGS and is used to demonstrate the use of a debugger.
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class Parameters:
    """
    Container to store model parameters
    """

    s: float = 0.75  # Exogenous savings rate
    R: float = 1.1  # Gross return
    sigma_eps: float = 0.1  # Conditional standard deviation of log income
    rho: float = 0.95  # Persistence of log income
    mu_y: float = -(sigma_eps**2.0) / 2.0 / (1 + rho)  # Intercept of log income


def simulate_wealth_ar1_income(par: Parameters, a0, T, N, rng=None):
    """
    Simulate the evolution of wealth over time if income follows an AR(1).

    The functions return T+1 values for each household, including the initial value.

    Parameters
    ----------
    par : Parameters
    a0 : float
        Initial wealth.
    T : int
        Number of time periods to simulate.
    N : int
        Number of households to simulate.
    rng : numpy.random.Generator, optional
        A random number generator instance.

    Returns
    -------
    a_sim : numpy.ndarray
        A (T+1, N) array where each column represents the simulated wealth path of a household.
    """

    if rng is None:
        rng = np.random.default_rng(seed=1234)

    # Random draws AR(1) innovations (epsilon)
    epsilon = rng.normal(loc=0, scale=par.sigma_eps, size=(T, 1))

    # Compute mean log income
    log_y_mean = par.mu_y / (1 - par.rho)

    # Assume that all households start with the same income
    log_y = np.full(N, fill_value=y_mean)

    a_sim = np.empty((T + 1, N))
    a_sim[0] = a0

    for t in range(1, T):
        # Savings out of beginning-of-period assets
        savings = par.s * a_sim[t]

        # Log income next period
        log_y_next = par.mu_y + par.rho * log_y + epsilon[t]

        # Next-period assets
        a_sim[t + 1] = par.R * savings + np.exp(log_y_next)

    return a_sim


if __name__ == '__main__':
    """
    Run all code for wealth dynamics with AR(1) income
    """

    # Create an instance of the Parameters class
    par = Parameters()

    # Mean of income
    y_mean = np.exp(par.mu_y / (1 - par.rho) + par.sigma_eps**2 / 2 / (1 - par.rho**2))

    # Mean of stationary distribution
    a_mean = y_mean / (1 - par.s * par.R)

    print(f'Mean income: {y_mean:.3f}')
    print(f'Mean wealth: {a_mean:.3f}')

    # --- Simulate wealth dynamics for 20 households ---

    # Initial wealth (identical for all households)
    a0 = 1.0
    # Number of periods to simulate
    T = 100
    # Number of households to simulate
    N = 20

    # Create RNG instance
    rng = np.random.default_rng(seed=1234)

    # Simulate the wealth paths (result is an array of shape (T+1, N))
    a_sim = simulate_wealth_ar1_income(par, a0, N, T, rng)

    # Mean of simulated time series
    a_sim_mean = np.mean(a_sim, axis=1)

    # Plot simulated wealth trajectories
    plt.figure(figsize=(7, 4))

    plt.plot(a_sim, alpha=0.75, lw=0.75)
    plt.xlabel('Period')
    plt.ylabel('Wealth')
    plt.title('Simulated wealth paths with AR(1) income')
    # Add unconditional mean of wealth distribution
    plt.axhline(a_mean, color='black', ls='--', lw=1, label='Stationary mean')
    # Add average of simulated wealth paths
    plt.plot(a_sim_mean, color='black', ls='-', lw=1.25, label='Mean of simulations')
    plt.legend(loc='lower right')
    plt.show()
