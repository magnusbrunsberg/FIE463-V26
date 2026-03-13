"""
Lecture 8: Code for wealth dynamics with iid income
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from stats import gini


@dataclass
class Parameters:
    """
    Container to store model parameters
    """

    s: float = 0.75         # Exogenous savings rate
    R: float = 1.1          # Gross return
    sigma_y: float = 0.1    # Standard deviation of log income
    mu_y: float = -(sigma_y**2.0) / 2.0  # Mean of log income


def simulate_wealth_iid_income(par: Parameters, a0, T, N, rng=None):
    """
    Simulate the evolution of wealth over time when income is IID.

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

    Returns
    -------
    a_sim : numpy.ndarray
        A (T+1, N) array where each column represents the simulated wealth path of a household.
    """

    if rng is None:
        rng = np.random.default_rng(seed=1234)

    # Random draws of IID income
    log_y = rng.normal(loc=par.mu_y, scale=par.sigma_y, size=(T, N))

    # Income in levels
    y = np.exp(log_y)

    # Create array to store the simulated wealth paths
    a_sim = np.zeros((T + 1, N))

    # Set initial value (identical for all households)
    a_sim[0] = a0

    # Simulate wealth forward, one period at a time
    for t in range(T):
        # Savings out of beginning-of-period assets
        savings = par.s * a_sim[t]
        # Next-period assets
        a_sim[t + 1] = par.R * savings + y[t]

    return a_sim


def compute_wealth_mean(par):
    """
    Compute the mean of the stationary wealth distribution assuming iid income.

    Parameters
    ----------
    par : Parameters

    Returns
    -------
    float
        The mean of the stationary wealth distribution.
    """

    # Mean of income (in levels)
    # Follows from the mean formula for the log-normal distribution
    y_mean = np.exp(par.mu_y + par.sigma_y**2 / 2)

    # Mean of wealth
    a_mean = y_mean / (1 - par.s * par.R)

    return a_mean


def compute_wealth_var(par):
    """
    Compute the variance of the stationary wealth distribution assuming iid income.

    Parameters
    ----------
    par : Parameters

    Returns
    -------
    float
        The variance of the stationary wealth distribution.
    """

    # Variance of income (in levels)
    # Follows from the variance formula for the log-normal distribution
    y_var = np.exp(2 * par.mu_y + par.sigma_y**2) * (np.exp(par.sigma_y**2) - 1)

    # Variance of wealth
    a_var = y_var / (1 - (par.s * par.R) ** 2.0)

    return a_var


if __name__ == '__main__':
    """
    Run all code for IID income section
    """

    # Create an instance of the Parameters class
    par = Parameters()

    # Check for finite mean and variance of stationary distribution
    assert par.R * par.s < 1

    # Mean of stationary INCOME distribution
    y_mean = np.exp(par.mu_y + par.sigma_y**2 / 2)

    # Mean of stationary ASSET distribution
    a_mean = y_mean / (1 - par.s * par.R)

    print(f'Mean income: {y_mean:.3f}')
    print(f'Mean wealth: {a_mean:.3f}')

    # --- Simulate wealth trajectories for 20 households ---

    # Initial wealth (identical for all households)
    a0 = 1.0
    # Number of periods to simulate
    T = 100
    # Number of households to simulate
    N = 20

    # Create RNG instance
    rng = np.random.default_rng(seed=1234)

    # Simulate the wealth paths (result is an array of shape (T+1, N))
    a_sim = simulate_wealth_iid_income(par, a0, T, N, rng)

    # Mean of simulated time series
    a_sim_mean = np.mean(a_sim, axis=1)

    plt.figure(figsize=(7, 4))

    plt.plot(a_sim, alpha=0.75, lw=0.75)
    plt.xlabel('Period')
    plt.ylabel('Wealth')
    plt.title('Simulated wealth paths with IID income')
    # Add unconditional mean of wealth distribution
    plt.axhline(a_mean, color='black', ls='--', lw=1, label='Stationary mean')
    # Add average of simulated wealth paths
    plt.plot(a_sim_mean, color='black', ls='-', lw=1.25, label='Mean of simulations')
    plt.legend(loc='lower right')
    plt.show()

    # --- Simulate large number of households ---

    # Number of households
    N = 100_000
    # Number of periods to simulate
    T = 100

    # Create RNG instance
    rng = np.random.default_rng(seed=1234)

    # Simulate the wealth paths (result is an array of shape (T+1, N))
    a_sim = simulate_wealth_iid_income(par, a0, T, N, rng)

    # Cross-sectional mean of simulated time series
    a_sim_mean = np.mean(a_sim, axis=1)

    # Cross-sectional variance of simulated time series
    a_sim_var = np.var(a_sim, axis=1)

    # Compute analytical mean and variance
    a_mean_exact = compute_wealth_mean(par)
    a_var_exact = compute_wealth_var(par)

    # Plot cross-sectional mean and variance
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3), sharex=True)

    # Plot simulated vs. analytical mean
    ax1.axhline(a_mean_exact, color='black', ls='--', lw=1, label='Exact')
    ax1.plot(a_sim_mean, lw=1, label='Simulated')
    ax1.set_xlabel('Period')
    ax1.set_title('Cross-sectional mean of wealth')
    ax1.legend(loc='lower right')

    # Plot simulated vs. analytical variance
    ax2.axhline(a_var_exact, color='black', ls='--', lw=1, label='Exact')
    ax2.plot(a_sim_var, lw=1, label='Simulated')
    ax2.set_title('Cross-sectional variance of wealth')
    ax2.set_xlabel('Period')
    ax2.legend(loc='lower right')
    plt.show()

    # Select cross section from last simulated period
    last_cross_section = a_sim[-1]

    # Compute and print the Gini coefficient
    G = gini(last_cross_section)
    print(f'Wealth Gini coefficient: {G:.3f}')
