"""
Lecture 8: Code for wealth dynamics with AR(1) income
"""

import numpy as np
import matplotlib.pyplot as plt
from stats import gini
from dataclasses import dataclass


@dataclass
class Parameters:
    """
    Container to store model parameters
    """
    s: float = 0.75                 # Exogenous savings rate
    R: float = 1.1                  # Gross return
    sigma_eps: float = 0.1          # Conditional standard deviation of log income
    rho: float = 0.95               # Persistence of log income
    mu_y: float = -sigma_eps**2.0/2.0/(1+rho)    # Intercept of log income


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
    epsilon = rng.normal(loc=0, scale=par.sigma_eps, size=(T, N))

    # Compute mean log income
    log_y_mean = par.mu_y/(1-par.rho)

    # Assume that all households start with the same income
    log_y = np.full(N, fill_value=log_y_mean)

    a_sim = np.zeros((T+1, N))
    a_sim[0] = a0

    for t in range(T):
        # Savings out of beginning-of-period assets
        savings = par.s * a_sim[t]

        # Log income next period
        log_y = par.mu_y + par.rho * log_y + epsilon[t]

        # Next-period assets
        a_sim[t+1] = par.R * savings + np.exp(log_y)

    return a_sim



def compute_wealth_mean(par):
    """
    Compute the mean of the stationary wealth distribution assuming income
    follows an AR(1) process.

    Parameters
    ----------
    par : Parameters

    Returns
    -------
    float
        The mean of the stationary wealth distribution.
    """

    # Unconditional mean of AR(1) log income
    log_y_mean = par.mu_y / (1 - par.rho)

    # Unconditional variance of AR(1) log income
    log_y_var = par.sigma_eps**2 / (1 - par.rho**2)

    # Mean of income (in levels)
    # Follows from the mean formula for the lognormal distribution
    y_mean = np.exp(log_y_mean + log_y_var/2)

    # Mean of wealth
    a_mean = y_mean / (1 - par.s * par.R)

    return a_mean


if __name__ == '__main__':
    """
    Run all code for wealth dynamics with AR(1) income
    """

    # Create an instance of the Parameters class
    par = Parameters()

    # Mean of income 
    y_mean = np.exp(par.mu_y/(1-par.rho) + par.sigma_eps**2/2/(1-par.rho**2))

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
    a_sim = simulate_wealth_ar1_income(par, a0, T, N, rng)

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

    # --- Simulate large number of households ---

    # Number of households
    N = 100_000
    # Number of periods to simulate
    T = 100

    # Create RNG instance
    rng = np.random.default_rng(seed=1234)

    # Simulate the wealth paths (result is an array of shape (T+1, N))
    a_sim = simulate_wealth_ar1_income(par, a0, T, N, rng)

    # Mean of simulated time series
    a_sim_mean = np.mean(a_sim, axis=1)

    # Cross-sectional variance of simulated time series
    a_sim_var = np.var(a_sim, axis=1)

    # Compute analytical mean
    a_mean_exact = compute_wealth_mean(par)

    # Plot cross-sectional mean and variance
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3), sharex=True)

    # Plot simulated vs. analytical mean
    ax1.axhline(a_mean_exact, color='black', ls='--', lw=1, label='Exact')
    ax1.plot(a_sim_mean, lw=1, label='Simulated')
    ax1.set_xlabel('Period')
    ax1.set_title('Cross-sectional mean of wealth')
    ax1.legend(loc='lower right')

    # Plot simulated variance
    ax2.plot(a_sim_var, lw=1, label='Simulated')
    ax2.set_title('Cross-sectional variance of wealth')
    ax2.set_xlabel('Period')
    ax2.legend(loc='lower right')
    plt.show()

    # Select cross section from last simulated period
    last_cross_section = a_sim[-1]

    G = gini(last_cross_section)
    print(f'Wealth Gini coefficient: {G:.3f}')
