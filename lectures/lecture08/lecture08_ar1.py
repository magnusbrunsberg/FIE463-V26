"""
Lecture 8: Code for section on AR(1) processes
"""

import numpy as np
import matplotlib.pyplot as plt


def simulate_ar1(x0, mu, rho, sigma, T, rng=None):
    """
    Simulate an AR(1) process for T periods, returning T+1 values including 
    the initial value.

    Parameters
    ----------
    x0 : float
        The initial value of the process.
    mu : float
        Intercept.
    rho : float
        The autoregressive parameter.
    sigma : float
        The standard deviation of the noise term.
    T : int
        The number of time periods to simulate.
    rng : Generator, optional
        Random number generator to use.

    Returns
    -------
    numpy.ndarray
        An array of length T+1 containing the simulated AR(1) process.
    """

    # Create an array to store the simulated values
    x = np.zeros(T + 1)

    # Set the initial value
    x[0] = x0

    # Create RNG instance if none was provided
    if rng is None:
        rng = np.random.default_rng(seed=1234)

    # Draw random shocks epsilons for each time period
    eps = rng.normal(loc=0, scale=sigma, size=T)

    # Simulate the AR(1) process
    for i in range(T):
        x[i + 1] = mu + rho * x[i] + eps[i]

    return x


if __name__ == '__main__':
    """
    Run code for AR(1) section
    """

    # --- Simulate single realization of AR(1) process ---

    # RNG instance with seed
    seed = 1234
    rng = np.random.default_rng(seed=seed)

    # Initial value
    x0 = 0.0

    # Intercept
    mu = 0.0

    # Autocorrelation parameter
    rho = 0.9

    # Standard deviation of the noise term
    sigma = 0.1

    # Number of periods to simulate
    T = 100

    # Simulate the AR(1) process
    simulated_data = simulate_ar1(x0, mu, rho, sigma, T, rng)

    plt.plot(simulated_data, label='Simulation')
    plt.xlabel('Time')
    plt.title('Simulated AR(1) Process')
    # Add unconditional mean
    uncond_mean = mu / (1 - rho)
    plt.axhline(uncond_mean, color='black', linestyle='--', lw=0.5, label='Mean')
    plt.legend()
    plt.show()

    # --- Simulate 20 realizations of AR(1) process ---

    # Simulate 20 different sequences
    N = 20

    # Create an array to store the simulated values
    data = np.zeros((N, T + 1))

    # Simulate the AR(1) process N times
    for i in range(N):
        data[i, :] = simulate_ar1(x0, mu, rho, sigma, T, rng)

    plt.figure(figsize=(7, 4))
    plt.plot(data.T, alpha=0.75, lw=0.75)
    plt.xlabel('Time')
    plt.title('Simulated AR(1) Process')
    # Add unconditional mean
    plt.axhline(uncond_mean, color='black', linestyle='--', lw=0.5, label='Mean')
    plt.show()
