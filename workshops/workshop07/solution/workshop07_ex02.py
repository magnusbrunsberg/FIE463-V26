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

    # Savings by the young today (capital of the old tomorrow)
    a = s * w

    # Capital-labor k ratio tomorrow:
    # From asset market clearing: K_{t+1} = N * a_t
    K_next = par.N * a
    # From labor market clearing: L = N
    L = par.N
    # => k_{t+1} = K_{t+1} / L = a_t
    k_next = K_next / L

    # Compute factor prices tomorrow given candidate capital and future TFP
    r_next, _ = compute_prices(k_next, z_next, par)

    # Consumption today and tomorrow
    c_y = (1 - s) * w
    c_o = (1 + r_next) * a

    # Euler equation: u'(c_y) = beta * (1 + r_next) * u'(c_o)
    # Re-arranged to f(s) = lhs - rhs
    lhs = c_y ** (-par.gamma)
    rhs = par.beta * (1 + r_next) * c_o ** (-par.gamma)

    return lhs - rhs


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

    # Retrieve parameter object
    par = eq.par

    # Number of periods to simulate (need z[t+1] to solve for s[t])
    T = len(z_series) - 2

    # Initialize simulation and allocate arrays
    sim = initialize_sim(T, eq)

    # Set the TFP path (entire series)
    sim.z = z_series

    # Iterate through time periods
    for t in range(1, T + 1):
        # Capital stock is predetermined by savings in previous period
        sim.K[t] = sim.a[t - 1] * par.N

        # Compute current factor prices
        sim.r[t], sim.w[t] = compute_prices(sim.K[t] / par.N, sim.z[t], par)

        # Solve for savings rate using root finding on Euler equation
        # We need current wage and next period's TFP to evaluate the FOC
        args = (sim.w[t], sim.z[t + 1], par)
        res = root_scalar(euler_err, bracket=(1e-5, 1 - 1e-5), args=args)

        if not res.converged:
            print(f'Root-finder did not converge at t={t}')

        # Store the found optimal savings rate
        sim.s[t] = res.root

        # Compute remaining choices and aggregates
        sim.a[t] = sim.s[t] * sim.w[t]
        sim.c_y[t] = (1 - sim.s[t]) * sim.w[t]
        sim.c_o[t] = (1 + sim.r[t]) * sim.a[t - 1]
        sim.Y[t] = sim.z[t] * sim.K[t] ** par.alpha * par.N ** (1 - par.alpha)

        # Verify goods market clearing
        demand = par.N * (sim.c_y[t] + sim.c_o[t] + sim.a[t])
        supply = sim.Y[t] + (1 - par.delta) * sim.K[t]
        assert abs(demand - supply) < 1.0e-8

    return sim


if __name__ == '__main__':
    # --- Case 1: Log preferences (gamma=1) ---

    # Create parameter instance
    par_rra1 = Parameters(gamma=1.0)

    # Solve for the initial equilibrium
    eq_rra1 = compute_steady_state(par_rra1)

    # Print equilibrium quantities and prices
    print_steady_state(eq_rra1)

    # Number of periods to simulate (excluding initial steady state)
    T = 20

    # Initialize empty TFP series
    # Note: we need T+2 periods to have z[t+1] for t=1,...,T in the simulation
    z_pers = np.empty(T + 2)
    # Set initial TFP to steady-state value
    z_pers[0] = par_rra1.z
    # Drop TFP by 10% in period 1
    z_pers[1] = 0.9 * z_pers[0]
    # Subsequently, TFP evolves according to the process: z_t = (1-kappa)*z_{t-1} + kappa*1.0
    for t in range(2, T + 2):
        z_pers[t] = (1 - par_rra1.kappa) * z_pers[t - 1] + par_rra1.kappa * 1.0

    # Perform simulation
    sim_rra1 = simulate_olg_crra(z_pers, eq_rra1)

    # Define file name for figure (placed in the same folder as this script)
    filename = Path(__file__).parent / 'workshop07_ex02_gamma1.pdf'

    # Plot simulation results
    plot_simulation(eq_rra1, sim_rra1, filename=filename)

    # --- Case 2: Generate CRRA with gamma = 5 ---

    # Create parameter instance
    par_rra5 = Parameters(gamma=5.0)

    # Solve for the initial equilibrium
    eq_rra5 = compute_steady_state(par_rra5)

    # Print equilibrium quantities and prices
    print_steady_state(eq_rra5)

    # Perform simulation
    sim_rra5 = simulate_olg_crra(z_pers, eq_rra5)

    # Define file name for figure (placed in the same folder as this script)
    filename = Path(__file__).parent / 'workshop07_ex02_gamma5.pdf'

    # Plot simulation results
    plot_simulation(
        eq_rra1,
        sim=sim_rra1,
        eq_other=eq_rra5,
        sim_other=sim_rra5,
        labels=(r'Transition: $\gamma=1$', r'Transition: $\gamma=5$'),
        filename=filename,
    )
