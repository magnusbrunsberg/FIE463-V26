"""
Solution for workshop 6, exercise 2
"""

from dataclasses import dataclass
import numpy as np
from scipy.optimize import root_scalar, root


from workshop06_ex1 import solve_firm, solve_hh


@dataclass
class Parameters:
    """
    Container to store the problem's parameters.
    """

    alpha: float = 0.36  # Capital share in production function
    z: float = 1.0  # TFP
    gamma: float = 2.0  # RRA in utility
    psi: float = 1.0  # Weight on disutility of working
    theta: float = 0.5  # Frisch elasticity of labor supply
    N1: int = 5  # Number of type-1 households
    N2: int = 5  # Number of type-2 households


@dataclass
class Equilibrium:
    """
    Container to store equilibrium allocations and prices.
    """

    par: Parameters = None  # Parameters used to solve the equilibrium
    c1: float = None  # Consumption of type 1 households
    h1: float = None  # Labor supply of type 1 households
    c2: float = None  # Consumption of type 2 households
    h2: float = None  # Labor supply of type 2 households
    pi2: float = None  # Per-capita profits of type 2 households
    w: float = None  # Equilibrium wage
    L: float = None  # Aggregate labor demand
    Y: float = None  # Aggregate output
    Pi: float = None  # Aggregate profits


def compute_labor_ex_demand(w, par: Parameters):
    """
    Compute the excess demand for labor.

    Parameters
    ----------
    w : float
        Wage rate.
    par : Parameters
        Model parameters.

    Returns
    -------
    float
        Excess demand for labor.
    """
    # Wage and profits implied by firm's first-order condition
    L, Y, Pi = solve_firm(w, par)

    # Equilibrium household choices
    # Type 1 receives no profits
    c1, h1 = solve_hh(w, 0.0, par)
    # Per-capita profits for type 2 households
    pi2 = Pi / par.N2
    c2, h2 = solve_hh(w, pi2, par)

    # Excess demand for labor
    ex_demand = L - par.N1 * h1 - par.N2 * h2

    return ex_demand


def compute_equilibrium(par):
    """
    Compute the equilibrium of the model.

    Parameters
    ----------
    par : Parameters
        Model parameters.

    Returns
    -------
    Equilibrium
        Equilibrium object containing the equilibrium values.
    """
    # Initial bracket for root finder
    bracket = (1.0e-3, 5)

    # Find equilibrium wage
    res = root_scalar(compute_labor_ex_demand, bracket=bracket, args=(par,))

    if not res.converged:
        print('Equilibrium root finder did not terminate successfully')

    # Create instance of equilibrium class
    eq = Equilibrium(par=par, w=res.root)

    # Equilibrium wage, output and profits
    eq.L, eq.Y, eq.Pi = solve_firm(eq.w, par)

    # Equilibrium household choices
    eq.c1, eq.h1 = solve_hh(eq.w, 0.0, par)
    # Per-capita profits for type 2 households
    eq.pi2 = eq.Pi / par.N2
    eq.c2, eq.h2 = solve_hh(eq.w, eq.pi2, par)

    return eq


def print_equilibrium(eq: Equilibrium):
    """
    Print equilibrium prices, allocations, and excess demand.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium object containing the equilibrium values.
    """

    N1, N2 = eq.par.N1, eq.par.N2

    print('Equilibrium:')
    print('  Households:')
    print(f'    Type 1 (N = {N1}):')
    print(f'      c1 = {eq.c1:.5f}')
    print(f'      h1 = {eq.h1:.5f}')
    print(f'    Type 2 (N = {N2}):')
    print(f'      c2 = {eq.c2:.5f}')
    print(f'      h2 = {eq.h2:.5f}')
    print(f'      pi2 = {eq.pi2:.5f}')
    print('  Firms:')
    print(f'    Y = {eq.Y:.5f}')
    print(f'    L = {eq.L:.5f}')
    print(f'    Pi = {eq.Pi:.5f}')
    print('  Prices:')
    print(f'    w = {eq.w:.5f}')
    print('  Market clearing:')
    print(f'    Labor market: {eq.L - N1 * eq.h1 - N2 * eq.h2:.5e}')
    print(f'    Goods market: {N1 * eq.c1 + N2 * eq.c2 - eq.Y:.5e}')
    print(f'    Profits: {N2 * eq.pi2 - eq.Pi:.5e}')


def foc_error(x, par: Parameters):
    """
    Compute errors in first-order conditions of the household problem
    for type 1 and type 2.

    (for advanced solution method ONLY)

    Parameters
    ----------
    x : array_like
        Candidate guess for labor supply (h1, h2).
    par : Parameters
        Model parameters.

    Returns
    -------
    numpy.ndarray
        Array containing the differences from the first-order conditions.
    """
    # Extract candidate guess for labor supply
    h1, h2 = x

    # Aggregate labor supply
    L = par.N1 * h1 + par.N2 * h2
    # wage from firm's FOC
    w = (1 - par.alpha) * par.z * L ** (-par.alpha)
    # Aggregate firm profits
    Pi = par.alpha * par.z * L ** (1 - par.alpha)

    # FOC for HH type 1
    diff1 = par.psi * h1 ** (1 / par.theta) / w - (w * h1) ** (-par.gamma)
    # Profits per capita for HH type 2
    pi2 = Pi / par.N2
    # FOC for HH type 2
    diff2 = par.psi * h2 ** (1 / par.theta) / w - (w * h2 + pi2) ** (-par.gamma)

    fx = np.array((diff1, diff2))

    return fx


def compute_equilibrium_root(par):
    """
    Compute the equilibrium of the model by running a root finder on
    the household's first-order conditions.

    (for advanced solution method ONLY)

    Parameters
    ----------
    par : Parameters
        Model parameters.

    Returns
    -------
    Equilibrium
        Equilibrium object containing the equilibrium values.
    """

    # Initial guess for labor supply (h1, h2)
    x0 = np.array((0.5, 0.5))

    # Find (h1, h2) that satisfy the FOCs
    res = root(foc_error, x0=x0, args=(par,), method='hybr')

    if not res.success:
        print('Equilibrium root finder did not terminate successfully')

    # Extract equilibrium labor supply choices
    h1, h2 = res.x

    # Aggregate labor supply
    L = par.N1 * h1 + par.N2 * h2

    # Wage implied by firm's FOC
    w = (1 - par.alpha) * par.z * L ** (-par.alpha)

    # Create instance of equilibrium class
    eq = Equilibrium(par=par, w=w, L=L, h1=h1, h2=h2)

    # Equilibrium wage, output and profits
    eq.L, eq.Y, eq.Pi = solve_firm(eq.w, par)

    # Equilibrium household choices
    eq.c1 = w * eq.h1
    eq.pi2 = eq.Pi / par.N2
    eq.c2 = w * eq.h2 + eq.pi2

    return eq


if __name__ == '__main__':
    """
    Main script to compute and print the equilibrium of the model.
    """

    # Get instance of default parameter values
    par = Parameters()

    # Solve for equilibrium
    eq = compute_equilibrium(par)

    # Print equilibrium quantities and prices
    print_equilibrium(eq)

    # Use root finder based on households' first-order conditions 
    # (advanced solution method)
    eq_ = compute_equilibrium_root(par)
    print('\nEquilibrium computed using root finder:')
    print_equilibrium(eq_)
