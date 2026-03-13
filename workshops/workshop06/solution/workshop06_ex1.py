"""
Solution for workshop 6, exercise 1
"""

from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize, root_scalar


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


@dataclass
class Equilibrium:
    """
    Container to store equilibrium allocations and prices.
    """

    par: Parameters = None  # Parameters used to solve the equilibrium
    c: float = None  # Optimal consumption
    h: float = None  # Optimal labor supply
    w: float = None  # Equilibrium wage
    L: float = None  # Aggregate labor demand
    Y: float = None  # Aggregate output
    Pi: float = None  # Aggregate profits


def util(c, h, par: Parameters):
    """
    Compute the utility of a given consumption/labor supply choice.

    Parameters
    ----------
    c : float
        Consumption
    h : float
        Labor supply
    par : Parameters
        Parameter instance

    Returns
    -------
    u : float
        Utility
    """

    # Consumption utility
    if par.gamma == 1:
        # Log utility
        u = np.log(c)
    else:
        # General CRRA utility
        u = (c ** (1 - par.gamma) - 1) / (1 - par.gamma)

    # add disutility of labor
    u -= par.psi * h ** (1 + 1 / par.theta) / (1 + 1 / par.theta)

    return u


def solve_hh(w, pi, par: Parameters):
    """
    Solve household problem for given prices and parameters.

    Parameters
    ----------
    w : float
        Wage rate
    pi : float
        Firm profits distributed to households
    par : Parameters
        Parameter instance

    Returns
    -------
    c_opt : float
        Optimal consumption
    h_opt : float
        Optimal labor supply
    """

    # Initial guess for h
    h_guess = 0.5

    res = minimize(
        lambda h: -util(w * h + pi, h, par),
        x0=h_guess,
        method='L-BFGS-B',
        bounds=((0, None),),
    )

    if not res.success:
        # Print diagnostic error message if minimizer had problems
        print('Minimizer did not terminate successfully')
        print(res.message)
        print(f'  Arguments: w={w}, pi={pi}')

    # Store optimal hours choice
    h_opt = res.x[0]
    # Optimal consumption follows from budget constraint
    c_opt = w * h_opt + pi

    return c_opt, h_opt


def solve_firm(w, par: Parameters):
    """
    Compute labor demand and profits implied by firm's first-order condition
    for given prices w.

    Parameters
    ----------
    w : float
        Wage rate
    par : Parameters
        Parameter instance

    Returns
    -------
    L : float
        Labor demand
    Y : float
        Aggregate output
    Pi : float
        Aggregate profits
    """

    # Labor demand
    L = ((1 - par.alpha) * par.z / w) ** (1 / par.alpha)

    # Output
    Y = par.z * L ** (1 - par.alpha)

    # Profits
    Pi = Y - w * L

    return L, Y, Pi


def compute_labor_ex_demand(w, par: Parameters):
    """
    Compute excess demand for labor.

    Parameters
    ----------
    w : float
        Wage rate
    par : Parameters
        Parameter instance

    Returns
    -------
    ex_demand : float
        Excess demand for labor
    """

    # Wage and profits implied by firm's first-order condition
    L, Y, Pi = solve_firm(w, par)

    # Optimal household choices for given prices
    c_opt, h_opt = solve_hh(w, Pi, par)

    # Excess demand for labor
    ex_demand = L - h_opt

    return ex_demand


def compute_equilibrium(par):
    """
    Compute the equilibrium for given parameters.

    Parameters
    ----------
    par : Parameters
        Parameter instance

    Returns
    -------
    eq : Equilibrium
        Equilibrium instance containing equilibrium values
    """

    # Define initial bracket for root finder
    bracket = (1.0e-3, 5)

    res = root_scalar(compute_labor_ex_demand, bracket=bracket, args=(par,))

    if not res.converged:
        print('Equilibrium root finder did not terminate successfully')

    # Create instance of equilibrium class
    eq = Equilibrium(par=par, w=res.root)

    # Equilibrium wage, output and profits
    eq.L, eq.Y, eq.Pi = solve_firm(eq.w, par)

    # Equilibrium household choices
    eq.c, eq.h = solve_hh(eq.w, eq.Pi, par)

    return eq


def print_equilibrium(eq: Equilibrium):
    """
    Print equilibrium prices, allocations, and excess demand.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium instance containing equilibrium values
    """

    print('Equilibrium:')
    print('  Households:')
    print(f'    c = {eq.c:.5f}')
    print(f'    h = {eq.h:.5f}')
    print('  Firms:')
    print(f'    Y = {eq.Y:.5f}')
    print(f'    L = {eq.L:.5f}')
    print(f'    Pi = {eq.Pi:.5f}')
    print('  Prices:')
    print(f'    w = {eq.w:.5f}')
    print('  Market clearing:')
    print(f'    Labor market: {eq.L - eq.h:.5e}')
    print(f'    Goods market: {eq.c - eq.Y:.5e}')


def compute_analytical_solution(par: Parameters):
    """
    Compute analytical solution for given parameters.

    Parameters
    ----------
    par : Parameters
        Parameter instance

    Returns
    -------
    L : float
        Analytical solution for labor supply
    """

    # Base from the analytical formula for L from (1.2)
    x = (1 - par.alpha) * par.z ** (1 - par.gamma) / par.psi
    # Exponent in the analytical formula for L
    xp = 1 / (1 / par.theta + par.alpha + par.gamma * (1 - par.alpha))

    L = x**xp

    return L


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

    # Compare to analytical solution
    L = compute_analytical_solution(par)
    print(f'Analytical solution: h = L = {L:.5f}')
