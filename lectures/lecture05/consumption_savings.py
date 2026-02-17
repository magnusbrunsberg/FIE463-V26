"""
Lecture 5: Consumption-savings problem
"""

import numpy as np
from scipy.optimize import minimize_scalar


def util(c, gamma):
    """
    Compute the per-period utility for given consumption.

    Parameters
    ----------
    c : float or array
        Consumption level.
    gamma : float
        Relative risk aversion parameter.

    Returns
    -------
    u : float or array
        Utility value.
    """
    if gamma == 1:
        # Log preferences
        u = np.log(c)
    else:
        # General CRRA preferences
        u = c**(1-gamma) / (1-gamma)
    return u


def util_life_c1(c1, beta, gamma, y, r):
    """
    Compute the lifetime utility of consuming c1, with c2 following from budget 
    constraint.

    Parameters
    ----------
    c1 : float or array
        Period-1 consumption.
    beta : float
        Discount factor.
    gamma : float
        Relative risk aversion parameter.
    y : float
        Present value of lifetime income.
    r : float
        Interest rate.

    Returns
    -------
    U : float or array
        Lifetime utility.
    """

    # Recover c2 from budget constraint
    c2 = (1+r) * (y - c1)

    # Evaluate period-1 and period-2 utility
    u1 = util(c1, gamma)
    u2 = util(c2, gamma)

    # Compute lifetime utility 
    U = u1 + beta * u2

    return U


def solve_cons_sav(beta, gamma, y1, y2, r):
    """
    Solve the consumption-savings problem for given parameters.

    Parameters
    ----------
    beta : float
        Discount factor.
    gamma : float
        Relative risk aversion parameter.
    y1 : float
        Period-1 income.
    y2 : float
        Period-2 income.
    r : float
        Interest rate.

    Returns
    -------
    c1_opt: float 
        Optimal period-1 consumption
    c2_opt: float
        Optimal period-2 consumption
    u_max : float
        Maximized lifetime utility
    res : OptimizeResult
        Optimization result object
    """

    # Lifetime income
    Y = y1 + y2 / (1+r)

    # Bounds for period-1 consumption (require min. consumption of 1e-8)
    bounds = (1.0e-8, Y - 1.0e-8)

    # Run minimizer, store result object
    res = minimize_scalar(
        lambda x: -util_life_c1(x, beta, gamma, Y, r), 
        bounds=bounds
    )

    # Store optimal period-1 consumption
    c1_opt = res.x

    # Recover implied optimal period-2 consumption from the budget constraint
    c2_opt = (1+r) * (Y - c1_opt)

    # Recover maximized lifetime utility
    u_max = - res.fun

    # Return optimal consumption and the result object
    return c1_opt, c2_opt, u_max, res


def solve_analytical(beta, gamma, y1, y2, r):
    """
    Solve the consumption-savings problem using the analytical solution.

    Parameters
    ----------
    beta : float
        Discount factor.
    gamma : float
        Relative risk aversion parameter.
    y1 : float
        Period-1 income.
    y2 : float
        Period-2 income.
    r : float
        Interest rate.

    Returns
    -------
    c1_opt: float 
        Optimal period-1 consumption
    c2_opt: float
        Optimal period-2 consumption
    u_max : float
        Maximized utility level
    """

    # Marginal propensity to consume out of lifetime income (from analytical solution)
    mpc = 1/(1 + beta**(1/gamma) * (1+r)**(1/gamma - 1))

    # Lifetime income
    Y = y1 + y2 / (1+r)

    # Analytical consumption choices
    c1_opt = mpc * Y
    c2_opt = (beta * (1+r))**(1/gamma) * c1_opt

    u_max = util(c1_opt, gamma) + beta * util(c2_opt, gamma)

    return c1_opt, c2_opt, u_max


def main():
    """
    Main function to solve the consumption-savings problem.
    """

    # Parameters
    beta = 0.96         # discount factor
    gamma = 1.0         # relative risk aversion (RRA)
    y1 = 1.0            # period-1 income
    y2 = 1.0            # period-2 income
    r = 0.04            # interest rate (4%)

    # === Solve using minimizer ===

    c1_opt, c2_opt, u_max, res = solve_cons_sav(beta, gamma, y1, y2, r)

    print('Minimizer solution:')
    print(f'  c1:    {c1_opt:.5f}')
    print(f'  c2:    {c2_opt:.5f}')
    print(f'  u_max: {u_max:.5f}')

    # === Solve using analytical solution ===

    c1_opt, c2_opt, u_max = solve_analytical(beta, gamma, y1, y2, r)

    print('\nAnalytical solution:')
    print(f'  c1:    {c1_opt:.5f}')
    print(f'  c2:    {c2_opt:.5f}')
    print(f'  u_max: {u_max:.5f}')


if __name__ == "__main__":
    # Execute main function
    main()
