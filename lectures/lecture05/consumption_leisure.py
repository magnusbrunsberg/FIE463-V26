"""
Lecture 5: Consumption-leisure problem
"""

import numpy as np
from scipy.optimize import minimize_scalar, root_scalar


def util(c, h, gamma, psi, theta):
    """
    Compute the utility of a given consumption/labor supply choice.

    Parameters
    ----------
    c : float or array
        Consumption level.
    h : float or array
        Hours worked.
    gamma : float
        Relative risk aversion parameter.
    psi : float
        Weight on disutility of labor.
    theta : float
        Labor supply elasticity.

    Returns
    -------
    u : float or array
        Utility value.
    """

    # Consumption utility
    if gamma == 1:
        # Log utility
        u = np.log(c)
    else:
        # General CRRA utility
        u = (c**(1-gamma) - 1) / (1-gamma)

    # add disutility of labor
    u -= psi * h**(1 + 1/theta) / (1 + 1/theta)

    return u


def util_h(h, gamma, psi, theta, a, w, diagnostics=False):
    """
    Compute utility for given labor choice and parameters.

    Parameters
    ----------
    h : float or array
        Hours worked.
    gamma : float
        Relative risk aversion parameter.
    psi : float
        Weight on disutility of labor.
    theta : float
        Labor supply elasticity.
    a : float
        Initial assets.
    w : float
        Wage rate.
    diagnostics : bool
        If True, print diagnostic information when the function is called.
        
    Returns
    -------
    u : float or array
        Utility value.
    """

    # Diagnostics: print the value of h when the function is called
    if diagnostics:
        print(f'Objective function called with h={h:.8f}')

    # Recover consumption from the budget constraint
    c = a + w * h

    # Use our previous implementation of u(c,h) to compute utility.
    u = util(c, h, gamma, psi, theta)

    return u


def foc(h, gamma, psi, theta, a, w):
    """
    Compute the difference LHS - RHS of the first-order condition.

    Parameters
    ----------
    h : float or array
        Hours worked.
    gamma : float
        Relative risk aversion parameter.
    psi : float
        Weight on disutility of labor.
    theta : float
        Labor supply elasticity.
    a : float
        Initial assets.
    w : float
        Wage rate.

    Returns
    -------
    diff : float or array
        Difference LHS - RHS of the first-order condition.  
    """

    # Compute the left-hand side (LHS) of the first-order condition
    lhs = (a + w * h)**(-gamma)
    # Compute the right-hand side (RHS) of the first-order condition
    rhs = psi * h**(1/theta) / w

    # Compute the difference of the LHS and RHS. Should be 0 at the optimum.
    diff = lhs - rhs
    return diff


def solve_grid_search(gamma, psi, theta, a, w, N=11):
    """
    Solves for the optimal consumption and leisure choices using grid search.

    Parameters
    ----------
    gamma : float
        Relative risk aversion parameter.
    psi : float
        Weight on disutility of labor.
    theta : float
        Labor supply elasticity.
    a : float
        Initial assets.
    w : float
        Wage rate.
    N : int, optional
        Number of grid points for hours worked.

    Returns
    -------
    c_opt : float
        Optimal consumption choice.
    h_opt : float
        Optimal hours worked choice.
    u_max : float
        Maximized utility level.
    """

    # Create candidate grid of hours worked
    h_grid = np.linspace(0.1, 2, N)

    # Recover consumption from budget constraint
    c_grid = a + w * h_grid

    # Evaluate utility for each consumption/hours choice
    u_grid = util(c_grid, h_grid, gamma, psi, theta)

    # Locate index where utility is maximized
    imax = np.argmax(u_grid)

    # Recover the maximizing hours and consumption choices
    h_opt = h_grid[imax]
    c_opt = c_grid[imax]

    # Recover the maximized utility level
    u_max = u_grid[imax]

    return c_opt, h_opt, u_max


def solve_analytical(gamma, psi, theta, a, w):
    """
    Solves for the optimal consumption and leisure choices using the analytical
    solution, assuming that assets are zero.

    Parameters
    ----------
    gamma : float
        Relative risk aversion parameter.
    psi : float
        Weight on disutility of labor.
    theta : float
        Labor supply elasticity.
    a : float
        Initial assets.
    w : float
        Wage rate.

    Returns
    -------
    c_opt : float
        Optimal consumption choice.
    h_opt : float
        Optimal hours worked choice.
    u_max : float
        Maximized utility level.

    """

    if a != 0:
        print('Analytical solution assumes a=0.')
        return np.nan, np.nan, np.nan

    # Analytical (exact) hours worked
    h_opt = (w**(1-gamma) / psi) ** (1 / (gamma + 1 / theta))

    # Analytical (exact) consumption
    c_opt = w * h_opt

    # Analytical (exact) utility
    u_max = util(c_opt, h_opt, gamma, psi, theta)

    return c_opt, h_opt, u_max


def solve_minimizer(gamma, psi, theta, a, w):
    """
    Solves for the optimal consumption and leisure choices using 
    a minimizer.

    Parameters
    ----------
    gamma : float
        Relative risk aversion parameter.
    psi : float
        Weight on disutility of labor.
    theta : float
        Labor supply elasticity.
    a : float
        Initial assets.
    w : float
        Wage rate.

    Returns
    -------
    c_opt : float
        Optimal consumption choice.
    h_opt : float
        Optimal hours worked choice.
    u_max : float
        Maximized utility level.
    """

    # Set boundaries for minimization.
    # We don't use 0 as the lower bound, since with a=0 that yields -inf utility.
    bounds = (1.0e-8, 10)

    # Perform maximization using bounded minimizer
    res = minimize_scalar(
        lambda x: -util_h(x, gamma, psi, theta, a, w), 
        bounds=bounds
    )

    # Maximizer is stored in the 'x' attribute
    h_opt = res.x
    # Optimal consumption follows from budget constraint
    c_opt = a + w * h_opt

    # Maximized utility is the NEGATIVE of the objective function
    u_max = -res.fun

    return c_opt, h_opt, u_max


def solve_root_finder(gamma, psi, theta, a, w):
    """
    Solves for the optimal consumption and leisure choices using 
    a root finder.

    Parameters
    ----------
    gamma : float
        Relative risk aversion parameter.
    psi : float
        Weight on disutility of labor.
    theta : float
        Labor supply elasticity.
    a : float
        Initial assets.
    w : float
        Wage rate.

    Returns
    -------
    c_opt : float
        Optimal consumption choice.
    h_opt : float
        Optimal hours worked choice.
    u_max : float
        Maximized utility level.
    """

    # Define the bracket (same as the boundaries) in which the root is located
    bracket = (1.0e-8, 10)

    # Auxiliary positional arguments to be passed to foc()
    args = (gamma, psi, theta, a, w)

    # Call root finder with an initial bracket and auxiliary arguments
    res = root_scalar(foc, bracket=bracket, args=args)

    # Function root is stored in the 'root' attribute
    h_opt = res.root
    # Optimal consumption follows from budget constraint
    c_opt = a + w * h_opt

    # The root finder does not evaluate the objective function, so we have to do 
    # it manually.
    u_max = util(c_opt, h_opt, gamma, psi, theta)

    return c_opt, h_opt, u_max


def main():
    """
    Main function to solve the consumption-leisure problem.
    """

    # Parameters for model with analytical solution
    a = 0           # initial assets
    w = 2           # wage rate
    gamma = 1       # Relative risk aversion
    psi = 1.5       # weight on disutility of labor
    theta = 0.5     # labor supply elasticity

    # === Grid search solution ===

    # Solve using grid search
    c_opt, h_opt, u_max = solve_grid_search(gamma, psi, theta, a, w, N=11)

    print('Grid search solution:')
    print(f'  c:     {c_opt:.5f}')
    print(f'  h:     {h_opt:.5f}')
    print(f'  u_max: {u_max:.5f}')

    # === Analytical solution ===

    c_exact, h_exact, u_max_exact = solve_analytical(gamma, psi, theta, a, w)

    print('\nExact solution:')
    print(f'  c:     {c_exact:.5f}')
    print(f'  h:     {h_exact:.5f}')
    print(f'  u_max: {u_max_exact:.5f}')

    # === Minimizer solution ===

    c_opt, h_opt, u_max = solve_minimizer(gamma, psi, theta, a, w)

    print('\nMinimizer solution:')
    print(f'  c:     {c_opt:.5f}')
    print(f'  h:     {h_opt:.5f}')
    print(f'  u_max: {u_max:.5f}')

    # === Root finder solution ===

    c_opt, h_opt, u_max = solve_root_finder(gamma, psi, theta, a, w)

    print('\nRoot finder solution:')
    print(f'  c:     {c_opt:.5f}')
    print(f'  h:     {h_opt:.5f}')
    print(f'  u_max: {u_max:.5f}')


if __name__ == "__main__":
    # Execute main function
    main()
