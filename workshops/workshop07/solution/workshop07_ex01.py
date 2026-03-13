"""
Workshop 7: Overlapping generations (OLG)
Exercise 1: Transitory vs persistent TFP changes

This module implements the solution for the steady state and transition
dynamics of the OLG model assuming log utility.
"""

from pathlib import Path
import numpy as np
from dataclasses import dataclass
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


@dataclass
class Parameters:
    """
    Parameters for the overlapping generations model.
    """

    alpha: float = 0.36  # Capital share in production function
    delta: float = 1.0  # Depreciation rate
    z: float = 1.0  # TFP
    beta: float = 0.96**30  # Discount factor (0.96 per year, 30-year periods)
    gamma: float = 1.0  # RRA in utility
    N: int = 1  # Number of households per cohort
    kappa: float = 0.1  # Parameter for TFP shock persistence (used in scenario B)


@dataclass
class SteadyState:
    """
    Steady-state equilibrium of the OLG model.
    """

    par: Parameters = None  # Parameters used to compute equilibrium
    c_y: float = None  # Consumption when young
    c_o: float = None  # Consumption when old
    a: float = None  # Savings when young
    s: float = None  # Savings rate when young
    r: float = None  # Interest rate (return on capital)
    w: float = None  # Wage rate
    K: float = None  # Aggregate capital stock
    L: float = None  # Aggregate labor demand
    I: float = None  # Aggregate investment
    Y: float = None  # Aggregate output


@dataclass
class Simulation:
    """
    Container to store simulation results
    """

    c_y: np.ndarray = None  # Time series for consumption when young
    c_o: np.ndarray = None  # Time series for consumption when old
    a: np.ndarray = None  # Time series for savings when young
    s: np.ndarray = None  # Time series for savings rate when young
    r: np.ndarray = None  # Time series for interest rate (return on capital)
    w: np.ndarray = None  # Time series for wages
    K: np.ndarray = None  # Time series for aggregate capital stock
    Y: np.ndarray = None  # Time series for aggregate output
    z: np.ndarray = None  # Time series for TFP


def compute_prices(k, z, par: Parameters):
    """
    Return factor prices for a given capital-labor ratio, TFP, and parameters.

    Parameters
    ----------
    k : float
        Capital-labor ratio
    z : float
        Total factor productivity (TFP)
    par : Parameters
        Parameters for the given problem

    Returns
    -------
    r : float
        Return on capital after depreciation (interest rate)
    w : float
        Wage rate

    """

    # Return on capital after depreciation (interest rate)
    r = par.alpha * z * k ** (par.alpha - 1) - par.delta

    # Wage rate
    w = (1 - par.alpha) * z * k**par.alpha

    return r, w


def compute_savings_rate(r, par: Parameters):
    """
    Compute the savings rate using the analytical solution
    to the household problem.

    Parameters
    ----------
    r : float
        Return on capital after depreciation (interest rate)
    par : Parameters
        Parameters for the given problem

    Returns
    -------
    s : float
        Savings rate
    """

    s = 1 / (1 + par.beta ** (-1 / par.gamma) * (1 + r) ** (1 - 1 / par.gamma))

    return s


def compute_capital_ex_demand(k, par: Parameters):
    """
    Compute the excess demand for capital.

    Parameters
    ----------
    k : float
        Capital-labor ratio
    par : Parameters
        Parameters for the given problem

    Returns
    -------
    ex_demand : float
        Excess demand for capital
    """

    # Compute prices from firm's FOCs
    r, w = compute_prices(k, par.z, par)

    # Compute savings rate
    srate = compute_savings_rate(r, par)

    # Aggregate supply of capital by households (savings)
    A = srate * w * par.N

    # Aggregate labor supply
    L = par.N

    # Aggregate capital demand
    K = k * L

    # Excess demand for capital
    ex_demand = K - A

    return ex_demand


def compute_steady_state(par: Parameters):
    """
    Compute the steady-state equilibrium for the OLG model.

    Parameters
    ----------
    par : Parameters
        Parameters for the given problem

    Returns
    -------
    eq : SteadyState
        Steady-state equilibrium of the OLG model
    """

    # Find the equilibrium k=K/L with a root finder. Excess demand for capital
    # has to be zero in equilibrium.
    res = root_scalar(compute_capital_ex_demand, bracket=(1.0e-3, 10), args=(par,))

    if not res.converged:
        print('Equilibrium root finder did not terminate successfully')

    # Equilibrium K
    K = res.root * par.N

    # Create instance of equilibrium class
    eq = SteadyState(par=par, K=K, L=par.N)

    # Equilibrium prices
    eq.r, eq.w = compute_prices(eq.K / eq.L, par.z, par)

    # Investment in steady state
    eq.I = eq.K * par.delta

    # Equilibrium household choices
    eq.s = compute_savings_rate(eq.r, par)
    eq.a = eq.s * eq.w
    eq.c_y = eq.w - eq.a
    eq.c_o = (1 + eq.r) * eq.a

    # Equilibrium output
    eq.Y = par.z * eq.K**par.alpha * eq.L ** (1 - par.alpha)

    # Aggregate consumption
    C = par.N * (eq.c_y + eq.c_o)
    # Check that goods market clearing holds using Y = C + I
    assert abs(eq.Y - C - eq.I) < 1.0e-8

    return eq


def print_steady_state(eq: SteadyState):
    """
    Print equilibrium prices, allocations, and excess demand.

    Parameters
    ----------
    eq : SteadyState
        SteadyState of the OLG model
    """

    # Number of households
    N = eq.par.N

    print('Steady-state equilibrium:')
    print('  Households:')
    print(f'    c_y = {eq.c_y:.5f}')
    print(f'    c_o = {eq.c_o:.5f}')
    print(f'    a = {eq.a:.5f}')
    print('  Firms:')
    print(f'    K = {eq.K:.5f}')
    print(f'    L = {eq.L:.5f}')
    print(f'    Y = {eq.Y:.5f}')
    print('  Prices:')
    print(f'    r = {eq.r:.5f}')
    print(f'    w = {eq.w:.5f}')
    print('  Market clearing:')
    print(f'    Capital market: {eq.K - eq.a * N:.5e}')
    print(
        f'    Goods market: {(eq.c_y + eq.c_o + eq.a) * N - eq.Y - (1 - eq.par.delta) * eq.K:.5e}'
    )


def initialize_sim(T, eq: SteadyState = None):
    """
    Initialize simulation instance (allocate arrays for time series).

    Parameters
    ----------
    T : int
        Number of periods to simulate
    eq : SteadyState, optional
        Steady-state equilibrium to use for initial period
    """

    # Initialize simulation instance
    sim = Simulation()

    # Initialize time series
    sim.c_y = np.empty(T + 1)
    sim.c_o = np.empty(T + 1)
    sim.a = np.empty(T + 1)
    sim.s = np.empty(T + 1)
    sim.r = np.empty(T + 1)
    sim.w = np.empty(T + 1)
    sim.K = np.empty(T + 1)
    sim.Y = np.empty(T + 1)
    sim.z = np.empty(T + 1)

    if eq is not None:
        # Set initial values to steady-state values
        sim.c_y[0] = eq.c_y
        sim.c_o[0] = eq.c_o
        sim.a[0] = eq.a
        sim.s[0] = eq.s
        sim.r[0] = eq.r
        sim.w[0] = eq.w
        sim.K[0] = eq.K
        sim.Y[0] = eq.Y
        sim.z[0] = eq.par.z

    return sim


def simulate_olg(z_series, eq: SteadyState):
    """
    Simulate the transition dynamics of the OLG model for a given TFP series.
    This implementation assumes log utility (gamma=1).

    Parameters
    ----------
    z_series : np.ndarray
        Time series of TFP values
    eq : SteadyState
        Initial steady-state equilibrium

    Returns
    -------
    sim : Simulation
        Simulation of the OLG model
    """

    # Retrieve parameter object
    par = eq.par

    # Check for log utility
    if par.gamma != 1:
        raise ValueError('simulate_olg only implemented for log utility')

    # Number of periods to simulate
    T = len(z_series) - 1

    # Initialize simulation and allocate arrays
    sim = initialize_sim(T, eq)

    # Set the TFP path
    sim.z[:] = z_series

    # Savings rate is constant over time for log utility
    s = par.beta / (1 + par.beta)
    sim.s[:] = s

    # Iterate through time periods
    for t in range(1, T + 1):
        # Capital stock is predetermined by savings in previous period
        sim.K[t] = sim.a[t - 1] * par.N

        # Compute factor prices given current K and current z
        sim.r[t], sim.w[t] = compute_prices(sim.K[t] / par.N, sim.z[t], par)

        # Savings by the young
        sim.a[t] = s * sim.w[t]

        # Consumption by the young and old
        sim.c_y[t] = (1 - s) * sim.w[t]
        sim.c_o[t] = (1 + sim.r[t]) * sim.a[t - 1]

        # Aggregate output
        sim.Y[t] = sim.z[t] * sim.K[t] ** par.alpha * par.N ** (1 - par.alpha)

        # Verify goods market clearing: Y + (1-delta)K = C + a*N
        demand = par.N * (sim.c_y[t] + sim.c_o[t] + sim.a[t])
        supply = sim.Y[t] + (1 - par.delta) * sim.K[t]
        assert abs(demand - supply) < 1.0e-8

    return sim


def plot_simulation(
    eq,
    sim,
    eq_new=None,
    deviations=True,
    eq_other=None,
    sim_other=None,
    labels=None,
    filename=None,
):
    """
    Plot the selected simulated time series of the OLG model.

    Parameters
    ----------
    eq : SteadyState
        The equilibrium containing the initial steady-state parameters.
    sim : Simulation
        The simulation containing the time series data.
    eq_new : SteadyState, optional
        The equilibrium containing the new steady-state parameters.
    deviations : bool
        If True, plot deviations from the initial steady state instead
        of absolute values.
    eq_other : SteadyState, optional
        Another steady-state equilibrium to plot for comparison.
    sim_other : Simulation, optional
        Another simulation to plot for comparison.
    labels : list of str, optional
        Labels for the different simulations.
    filename : str, optional
        If provided, save the figure to this location.
    """

    fig, axes = plt.subplots(
        nrows=4,
        ncols=2,
        figsize=(7, 7),
        sharex=True,
        sharey=False,
        constrained_layout=True,
    )

    if eq_other is None and sim_other is not None:
        raise ValueError('sim_other provided without eq_other')
    if eq_other is not None and sim_other is None:
        raise ValueError('eq_other provided without sim_other')

    # Keyword arguments for time series plots
    kwargs = {
        'color': 'steelblue',
        'linestyle': '-',
        'linewidth': 1.25,
        'marker': 'o' if len(sim.K) < 30 else None,
        'markersize': 2.5,
    }
    # Keyword arguments for time series plots of the other simulation (if provided)
    kwargs_other = {
        'color': 'darkred',
        'linestyle': '-',
        'linewidth': 1.0,
        'marker': 'o' if len(sim.K) < 30 else None,
        'markersize': 3.0,
        'mfc': 'none',
        'mew': 0.75,
    }

    # Keyword arguments for horizontal lines indicating (initial) steady state
    kwargs_init = {
        'color': 'black',
        'linewidth': 0.5,
        'linestyle': '--',
        'label': 'Steady state' if eq_new is None else 'Initial steady state',
    }

    # Keyword arguments for horizontal lines indicating new steady state
    kwargs_new = {
        'color': 'red',
        'linewidth': 0.5,
        'linestyle': '--',
        'label': 'New steady state',
    }

    if eq_new is not None:
        ylabel = 'Deviation from initial SS' if deviations else None
    else:
        ylabel = 'Deviation from SS' if deviations else None

    # Panel showing TFP time series
    ax = axes[0, 0]
    # Horizontal line at old steady state
    yvalues = 0 if deviations else eq.par.z
    ax.axhline(yvalues, **kwargs_init)
    # Plot TFP time series
    yvalues = sim.z / eq.par.z - 1 if deviations else sim.z
    ymin, ymax = yvalues.min(), yvalues.max()
    label = labels[0] if labels else 'Transition'
    ax.plot(yvalues, label=label, **kwargs)
    # Horizontal line at new steady state
    if eq_new is not None:
        yvalues = eq_new.par.z / eq.par.z - 1 if deviations else eq_new.par.z
        ax.axhline(yvalues, **kwargs_new)
    ax.set_ylabel(ylabel)
    ax.set_title('TFP $z$')
    if sim_other is not None:
        yvalues = sim_other.z / eq_other.par.z - 1 if deviations else sim_other.z
        label = labels[1] if labels else 'Other simulation'
        ax.plot(yvalues, label=label, **kwargs_other)
        ymin, ymax = min(ymin, yvalues.min()), max(ymax, yvalues.max())
    # Expand y-limits for nicer plots
    ax.set_ylim(ymin - 0.1 * (ymax - ymin), ymax + 0.1 * (ymax - ymin))
    ax.legend(loc='lower right')

    # Plot output time series
    ax = axes[0, 1]
    yvalues = sim.Y / eq.Y - 1 if deviations else sim.Y
    ymin, ymax = yvalues.min(), yvalues.max()
    ax.plot(yvalues, **kwargs)
    # Horizontal line at old steady state
    yvalues = 0 if deviations else eq.Y
    ax.axhline(yvalues, **kwargs_init)
    # Horizontal line at new steady state
    if eq_new is not None:
        yvalues = eq_new.Y / eq.Y - 1 if deviations else eq_new.Y
        ax.axhline(yvalues, **kwargs_new)
    ax.set_title('Output $Y$')
    ax.set_ylabel(ylabel)
    if sim_other is not None:
        yvalues = sim_other.Y / eq_other.Y - 1 if deviations else sim_other.Y
        ax.plot(yvalues, **kwargs_other)
        ymin, ymax = min(ymin, yvalues.min()), max(ymax, yvalues.max())
    # Expand y-limits for nicer plots
    ax.set_ylim(ymin - 0.1 * (ymax - ymin), ymax + 0.1 * (ymax - ymin))

    # Plot capital time series
    ax = axes[1, 0]
    yvalues = sim.K / eq.K - 1 if deviations else sim.K
    ymin, ymax = yvalues.min(), yvalues.max()
    ax.plot(yvalues, **kwargs)
    # Horizontal line at old steady state
    yvalues = 0 if deviations else eq.K
    ax.axhline(yvalues, **kwargs_init)
    # Horizontal line at new steady state
    if eq_new is not None:
        yvalues = eq_new.K / eq.K - 1 if deviations else eq_new.K
        ax.axhline(yvalues, **kwargs_new)
    ax.set_title('Capital $K$')
    ax.set_ylabel(ylabel)
    if sim_other is not None:
        yvalues = sim_other.K / eq_other.K - 1 if deviations else sim_other.K
        ax.plot(yvalues, **kwargs_other)
        ymin, ymax = min(ymin, yvalues.min()), max(ymax, yvalues.max())
    # Expand y-limits for nicer plots
    ax.set_ylim(ymin - 0.1 * (ymax - ymin), ymax + 0.1 * (ymax - ymin))

    # Plot savings rate time series
    ax = axes[1, 1]
    # Plot savings rate in levels, not deviations
    yvalues = sim.s
    ymin, ymax = yvalues.min(), yvalues.max()
    ax.plot(sim.s, **kwargs)
    # Horizontal line at old steady state
    ax.axhline(eq.s, **kwargs_init)
    # Horizontal line at new steady state
    if eq_new is not None:
        ax.axhline(eq_new.s, **kwargs_new)
    ax.set_title('Savings rate $s$')
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    if sim_other is not None:
        yvalues = sim_other.s
        ax.plot(yvalues, **kwargs_other)
        ymin, ymax = min(ymin, yvalues.min()), max(ymax, yvalues.max())
    # Handle cases where savings rate is nearly constant as this creates
    # misleading plot ranges
    if abs(ymin - ymax) < 0.05:
        ymin, ymax = ymin - 0.025, ymax + 0.025
    # Expand y-limits for nicer plots
    ax.set_ylim(ymin - 0.1 * (ymax - ymin), ymax + 0.1 * (ymax - ymin))

    # Plot consumption of the young time series
    ax = axes[2, 0]
    yvalues = sim.c_y / eq.c_y - 1 if deviations else sim.c_y
    ymin, ymax = yvalues.min(), yvalues.max()
    ax.plot(yvalues, **kwargs)
    # Horizontal line at old steady state
    yvalues = 0 if deviations else eq.c_y
    ax.axhline(yvalues, **kwargs_init)
    # Horizontal line at new steady state
    if eq_new is not None:
        yvalues = eq_new.c_y / eq.c_y - 1 if deviations else eq_new.c_y
        ax.axhline(yvalues, **kwargs_new)
    ax.set_title('Consumption when young $c_y$')
    ax.set_ylabel(ylabel)
    if sim_other is not None:
        yvalues = sim_other.c_y / eq_other.c_y - 1 if deviations else sim_other.c_y
        ax.plot(yvalues, **kwargs_other)
        ymin, ymax = min(ymin, yvalues.min()), max(ymax, yvalues.max())
    # Expand y-limits for nicer plots
    ax.set_ylim(ymin - 0.1 * (ymax - ymin), ymax + 0.1 * (ymax - ymin))

    # Plot consumption of the old time series
    ax = axes[2, 1]
    yvalues = sim.c_o / eq.c_o - 1 if deviations else sim.c_o
    ymin, ymax = yvalues.min(), yvalues.max()
    ax.plot(yvalues, **kwargs)
    # Horizontal line at old steady state
    yvalues = 0 if deviations else eq.c_o
    ax.axhline(yvalues, **kwargs_init)
    # Horizontal line at new steady state
    if eq_new is not None:
        yvalues = eq_new.c_o / eq.c_o - 1 if deviations else eq_new.c_o
        ax.axhline(yvalues, **kwargs_new)
    ax.set_title('Consumption when old $c_o$')
    ax.set_ylabel(ylabel)
    if sim_other is not None:
        yvalues = sim_other.c_o / eq_other.c_o - 1 if deviations else sim_other.c_o
        ax.plot(yvalues, **kwargs_other)
        ymin, ymax = min(ymin, yvalues.min()), max(ymax, yvalues.max())
    # Expand y-limits for nicer plots
    ax.set_ylim(ymin - 0.1 * (ymax - ymin), ymax + 0.1 * (ymax - ymin))

    # Plot interest rate time series (in levels, not as deviations)
    ax = axes[3, 0]
    yvalues = sim.r
    ymin, ymax = yvalues.min(), yvalues.max()
    ax.plot(yvalues, **kwargs)
    # Horizontal line at old steady state
    ax.axhline(eq.r, **kwargs_init)
    # Horizontal line at new steady state
    if eq_new is not None:
        ax.axhline(eq_new.r, **kwargs_new)
    ax.set_xlabel('Period')
    ax.set_title('Interest rate $r$')
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    if sim_other is not None:
        yvalues = sim_other.r
        ax.plot(yvalues, **kwargs_other)
        ymin, ymax = min(ymin, yvalues.min()), max(ymax, yvalues.max())
    # Expand y-limits for nicer plots
    ax.set_ylim(ymin - 0.1 * (ymax - ymin), ymax + 0.1 * (ymax - ymin))

    # Plot wage time series
    ax = axes[3, 1]
    yvalues = sim.w / eq.w - 1 if deviations else sim.w
    ymin, ymax = yvalues.min(), yvalues.max()
    ax.plot(yvalues, **kwargs)
    # Horizontal line at old steady state
    yvalues = 0 if deviations else eq.w
    ax.axhline(yvalues, **kwargs_init)
    # Horizontal line at new steady state
    if eq_new is not None:
        yvalues = eq_new.w / eq.w - 1 if deviations else eq_new.w
        ax.axhline(yvalues, **kwargs_new)
    ax.set_title('Wage $w$')
    ax.set_xlabel('Period')
    ax.set_ylabel(ylabel)
    if sim_other is not None:
        yvalues = sim_other.w / eq_other.w - 1 if deviations else sim_other.w
        ax.plot(yvalues, **kwargs_other)
        ymin, ymax = min(ymin, yvalues.min()), max(ymax, yvalues.max())
    # Expand y-limits for nicer plots
    ax.set_ylim(ymin - 0.1 * (ymax - ymin), ymax + 0.1 * (ymax - ymin))

    # Apply settings common to all axes
    if deviations:
        for ax in axes.flat:
            # Set percent formatting for y-tick labels
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))

    # Optionally save the figure
    if filename:
        plt.savefig(filename)


if __name__ == '__main__':
    # Create parameter instance
    par = Parameters()

    # Solve for the initial equilibrium
    eq = compute_steady_state(par)

    # Print equilibrium quantities and prices
    print_steady_state(eq)

    # --- Transition dynamics ---

    # Number of periods to simulate
    T = 20

    # -- Scenario A: Transitory TFP shock --

    # Initialize TFP series with steady-state value
    z_trans = np.full(T + 1, fill_value=par.z)
    # Drop TFP by 10% in period 1, leave other periods unchanged
    z_trans[1] = 0.9 * z_trans[0]

    # Perform simulation
    sim_trans = simulate_olg(z_trans, eq)

    # Define file name for figure (placed in the same folder as this script)
    filename = Path(__file__).parent / 'workshop07_ex01_transitory.pdf'

    # Plot simulation results
    plot_simulation(eq, sim_trans, filename=filename)

    # -- Scenario B: Persistent TFP shock --

    # Initialize empty TFP series
    z_pers = np.empty(T + 1)
    # Set initial TFP to steady-state value
    z_pers[0] = par.z
    # Drop TFP by 10% in period 1
    z_pers[1] = 0.9 * z_pers[0]
    # Subsequently, TFP evolves according to the process: z_t = (1-kappa)*z_{t-1} + kappa*1.0
    for t in range(2, T + 1):
        z_pers[t] = (1 - par.kappa) * z_pers[t - 1] + par.kappa * 1.0

    # Perform simulation
    sim_pers = simulate_olg(z_pers, eq)

    # Define file name for figure (placed in the same folder as this script)
    filename = Path(__file__).parent / 'workshop07_ex01_persistent.pdf'

    # Plot simulation results
    plot_simulation(eq, sim_pers, filename=filename)
