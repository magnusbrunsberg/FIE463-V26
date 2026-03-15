import numpy as np
from scipy.optimize import root_scalar, minimize_scalar
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class Parameters:
    beta: float = 0.99**30
    gamma: float = 2.1
    tau: float = 0.3
    phi: float = 0.5
    alpha: float = 0.36
    delta: float = 1.0 - 0.94**30
    z: float = 1.0
    Nw: float = 0.8
    Nk: float = 0.2
    tau_a: float = 0.0
    pi_a: float = 0.5
    tau_a_iid: float = 0.05

def u_prime(c, gamma):
    return c**(-gamma)

def euler_err(K_next, K, par: Parameters):
    L_t = par.Nw / 2
    k_t = K / L_t
    r_t = par.z * par.alpha * k_t**(par.alpha - 1) - par.delta
    
    k_next = K_next / L_t
    r_next = par.z * par.alpha * k_next**(par.alpha - 1) - par.delta
    
    a_prev = 2 * K / par.Nk
    a_t = 2 * K_next / par.Nk
    
    Ta_t = par.tau_a * (1 + r_t) * K
    Ta_next = par.tau_a * (1 + r_next) * K_next
    
    b_t = par.phi * (1 - par.tau_a) * (1 + r_t) * a_prev
    ck_y_t = b_t + Ta_t - a_t
    
    if ck_y_t <= 0:
        return 1e10
        
    ck_o_next = (1 - par.phi) * (1 - par.tau_a) * (1 + r_next) * a_t + Ta_next
    
    if ck_o_next <= 0:
        return -1e10
        
    LHS = u_prime(ck_y_t, par.gamma)
    RHS = par.beta * (1 + r_next) * (1 - par.tau_a) * (1 - par.phi) * u_prime(ck_o_next, par.gamma)
    
    return LHS - RHS

@dataclass
class SteadyState:
    K: float
    Y: float
    w: float
    r: float
    cw_y: float
    cw_o: float
    ck_y: float
    ck_o: float
    a: float
    b: float
    p: float
    L: float
    Ta: float

def compute_steady_state(par: Parameters, K_guess=0.05):
    res = root_scalar(lambda K: euler_err(K, K, par), bracket=[1e-5, 1.0])
    if not res.converged:
        raise ValueError("Steady state not found")
    
    K = res.root
    L = par.Nw / 2
    k = K / L
    Y = par.z * K**par.alpha * L**(1 - par.alpha)
    r = par.z * par.alpha * k**(par.alpha - 1) - par.delta
    w = par.z * (1 - par.alpha) * k**par.alpha
    
    a = 2 * K / par.Nk
    Ta = par.tau_a * (1 + r) * K
    b = par.phi * (1 - par.tau_a) * (1 + r) * a
    
    p = par.tau * w
    
    cw_y = (1 - par.tau) * w + Ta
    cw_o = p + Ta
    
    ck_y = b + Ta - a
    ck_o = (1 - par.phi) * (1 - par.tau_a) * (1 + r) * a + Ta
    
    return SteadyState(K=K, Y=Y, w=w, r=r, cw_y=cw_y, cw_o=cw_o, ck_y=ck_y, ck_o=ck_o, a=a, b=b, p=p, L=L, Ta=Ta)

@dataclass
class Simulation:
    K: np.ndarray
    Y: np.ndarray
    w: np.ndarray
    r: np.ndarray
    cw_y: np.ndarray
    cw_o: np.ndarray
    ck_y: np.ndarray
    ck_o: np.ndarray
    a: np.ndarray
    b: np.ndarray
    st: np.ndarray

def simulate_olg(K0, T, par: Parameters):
    K = np.zeros(T + 1)
    K[0] = K0
    
    for t in range(T):
        res = root_scalar(lambda K_next: euler_err(K_next, K[t], par), bracket=[1e-5, 1.0])
        K[t+1] = res.root
        
    K_t = K[:-1]
    K_next = K[1:]
    
    L = par.Nw / 2
    Y = par.z * K_t**par.alpha * L**(1 - par.alpha)
    r = par.z * par.alpha * (K_t / L)**(par.alpha - 1) - par.delta
    w = par.z * (1 - par.alpha) * (K_t / L)**par.alpha
    
    a = 2 * K_next / par.Nk
    a_prev = 2 * K_t / par.Nk
    
    Ta = par.tau_a * (1 + r) * K_t
    b = par.phi * (1 - par.tau_a) * (1 + r) * a_prev
    
    st = a / (b + Ta)
    
    p = par.tau * w
    cw_y = (1 - par.tau) * w + Ta
    cw_o = p + Ta
    ck_y = b + Ta - a
    ck_o = (1 - par.phi) * (1 - par.tau_a) * (1 + r) * a_prev + Ta
    
    return Simulation(K=K_t, Y=Y, w=w, r=r, cw_y=cw_y, cw_o=cw_o, ck_y=ck_y, ck_o=ck_o, a=a, b=b, st=st)

if __name__ == "__main__":
    par = Parameters()
    ss = compute_steady_state(par)
    print(f"Base SS: K={ss.K:.4f}, Y={ss.Y:.4f}, r={ss.r:.4f}, w={ss.w:.4f}")
    
    print(f"Annualized interest rate: {(1 + ss.r)**(1/30) - 1:.4%}")
    
    sim = simulate_olg(ss.K / 2, 20, par)
    print(f"Final sim K: {sim.K[-1]:.4f} (expect {ss.K:.4f})")
    
    par_tax = Parameters(tau_a=0.05)
    ss_tax = compute_steady_state(par_tax)
    print(f"Tax SS: K={ss_tax.K:.4f}, Y={ss_tax.Y:.4f}")
    
    sim_tax = simulate_olg(ss.K, 20, par_tax)
    print(f"Tax Sim Final K: {sim_tax.K[-1]:.4f} (expect {ss_tax.K:.4f})")
