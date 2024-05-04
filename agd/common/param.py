from math import log, sqrt, exp
from scipy.optimize import fsolve


def dp_to_zcdp(eps, delta):
    def eq_epsilon(rho):
        if rho <= 0.0:
            rhs = rho
        else:
            rhs = rho + 2.0 * sqrt(rho * log(1.0/delta))

        return eps - rhs
    rho = fsolve(eq_epsilon, 0.0)
    return rho[0]


def compute_sigma(eps, delta, sens):
    sigma = sens / eps
    sigma *= sqrt(2.0 * log(1.25 / delta))
    return sigma


def compute_epsilon(rho):
    return sqrt(2.0*rho)
