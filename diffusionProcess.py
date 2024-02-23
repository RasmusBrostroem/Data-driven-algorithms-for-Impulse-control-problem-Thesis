import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Union
from scipy.optimize import minimize_scalar, fsolve
from scipy.integrate import quad, nquad, dblquad
from functools import partial
from collections.abc import Iterable
import math
from mpmath import hyp2f2

# Define the drift function and diffusion coefficient
def b(x: float, t: float) -> float:
    # return the drift function evaluated at x and t
    return -x/2

def sigma(x: float, t: float) -> float:
    # return the diffusion coefficient evaluated at x and t
    return 1

def EulerMaruymaMethod(b: Callable[[float, float], float],
                       sigma: Callable[[float, float], float],
                       T: float,
                       dt: float,
                       x0: float) -> Union[np.array, np.array]:
    """Simmulating the Itô diffusion process using Euler Maruyma Method
    See https://en.wikipedia.org/wiki/Euler%E2%80%93Maruyama_method

    Args:
        b (Callable[[float, float], float]): The drift function
        sigma (Callable[[float, float], float]): The diffusion coefficient
        T (float): Number of time the process should run
        dt (float): The increment of time before simulating new value
        x0 (float): Initial value of process

    Returns:
        Union[np.array, np.array]: 
            x-values: the simulated values of the process
            t-values: the time steps, where simulated values were generated
    """
    N = int(T/dt) # number of time steps
    x = np.zeros(N+1)
    t = np.zeros(N+1)
    x[0] = x0
    t[0] = 0.0

    for n in range(N):
        t[n+1] = t[n] + dt
        z = np.random.normal(loc=0.0, scale=np.sqrt(dt))
        x[n+1] = x[n] + b(x[n], t[n])*dt + sigma(x[n], t[n])*z

    return x, t


def diffusionProcessStep(b: Callable[[float, float], float],
                         sigma: Callable[[float, float], float],
                         x: float,
                         t: float,
                         dt: float) -> float:
    """_summary_

    Args:
        b (Callable[[float, float], float]): _description_
        sigma (Callable[[float, float], float]): _description_
        x (float): _description_
        t (float): _description_
        dt (float): _description_

    Returns:
        float: _description_
    """
    return x + b(x, t)*dt + sigma(x, t)*np.random.normal(loc=0.0, scale=np.sqrt(dt))

def reward(x):
    return 1 - np.abs(1-x)**2

def get_y1_and_zeta(g):
    roots = fsolve(g, [0, 10000])
    f = lambda y: y if reward(y) > 0 else np.inf
    result = minimize_scalar(f, bounds=(0, max(roots)), method="bounded", options={'xatol': 1e-8})
    y1 = round(result.x, 5)
    f = lambda y: -reward(y) if reward(y) > 0 else np.inf
    result = minimize_scalar(f, bounds=(0, max(roots)), method="bounded", options={'xatol': 1e-8})
    zeta = round(result.x, 5)

    return y1, zeta

def getC_b_sigma(b, sigma):
    def inner_integral(u, b, sigma):
        f = lambda y: 2*b(y, 0)/sigma(y, 0)**2
        return quad(f, 0, u)[0]
    
    f = lambda u: 1/sigma(u,0)**2 * np.exp(inner_integral(u, b, sigma))
    return quad(f, -np.inf, np.inf)[0]

def invariant_density(x, b, sigma):
    def integral(x, b, sigma):
        f = lambda y: 2*b(y, 0)/sigma(y, 0)**2
        if isinstance(x, Iterable):
            return np.array(list(map(partial(quad, f, 0), x)))[:, 0]
        
        return quad(f, 0, x)[0]

    
    C_b_sigma = getC_b_sigma(b, sigma)

    return 1/C_b_sigma * np.exp(integral(x, b, sigma))

def xi(x, b, sigma):
    def inner_integral(y, b, sigma):
        return quad(invariant_density, -np.inf, y, args=(b, sigma))[0]
    
    f = lambda y: 1/(sigma(y,0)**2 * invariant_density(y, b, sigma)) * inner_integral(y, b, sigma)
    return 2*quad(f, 0, x)[0]

def erfi(x, N):
    f = lambda n: x**(2*n+1)/(math.factorial(n)*(2*n+1))
    return 2/np.sqrt(np.pi) * sum(map(f, range(N)))

def xi_teoretial(x):
    return float(x**2 * hyp2f2(1,1,3/2, 2, x**2/2) + np.pi * erfi(x/np.sqrt(2), 10))

if __name__ == "__main__":
    # x, t = EulerMaruymaMethod(b, sigma, 100, 0.01, 0)
    # # Plot the simulated values
    # plt.plot(t, x)
    # plt.xlabel('Time')
    # plt.ylabel('Value')
    # plt.show()
    f = lambda y: -reward(y)/xi_teoretial(y)

    y1, zeta = get_y1_and_zeta(reward)
    y = np.linspace(y1+0.00001, zeta, 20)
    gs = list(map(reward, y))
    # xif = lambda y: xi(y, b, sigma)
    # xis = list(map(xif, y))
    xis_theo = list(map(xi_teoretial, y))
    print(gs)
    print(xis_theo)
    # plt.plot(y, xis, label="Calculated")
    # plt.show()
    # plt.plot(y, xis_theo, label="Theoretical")
    # plt.legend()
    # plt.show()

    vals = [g/z for g, z in zip(gs,xis_theo)]
    result = minimize_scalar(f, bounds=(y1, zeta), method="bounded", options={'xatol': 1e-8})
    y_star = result.x
    print(f"Optimal stopping time y: {y_star}")
    print(f"Optimal reward per expected time unit: {-f(y_star)}")
    plt.plot(y, gs)
    plt.show()
    plt.plot(y, xis_theo)
    plt.show()
    plt.plot(y, vals)
    plt.show()