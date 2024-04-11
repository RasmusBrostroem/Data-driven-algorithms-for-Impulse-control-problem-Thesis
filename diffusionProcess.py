import numpy as np
from typing import Callable, Union
from scipy.integrate import quad, nquad, dblquad
from functools import partial
from collections.abc import Iterable
import math
from mpmath import hyp2f2

# Define the drift function and diffusion coefficient
def drift(x: float) -> float:
    # return the drift function evaluated at x and t
    return -x/2

def generate_linear_drift(C: float, A: float = 0):
    return lambda x: -C*x

def sigma(x: float) -> float:
    # return the diffusion coefficient evaluated at x and t
    return 1

class DiffusionProcess():
    def __init__(self, b, sigma) -> None:
        self.b = b
        self.sigma = sigma
        self.C_b_sigma_val = self.getC_b_sigma()
        self.noise = None

    def generate_noise(self, T, dt) -> None:
        self.noise = np.random.normal(loc=0.0, scale=np.sqrt(dt), size=int(T/dt))
        return

    def EulerMaruymaMethod(self,
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
            x[n+1] = x[n] + self.b(x[n])*dt + self.sigma(x[n])*z

        return x, t

    def step(self, x: float, t: float, dt: float) -> float:
        if not self.noise is None:
            return x + self.b(x)*dt + self.sigma(x)*self.noise[int(t/dt)]
        
        return x + self.b(x)*dt + self.sigma(x)*np.random.normal(loc=0.0, scale=np.sqrt(dt))

    def getC_b_sigma(self):
        def inner_integral(u):
            f = lambda y: 2*self.b(y)/self.sigma(y)**2
            return quad(f, 0, u, epsabs=1e-3, limit=250)[0]
        
        f = lambda u: 1/self.sigma(u)**2 * np.exp(inner_integral(u))
        return quad(f, -np.inf, np.inf, epsabs=1e-3, limit=250)[0]

    def invariant_density(self, x):
        def integral(x):
            f = lambda y: 2*self.b(y)/(self.sigma(y)**2)
            if isinstance(x, Iterable):
                return np.array(list(map(partial(quad, f, 0, limit=100, epsabs = 1e-3), x)))[:, 0]
            return quad(f, 0, x, epsabs=1e-3, limit=100)[0]
        if isinstance(x, Iterable):
            sigmas = np.array([self.sigma(x_val) for x_val in x])
            return 1/(self.C_b_sigma_val*sigmas**2) * np.exp(integral(x))
        
        return 1/(self.C_b_sigma_val*self.sigma(x)**2) * np.exp(integral(x))
    
    def invariant_distribution(self, x):
        return quad(self.invariant_density, -1000, x, epsabs=1e-3, limit=250, points=np.linspace(-5,5,3))[0]

    def xi(self, x):
        def inner_integral(y):
            if isinstance(y, Iterable):
                return np.array(list(map(partial(quad, self.invariant_density, -np.inf, epsabs=1e-3, limit=100), y)))[:, 0]
            return quad(self.invariant_density, -np.inf, y, epsabs=1e-3, limit=100)[0]
        
        f = lambda y: 1/(self.sigma(y)**2 * self.invariant_density(y)) * inner_integral(y)
        if isinstance(x, Iterable):
            return 2*np.array(list(map(partial(quad, f, 0, epsabs=1e-3, limit=100), x)))[:, 0]
        return 2*quad(f, 0, x, epsabs=1e-3, limit=100)[0]

    def erfi(self, x, N):
        f = lambda n: x**(2*n+1)/(math.factorial(n)*(2*n+1))
        return 2/np.sqrt(np.pi) * sum(map(f, range(N)))

    def xi_theoretical(self, x):
        f = lambda x: float(x**2 * hyp2f2(1,1,3/2, 2, x**2/2) + np.pi * self.erfi(x/np.sqrt(2), 10))
        if isinstance(x, Iterable):
            return list(map(f, x))
        return f(x)

