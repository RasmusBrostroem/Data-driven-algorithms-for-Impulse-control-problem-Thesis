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

def reward(x):
    return 1/2 - np.abs(1-x)**2
    # return 10 - np.abs(4-3*x)**2

def get_y1_and_zeta(g):
    roots = fsolve(g, [0, 10000])
    f = lambda y: y if reward(y) > 0 else np.inf
    result = minimize_scalar(f, bounds=(0, max(roots)), method="bounded", options={'xatol': 1e-8})
    y1 = result.x
    f = lambda y: -reward(y) if reward(y) > 0 else np.inf
    result = minimize_scalar(f, bounds=(0, max(roots)), method="bounded", options={'xatol': 1e-8})
    zeta = result.x

    return y1, zeta

class DiffusionProcess():
    def __init__(self, b, sigma) -> None:
        self.b = b
        self.sigma = sigma


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
            x[n+1] = x[n] + self.b(x[n], t[n])*dt + self.sigma(x[n], t[n])*z

        return x, t

    def step(self,
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
        return x + self.b(x, t)*dt + self.sigma(x, t)*np.random.normal(loc=0.0, scale=np.sqrt(dt))


    def getC_b_sigma(self):
        def inner_integral(u):
            f = lambda y: 2*self.b(y, 0)/self.sigma(y, 0)**2
            return quad(f, 0, u)[0]
        
        f = lambda u: 1/self.sigma(u,0)**2 * np.exp(inner_integral(u))
        return quad(f, -np.inf, np.inf)[0]

    def invariant_density(self, x):
        def integral(x):
            f = lambda y: 2*self.b(y, 0)/self.sigma(y, 0)**2
            if isinstance(x, Iterable):
                return np.array(list(map(partial(quad, f, 0), x)))[:, 0]
            return quad(f, 0, x)[0]

        C_b_sigma = self.getC_b_sigma()
        return 1/C_b_sigma * np.exp(integral(x))

    def xi(self, x):
        def inner_integral(y):
            if isinstance(y, Iterable):
                return np.array(list(map(partial(quad, self.invariant_density, -np.inf), y)))[:, 0]
            return quad(self.invariant_density, -np.inf, y)[0]
        
        f = lambda y: 1/(self.sigma(y,0)**2 * self.invariant_density(y)) * inner_integral(y)
        if isinstance(x, Iterable):
            return 2*np.array(list(map(partial(quad, f, 0), x)))[:, 0]
        return 2*quad(f, 0, x)[0]

    def erfi(self, x, N):
        f = lambda n: x**(2*n+1)/(math.factorial(n)*(2*n+1))
        return 2/np.sqrt(np.pi) * sum(map(f, range(N)))

    def xi_theoretical(self, x):
        f = lambda x: float(x**2 * hyp2f2(1,1,3/2, 2, x**2/2) + np.pi * self.erfi(x/np.sqrt(2), 10))
        if isinstance(x, Iterable):
            return list(map(f, x))
        return f(x)

class OptimalStrategy():
    def __init__(self, diffusionProcess, rewardFunc):
        self.difPros = diffusionProcess
        self.g = rewardFunc
        self.y1, self.zeta = get_y1_and_zeta(rewardFunc)
        self.y_star = self.get_optimal_threshold()
        self.reward = 0

    def get_optimal_threshold(self):
        #obj = lambda y: -self.g(y)/self.difPros.xi_theoretical(y)
        obj = lambda y: -self.g(y)/self.difPros.xi_theoretical(y)
        result = minimize_scalar(obj, bounds=(self.y1, self.zeta), method="bounded", options={'xatol': 1e-8})
        return result.x
    
    def take_decision(self, x):
        if x >= self.y_star:
            self.reward += self.g(x)
            return True
        
        return False

def simulate_optimal_strategy(T=10, dt=0.01):
    difPros = DiffusionProcess(b, sigma)
    optStrat = OptimalStrategy(difPros, reward)
    print(f"Optimal threshold: {optStrat.y_star}")
    t = [0]
    X_strat = [0]
    x_plot = []
    t_plot = []
    i = 0
    while t[i] < T:
        if optStrat.take_decision(x=X_strat[i]):
            t_plot.append(t)
            x_plot.append(X_strat)
            t = [t[i]]
            X_strat = [0]
            i = 0

        t.append(t[i] + dt)
        X_strat.append(difPros.step(X_strat[i], t[i], dt))
        i += 1

    total_reward = optStrat.reward
    print(f"Total reward was: {total_reward}")
    for t, x in zip(t_plot, x_plot):
        plt.plot(t,x, color="b")
        
    plt.show()
    return

def plot_uncontrolled_diffusion(T=100, dt=0.01, x0=0):
    difPros = DiffusionProcess(b, sigma)
    x, t = difPros.EulerMaruymaMethod(T, dt, x0)
    plt.plot(t, x)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.show()
    return

def plot_reward_xi_obj():
    difPros = DiffusionProcess(b, sigma)
    optStrat = OptimalStrategy(difPros, reward)
    y1, zeta = get_y1_and_zeta(g=reward)
    print(f"y1 = {y1} and zeta = {zeta}")

    y = np.linspace(y1, zeta*2, 20)
    gs = reward(y)

    xis = difPros.xi(y)
    xis_theo = difPros.xi_theoretical(y)

    vals = gs/xis

    y_star = optStrat.get_optimal_threshold()

    plt.plot(y, gs)
    plt.title("Reward function")
    plt.show()

    fig, (ax1, ax2) = plt.subplots(1,2)
    fig.suptitle("Expected time before reaching value")
    ax1.plot(y,xis)
    ax1.set_title("Calculated xi")
    ax2.plot(y,xis_theo)
    ax2.set_title("Theoretical xi")
    plt.show()

    print(f"Optimal threshold = {y_star}")
    plt.plot(y, vals)
    plt.title("Objective function")
    plt.show()
    return

if __name__ == "__main__":
    plot_uncontrolled_diffusion()

    # simulate_optimal_strategy()

    # plot_reward_xi_obj()