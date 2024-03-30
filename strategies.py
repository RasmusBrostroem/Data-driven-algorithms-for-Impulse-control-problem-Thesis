import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Union
from scipy.optimize import minimize_scalar, fsolve
from scipy.integrate import quad, nquad, dblquad
from functools import partial
from collections.abc import Iterable
import math
from mpmath import hyp2f2
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from statsmodels.nonparametric.kde import KDEUnivariate
from scipy.stats import ecdf
from collections.abc import Iterable

from functools import lru_cache

from diffusionProcess import DiffusionProcess


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


class OptimalStrategy():
    def __init__(self, diffusionProcess, rewardFunc):
        self.difPros = diffusionProcess
        self.g = rewardFunc
        self.y1, self.zeta = get_y1_and_zeta(rewardFunc)
        self.y_star = self.get_optimal_threshold()
        self.reward = 0

    def get_optimal_threshold(self):
        obj = lambda y: -self.g(y)/self.difPros.xi(y)
        #obj = lambda y: -self.g(y)/self.difPros.xi_theoretical(y)
        result = minimize_scalar(obj, bounds=(self.y1, self.zeta), method="bounded", options={'xatol': 1e-8})
        return result.x
    
    def take_decision(self, x):
        if x >= self.y_star:
            self.reward += self.g(x)
            return True
        
        return False
    
    def simulate(self, diffpros: DiffusionProcess, T: int, dt: float) -> float:
        self.reward = 0
        t = 0
        X = 0
        while t < T:
            if self.take_decision(x=X):
                X = 0
            X = diffpros.step(x=X, t=t, dt=dt)
            t += dt

        return self.reward
    
class DataDrivenImpulseControl():
    def __init__(self, rewardFunc, sigma, **kwargs):
        self.g = rewardFunc
        self.sigma = sigma
        self.y1, self.zeta = get_y1_and_zeta(rewardFunc)

        # Kernel attributes
        self.kernel_method = "gau"
        self.bandwidth = None
        self.bandwidth_start = 0.01
        self.bandwidth_end = 1
        self.bandwidth_increment = 0.02

        # bounds on xi and invariant density
        self.a = 0.000001
        self.M1 = 0.000001

        self.update_attributes_on_kwargs(**kwargs)

        self.kde = None
        self.cdf = None

    def update_attributes_on_kwargs(self, **kwargs):
        # Get a list of all predefined attributes
        allowed_keys = list(self.__dict__.keys())

        # Update attributes
        self.__dict__.update((key, value) for key, value in kwargs.items() 
                             if key in allowed_keys)
        
        # Raise error for attributes given, that does not exist
        # To NOT silently ignore rejected keys
        rejected_keys = set(kwargs.keys()) - set(allowed_keys)
        if rejected_keys:
            raise ValueError("Invalid arguments in constructor:{}".format(rejected_keys))
    

    def kernel_fit(self, data: list[float]) -> None:
        self.kde = KDEUnivariate(data)
        self.kde.fit(kernel=self.kernel_method, bw=self.bandwidth)

    def ecdf_fit(self, data: list[float]) -> None:
        res = ecdf(sample=data)
        self.cdf = res.cdf

    def fit(self, data: list[float]) -> None:
        self.kernel_fit(data)
        self.ecdf_fit(data)
    
    def cdf_eval(self, x: Union[list[float], float]) -> Union[list[float], float]:
        return self.cdf.evaluate(x)
    
    def pdf_eval(self, x: float) -> float:
        return self.kde.evaluate(x)[0]
    
    def MISE_eval(self, diffpros: DiffusionProcess):
        f = lambda x: (self.pdf_eval(x) - diffpros.invariant_density(x))**2
        return quad(f, self.y1, self.zeta, limit=250, epsrel=1e-8)[0]
    
    def KL_eval(self, diffpros: DiffusionProcess):
        eps = 0.0000001
        f = lambda x: self.pdf_eval(x)*np.log((self.pdf_eval(x)+eps)/(diffpros.invariant_density(x)+eps))
        return quad(f, self.y1, self.zeta, limit=250, epsabs=1e-3)[0]

    def xi_eval(self, x):
        f = lambda y: self.cdf_eval(y)/(max(self.pdf_eval(y), self.a)*self.sigma(y)**2)
        xi_estimate = 2*quad(f, 0, x, limit=250, epsabs=1e-3)[0]
        return np.maximum(xi_estimate, self.M1)
    
    def estimate_threshold(self) -> float:
        obj = lambda y: -self.g(y)/self.xi_eval(y)
        result = minimize_scalar(obj, bounds=(self.y1, self.zeta), method="bounded", options={'xatol': 1e-4})
        return result.x
    
    def simulate(self, diffpros: DiffusionProcess, T: int, dt: float) -> float:
        self.kde = None
        self.cdf = None

        data = []
        X = 0
        t = 0
        S_t = 0
        reachedZeta = False
        exploring = True
        threshold = None
        cumulativeReward = 0

        while t < T:
            if exploring:
                data.append(X)
                S_t += dt
                if X >= self.zeta:
                    reachedZeta = True
            
            if reachedZeta and X <= 0:
                self.fit(data)
                threshold = self.estimate_threshold()
                exploring = False
                reachedZeta = False
            
            if not exploring and X >= threshold:
                cumulativeReward += self.g(X)
                X = 0
                if S_t < t**(2/3):
                    exploring = True
            
            X = diffpros.step(X, t, dt)
            t += dt
        
        return cumulativeReward, S_t

if __name__ == "__main__":
    pass