import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Union
from scipy.optimize import minimize_scalar, fsolve
from scipy.integrate import quad
from scipy.interpolate import interp1d
from functools import partial
from collections.abc import Iterable
from mpmath import hyp2f2
from statsmodels.nonparametric.kde import KDEUnivariate
from sklearn.neighbors import KernelDensity
from statsmodels.distributions.empirical_distribution import ECDF
from collections.abc import Iterable

from diffusionProcess import DiffusionProcess, sigma, generate_linear_drift


def get_bandwidth(ST, a=1, p=-1/2, log=False):
    if log:
        return np.log(ST)**2 / np.sqrt(ST)
    
    if isinstance(a, str):
        return a
    
    return a*ST**p

def reward(x):
    return 7/10 - np.abs(1-x)**(1)
    # return 10 - np.abs(4-3*x)**2

def generate_reward_func(power: float, zeroVal: float):
    return lambda x: zeroVal - np.abs(1-x)**power

def get_y1_and_zeta(g):
    eps = 0
    roots = fsolve(g, [0, 2, 10])
    f1 = lambda y: y if g(y) > 0 else np.inf
    result = minimize_scalar(f1, bounds=(0, max(roots)+eps), method="bounded", options={'xatol': 1e-8})
    y1 = result.x
    f2 = lambda y: -g(y) if g(y) > 0 else np.inf
    result = minimize_scalar(f2, bounds=(0, max(roots)+eps), method="bounded", options={'xatol': 1e-8})
    zeta = result.x

    return y1, zeta


class OptimalStrategy():
    def __init__(self, diffusionProcess, rewardFunc):
        self.difPros = diffusionProcess
        self.g = rewardFunc
        self.y1, self.zeta = get_y1_and_zeta(rewardFunc)
        self.y_star = self.get_optimal_threshold()
        self.reward = 0
        self.nrDecisions = 0

    def get_optimal_threshold(self):
        eps = 0.0001
        obj = lambda y: -self.g(y)/self.difPros.xi(y)
        #obj = lambda y: -self.g(y)/self.difPros.xi_theoretical(y)
        result = minimize_scalar(obj, bounds=(self.y1-eps, self.zeta+eps), method="bounded", options={'xatol': 1e-6, 'maxiter': 25})
        return result.x
    
    def take_decision(self, x):
        if x >= self.y_star:
            self.nrDecisions += 1
            self.reward += self.g(x)
            return True
        
        return False
    
    def simulate(self, diffpros: DiffusionProcess, T: int, dt: float) -> float:
        self.reward = 0
        self.nrDecisions = 0
        t = 0
        X = 0
        while t < T:
            if self.take_decision(x=X):
                X = 0
            X = diffpros.step(x=X, t=t, dt=dt)
            t += dt

        return self.reward, self.nrDecisions
    
class DataDrivenImpulseControl():
    def __init__(self, rewardFunc, sigma, **kwargs):
        self.g = rewardFunc
        self.sigma = sigma
        self.y1, self.zeta = get_y1_and_zeta(rewardFunc)

        # Exploration form
        self.ST_form = lambda t: t**(2/3)

        # Kernel attributes
        self.kernel_method = "gaussian"
        self.bandwidthFunc = lambda t: 1/np.sqrt(t)
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

        self.xi_evaluated_x = {}
        self.pdf_evaluated_x = None

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

    def clear_cache(self):
        self.xi_evaluated_x.clear()
        self.pdf_evaluated_x = None
    
    def kernel_fit(self, data: list[float]) -> None:
        self.kde = KernelDensity(kernel=self.kernel_method, bandwidth=self.bandwidth).fit(np.array(data, dtype=object).reshape(-1,1))

    def ecdf_fit(self, data: list[float]) -> None:
        self.cdf = ECDF(data)

    def fit(self, data: list[float]) -> None:
        self.clear_cache()
        self.kernel_fit(data)
        self.ecdf_fit(data)
        xs = np.linspace(0, self.zeta+0.1, int(100*(self.zeta+0.1)))
        pdf_values = np.array([self.pdf_eval(x) for x in xs])
        self.pdf_evaluated_x = interp1d(xs, pdf_values, kind="linear", assume_sorted=True, bounds_error=False)
    
    def cdf_eval(self, x: Union[list[float], float]) -> Union[list[float], float]:
        return self.cdf(x)
    
    def pdf_eval(self, x: float) -> float:
        return np.exp(self.kde.score_samples(np.array(x, dtype=object).reshape(1,-1)))[0]
    
    def MISE_eval_pdf(self, diffpros: DiffusionProcess):
        points = 5000
        start = -10
        end = 10
        vals = np.linspace(start, end, points)
        f = lambda x: (self.pdf_eval(x) - diffpros.invariant_density(x))**2
        MISE = sum([f(v)*(end-start)/points for v in vals])
        return MISE
        #return quad(f, -10, 10, limit=2500, epsrel=1e-3, points=np.linspace(-5, 5, 1000))[0]
    
    def MISE_eval_cdf(self, diffpros: DiffusionProcess):
        points = 5000
        start = -10
        end = 10
        vals = np.linspace(start, end, points)
        f = lambda x: (self.cdf_eval(x) - diffpros.invariant_distribution(x))**2
        MISE = sum([f(v)*(end-start)/points for v in vals])
        return MISE
        #return quad(f, -10, 10, limit=2500, epsrel=1e-3, points=np.linspace(-5, 5, 1000))[0]
    
    def KL_eval(self, diffpros: DiffusionProcess):
        eps = 0.0000001
        f = lambda x: self.pdf_eval(x)*np.log((self.pdf_eval(x)+eps)/(diffpros.invariant_density(x)+eps))
        return quad(f, -np.inf, np.inf, limit=250, epsabs=1e-3)[0]

    def pdf_eval_interpolate(self, x):
        return self.pdf_evaluated_x(x)
    
    def xi_eval(self, x):
        f = lambda y: self.cdf_eval(y)/(max(self.pdf_eval_interpolate(y), self.a)*self.sigma(y)**2)

        if not self.xi_evaluated_x:
            xi_estimate = 2*quad(f, 0, x, limit=250, epsabs=1e-2)[0]
            self.xi_evaluated_x[x] = xi_estimate
            return np.maximum(xi_estimate, self.M1)

        closest_x_evaluated, xi_val = min(self.xi_evaluated_x.items(), key=lambda y: abs(x - y[0]))

        if closest_x_evaluated <= x:
            xi_estimate = xi_val + 2*quad(f, closest_x_evaluated, x, limit=250, epsabs=1e-2)[0] 
            
            self.xi_evaluated_x[x] = xi_estimate
            return np.maximum(xi_estimate, self.M1)
        
        xi_estimate = xi_val - 2*quad(f, x, closest_x_evaluated, limit=250, epsabs=1e-2)[0] 
        
        self.xi_evaluated_x[x] = xi_estimate
        return np.maximum(xi_estimate, self.M1)
    
    def estimate_threshold(self) -> float:
        eps = 0.01
        obj = lambda y: -self.g(y)/self.xi_eval(y)
        result = minimize_scalar(obj, bounds=(self.y1-eps, self.zeta+eps), method="bounded", options={'xatol': 1e-2})
        return result.x
    
    def simulate(self, diffpros: DiffusionProcess, T: int, dt: float):
        self.kde = None
        self.cdf = None

        data = []
        X = 0
        t = 0
        S_t = 0
        nrDecisions = 0
        reachedZeta = False
        exploring = True
        threshold = None
        cumulativeReward = 0
        thresholds_and_Sts = []

        while t < T:
            if exploring:
                data.append(X)
                S_t += dt
                if X >= self.zeta:
                    reachedZeta = True
            
            if reachedZeta and X <= 0:
                self.bandwidth = self.bandwidthFunc(S_t)
                self.fit(data)
                threshold = self.estimate_threshold()
                thresholds_and_Sts.append((threshold,S_t))
                exploring = False
                reachedZeta = False
            
            if not exploring and X >= threshold:
                nrDecisions += 1
                cumulativeReward += self.g(X)
                X = 0
                if S_t < self.ST_form(t):
                    exploring = True
            
            X = diffpros.step(X, t, dt)
            t += dt

        return cumulativeReward, S_t, thresholds_and_Sts, nrDecisions

if __name__ == "__main__":
    pass