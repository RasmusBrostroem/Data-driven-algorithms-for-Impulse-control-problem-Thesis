from scipy.integrate import odeint, quad
from diffusionProcess import b, sigma, DiffusionProcess
from strategies import DataDrivenImpulseControl, reward, get_y1_and_zeta, OptimalStrategy
from time import time
import numpy as np
from collections.abc import Iterable
from functools import partial
import cProfile
import pstats
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from sklearn.neighbors import KernelDensity
from statsmodels.nonparametric.kde import KDEUnivariate
from scipy.stats import gaussian_kde

import inspect
import neptune

# difPros = DiffusionProcess(b, sigma)
# optStrat = OptimalStrategy(diffusionProcess=difPros, rewardFunc=reward)

# x, t = difPros.EulerMaruymaMethod(100, 0.01, 0)

# T=100
# dataStrat = DataDrivenImpulseControl(rewardFunc=reward, bandwidth=1/np.sqrt(T))
# y1, zeta = get_y1_and_zeta(reward)


diffPros = DiffusionProcess(b=b, sigma=sigma)
#opStrat = OptimalStrategy(diffusionProcess=diffPros, rewardFunc=reward)
dataStrat = DataDrivenImpulseControl(rewardFunc=reward, sigma=sigma)
dataStrat.bandwidth = 1/np.sqrt(100)

y1, zeta = get_y1_and_zeta(reward)

start = time()
dataStrat.simulate(diffpros=diffPros, T=100, dt=0.01)
print(f"Data simulation took {time()-start} seconds")




