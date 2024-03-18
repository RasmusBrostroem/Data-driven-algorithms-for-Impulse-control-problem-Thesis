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

from sklearn.neighbors import KernelDensity
from statsmodels.nonparametric.kde import KDEUnivariate
from scipy.stats import gaussian_kde

difPros = DiffusionProcess(b, sigma)
optStrat = OptimalStrategy(diffusionProcess=difPros, rewardFunc=reward)

x, t = difPros.EulerMaruymaMethod(100, 0.01, 0)

T=100
dataStrat = DataDrivenImpulseControl(rewardFunc=reward, bandwidth=1/np.sqrt(T))
y1, zeta = get_y1_and_zeta(reward)



# cProfile.run("dataStrat.simulate(diffpros=difPros, T=100, dt=0.01)", "threshold_stats")
# p = pstats.Stats("threshold_stats")
# p.sort_stats("cumulative").print_stats()




