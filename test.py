from scipy.integrate import odeint, quad
from diffusionProcess import b, sigma, DiffusionProcess
#from diffusionProcess_cy import b, sigma, DiffusionProcess
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

from itertools import islice
from sortedcontainers import SortedDict



diffPros = DiffusionProcess(b=b, sigma=sigma)
#opStrat = OptimalStrategy(diffusionProcess=diffPros, rewardFunc=reward)
dataStrat = DataDrivenImpulseControl(rewardFunc=reward, sigma=sigma)
dataStrat.bandwidth = 1/np.sqrt(100)

data, t = diffPros.EulerMaruymaMethod(100, 0.01, 0)

dataStrat.fit(data)

start = time()
estimate = dataStrat.estimate_threshold()
t1 = time()
estimate_new = dataStrat.estimate_threshold_new()
end = time()

print(f"original estimate took {t1-start}")
print(f"new estimate took {end-t1}")



print(f"old estimate = {estimate}")
print(f"new estimate {estimate_new}")

#cProfile.run("dataStrat.estimate_threshold()", sort="cumtime")

# y1, zeta = get_y1_and_zeta(reward)

# start = time()
# diffPros.generate_noise(5000, 0.01)
# t = 0
# dt = 0.01
# x = 0
# while t <= 5000:
#     x = diffPros.step(x, t, dt)
#     t += dt
# # for i in range(10):
# #     x, t = diffPros.EulerMaruymaMethod(T=100, dt=0.01, x0=0)
# #     dataStrat.fit(x)
# #     dataStrat.estimate_threshold()
# print(f"Data simulation took {time()-start} seconds")




