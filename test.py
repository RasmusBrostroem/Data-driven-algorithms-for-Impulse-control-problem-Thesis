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

from scipy.integrate import quad

from sklearn.neighbors import KernelDensity
from statsmodels.nonparametric.kde import KDEUnivariate
from scipy.stats import gaussian_kde

import inspect

from itertools import islice
from sortedcontainers import SortedDict



diffPros = DiffusionProcess(b=b, sigma=sigma)
opStrat = OptimalStrategy(diffusionProcess=diffPros, rewardFunc=reward)
dataStrat = DataDrivenImpulseControl(rewardFunc=reward, sigma=sigma)
dataStrat.bandwidth = 1/np.sqrt(100)

data, t = diffPros.EulerMaruymaMethod(5000, 0.01, 0)




fitStart = time()
dataStrat.fit(data)
fitEnd = time()
print(f"Original fit took {fitEnd-fitStart} seconds")

start = time()
estimate, nevals, nitr = dataStrat.estimate_threshold()
t1 = time()
estimate_new, nevals_new, nitr_new = dataStrat.estimate_threshold_new()
t2 = time()


dataStrat.clear_sorted_dict()

fitStart_new = time()
dataStrat.fit_new(data)
fitEnd_new = time()
print(f"new fit took {fitEnd_new-fitStart_new} seconds")

t3 = time()
estimate_new_new, nevals_new_new, nitr_new_new = dataStrat.estimate_threshold_new_new()
t4 = time()



print(f"original estimate took {t1-start} whith {nevals} evaluations, which on average is {(t1-start)/nevals}")
print(f"new estimate took {t2-t1} whith {nevals_new} evaluations, which on average is {(t2-t1)/nevals_new}")
print(f"new new estimate took {t4-t3} whith {nevals_new_new} evaluations, which on average is {(t4-t3)/nevals_new_new}")

print(f"Optimal threshold = {opStrat.y_star}")
print(f"old estimate = {estimate}")
print(f"new estimate = {estimate_new}")
print(f"new new estimate = {estimate_new_new}")





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




