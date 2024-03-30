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

import KDEpy

import inspect

from itertools import islice
from sortedcontainers import SortedDict

def time_function(func, args_list, repetitions=10):
    total_time = 0
    for _ in range(repetitions):
        start_time = time()
        func(*args_list)
        end_time = time()
        total_time += end_time - start_time
    average_time = total_time / repetitions
    return average_time

T = 1000


diffPros = DiffusionProcess(b=b, sigma=sigma)
opStrat = OptimalStrategy(diffusionProcess=diffPros, rewardFunc=reward)
dataStrat = DataDrivenImpulseControl(rewardFunc=reward, sigma=sigma)
dataStrat.bandwidth = 1/np.sqrt(T)

data, t = diffPros.EulerMaruymaMethod(T, 0.01, 0)


# ogfit = time_function(dataStrat.fit, [data])
# print(f"Original fit took {ogfit} on average")
# # ogEstimate = time_function(dataStrat.estimate_threshold, [])
# # print(f"Original estimate took {ogEstimate} on average")
# newEstimate = time_function(dataStrat.estimate_threshold_new, [])
# print(f"new estimate took {newEstimate} on average, which is a total average time of {newEstimate+ogfit}")

# dataStrat.clear_cache()
# newFit = time_function(dataStrat.fit_new, [data])
# print(f"New fit took {newFit} on average")
# newNewEstimate = time_function(dataStrat.estimate_threshold_new_new, [])
# print(f"new new estimate took {newNewEstimate} on average, which is a total average time of {newNewEstimate+newFit}")

# fitStart = time()
# dataStrat.fit(data)
# fitEnd = time()
# print(f"Original fit took {fitEnd-fitStart} seconds")

# start = time()
# estimate, nevals, nitr = dataStrat.estimate_threshold()
# t1 = time()
# estimate_new, nevals_new, nitr_new = dataStrat.estimate_threshold_new()
# t2 = time()

# dataStrat.clear_cache()

# fitStart_new = time()
# dataStrat.fit_new(data)
# fitEnd_new = time()
# print(f"New fit took {fitEnd_new-fitStart_new} seconds")

# t3 = time()
# estimate_new_new, nevals_new_new, nitr_new_new = dataStrat.estimate_threshold_new_new()
# t4 = time()

# print(f"original estimate took {t1-start} whith {nevals} evaluations, which on average is {(t1-start)/nevals}")
# print(f"new estimate took {t2-t1} whith {nevals_new} evaluations, which on average is {(t2-t1)/nevals_new}, but with fitting total time was: {t2-t1 + fitEnd-fitStart}")
# print(f"new new estimate took {t4-t3} whith {nevals_new_new} evaluations, which on average is {(t4-t3)/nevals_new_new}, but with fitting total time was: {t4-t3 + fitEnd_new-fitStart_new}")

# print(f"Optimal threshold = {opStrat.y_star}")
# print(f"old estimate = {estimate}")
# print(f"new estimate = {estimate_new}")
# print(f"new new estimate = {estimate_new_new}")

diffPros.generate_noise(T, 0.01)

start = time()
r, S_T = dataStrat.simulate(diffPros, T, 0.01)
t1 = time()
r1, S_T1 = dataStrat.simulate_new(diffPros, T, 0.01)
t2 = time()
r2, S_T2 = dataStrat.simulate_new_new(diffPros, T, 0.01)
end = time()

print(f"Original simulation took = {t1-start} with reward {r} and exploration time {S_T}")
print(f"New simulation took = {t2-t1} with reward {r1} and exploration time {S_T1}")
print(f"New New simulation took = {end-t2} with reward {r2} and exploration time {S_T2}")


# cProfile.run("dataStrat.estimate_threshold()", sort="cumtime")

# cProfile.run("dataStrat.estimate_threshold_new()", sort="cumtime")






# start = time()
# dataStrat.pdf_eval(xs)
# print(f"vector based evaluation = {time()-start}")





#cProfile.run("dataStrat.estimate_threshold()", sort="cumtime")



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




