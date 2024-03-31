from scipy.integrate import odeint, quad
from diffusionProcess import b, sigma, DiffusionProcess
from strategies import DataDrivenImpulseControl, reward, get_y1_and_zeta, OptimalStrategy, generate_reward_func
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

T = 100

diffPros = DiffusionProcess(b=b, sigma=sigma)
opStrat = OptimalStrategy(diffusionProcess=diffPros, rewardFunc=reward)
dataStrat = DataDrivenImpulseControl(rewardFunc=reward, sigma=sigma)
dataStrat.bandwidth = 1/np.sqrt(T)

data, t = diffPros.EulerMaruymaMethod(T, 0.01, 0)

r = generate_reward_func(5)

def plot_reward_xi_obj():
    difPros = DiffusionProcess(b, sigma)
    optStrat = OptimalStrategy(difPros, r)
    y1, zeta = get_y1_and_zeta(g=r)
    print(f"y1 = {y1} and zeta = {zeta}")

    y = np.linspace(y1, zeta*2, 100)
    gs = r(y)

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


print(f"reward at 0 = {r(0)}")

#plot_reward_xi_obj()

print(inspect.getsource(r))
#run = neptune.init_run(project='rasmusbrostroem/DiffusionControl')
# def simulate_dataDriven_vs_optimal(T, sims):
#     run = neptune.init_run(project='rasmusbrostroem/DiffusionControl')
#     diffusionProcess = DiffusionProcess(b=b, sigma=sigma)
#     OptimalStrat = OptimalStrategy(diffusionProcess=diffusionProcess, rewardFunc=reward)
#     DataStrat = DataDrivenImpulseControl(rewardFunc=reward)

#     run["AlgoParams"] = {
#         "kernelMethod": DataStrat.kernel_method,
#         "bandwidth": "1/sqrt(T)",
#         "a": DataStrat.a,
#         "M1": DataStrat.M1,
#     }
#     run["ModelParams"] = {
#         "driftFunc": inspect.getsource(diffusionProcess.b),
#         "diffusionCoef": inspect.getsource(diffusionProcess.sigma),
#         "rewardFunc": inspect.getsource(DataStrat.g)
#     }


#     DataStrat.bandwidth = 1/np.sqrt(T)
#     for s in range(sims):
#         diffusionProcess.generate_noise(T, 0.01)
#         dataReward, S_T = DataStrat.simulate(diffpros=diffusionProcess, T=T, dt=0.01)
#         opt_reward = OptimalStrat.simulate(diffpros=diffusionProcess, T=T, dt=0.01)

#         run[f"Metrics/T"].append(T)
#         run[f"Metrics/simNr"].append(s)
#         run[f"Metrics/S_T"].append(value=S_T, step=T)
#         run[f"Metrics/dataDriveReward"].append(value=dataReward, step=T)
#         run[f"Metrics/OptimalStratReward"].append(value=opt_reward, step=T)
#         run[f"Metrics/regret"].append(value=opt_reward-dataReward, step=T)
#         # output.append({
#         #     "T": T,
#         #     "simNr": s,
#         #     "S_T": S_T,
#         #     "data_reward": dataReward,
#         #     "optimal_reward": opt_reward,
#         #     "regret": opt_reward-dataReward
#         # })

# Ts = [100*i for i in range(1,6)]
# sims = 10
# diffPros.generate_noise(T, 0.01)

# start = time()
# r, S_T = dataStrat.simulate(diffPros, T, 0.01)
# t1 = time()
# r1, S_T1 = dataStrat.simulate_new(diffPros, T, 0.01)
# t2 = time()
# r2, S_T2 = dataStrat.simulate_new_new(diffPros, T, 0.01)
# end = time()

# print(f"Original simulation took = {t1-start} with reward {r} and exploration time {S_T}")
# print(f"New simulation took = {t2-t1} with reward {r1} and exploration time {S_T1}")
# print(f"New New simulation took = {end-t2} with reward {r2} and exploration time {S_T2}")


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




