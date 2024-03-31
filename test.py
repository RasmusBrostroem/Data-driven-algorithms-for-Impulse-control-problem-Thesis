from scipy.integrate import odeint, quad
from diffusionProcess import drift, sigma, DiffusionProcess, b_linear_generate
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

# def simulate_dataDriven_vs_optimal2(rewardPower, Ts, sims, diffusionProcess):
#     output = []
#     r = generate_reward_func(power=rewardPower)
#     OptimalStrat = OptimalStrategy(diffusionProcess=diffusionProcess, rewardFunc=r)
#     DataStrat = DataDrivenImpulseControl(rewardFunc=r, sigma=sigma)
#     for T in Ts:
#         DataStrat.bandwidth = 1/np.sqrt(T)
#         for s in range(sims):
#             diffusionProcess.generate_noise(T, 0.01)
#             dataReward, S_T = DataStrat.simulate(diffpros=diffusionProcess, T=T, dt=0.01)
#             opt_reward = OptimalStrat.simulate(diffpros=diffusionProcess, T=T, dt=0.01)

#             output.append({
#                 "Drift": inspect.getsource(diffusionProcess.b),
#                 "Sigma": inspect.getsource(diffusionProcess.sigma),
#                 "rewardFunc": inspect.getsource(r),
#                 "rewardPower": rewardPower,
#                 "T": T,
#                 "simNr": s,
#                 "kernel": DataStrat.kernel_method,
#                 "bandwidth": "1/sqrt(T)",
#                 "a": DataStrat.a,
#                 "M1": DataStrat.M1,
#                 "S_T": S_T,
#                 "data_reward": dataReward,
#                 "optimal_reward": opt_reward,
#                 "regret": opt_reward-dataReward
#             })
    
#     return output


# diffPros = DiffusionProcess(b=drift, sigma=sigma)

# cProfile.run("simulate_dataDriven_vs_optimal2(2, [100, 200, 300], 50, diffPros)", sort="cumtime")



# T = 100

# b = b_linear_generate(1/3, True)

# diffPros = DiffusionProcess(b=b, sigma=sigma)
# opStrat = OptimalStrategy(diffusionProcess=diffPros, rewardFunc=reward)
# dataStrat = DataDrivenImpulseControl(rewardFunc=reward, sigma=sigma)
# dataStrat.bandwidth = 1/np.sqrt(T)

# #data, t = diffPros.EulerMaruymaMethod(T, 0.01, 0)

# r = generate_reward_func(1/2)


# def plot_reward_xi_obj():
#     difPros = DiffusionProcess(b, sigma)
#     optStrat = OptimalStrategy(difPros, r)
#     y1, zeta = get_y1_and_zeta(g=r)
#     print(f"y1 = {y1} and zeta = {zeta}")

#     y = np.linspace(y1, zeta*2, 100)
#     gs = r(y)

#     xis = difPros.xi(y)
#     #xis_theo = difPros.xi_theoretical(y)

#     vals = gs/xis

#     y_star = optStrat.get_optimal_threshold()

#     plt.plot(y, gs)
#     plt.title("Reward function")
#     plt.show()

#     # fig, (ax1, ax2) = plt.subplots(1,2)
#     # fig.suptitle("Expected time before reaching value")
#     # ax1.plot(y,xis)
#     # ax1.set_title("Calculated xi")
#     # ax2.plot(y,xis_theo)
#     # ax2.set_title("Theoretical xi")
#     plt.plot(y, xis)
#     plt.title("Expected hitting times")
#     plt.show()

#     print(f"Optimal threshold = {y_star}")
#     plt.plot(y, vals)
#     plt.title("Objective function")
#     plt.show()
#     return


# print(f"reward at 0 = {r(0)}")

# plot_reward_xi_obj()




























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










