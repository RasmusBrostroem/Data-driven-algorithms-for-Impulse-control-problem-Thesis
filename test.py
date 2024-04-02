from scipy.integrate import odeint, quad
from diffusionProcess import drift, sigma, DiffusionProcess, generate_linear_drift
from strategies import DataDrivenImpulseControl, reward, get_y1_and_zeta, OptimalStrategy, generate_reward_func
from time import time
import numpy as np
from collections.abc import Iterable
from functools import partial
import cProfile
import pstats
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import neptune
from scipy.integrate import quad
from sklearn.neighbors import KernelDensity
from statsmodels.nonparametric.kde import KDEUnivariate
import inspect
from itertools import product
from simulations import plot_reward_xi_obj

def time_function(func, args_list, repetitions=10):
    total_time = 0
    for _ in range(repetitions):
        start_time = time()
        func(*args_list)
        end_time = time()
        total_time += end_time - start_time
    average_time = total_time / repetitions
    return average_time


powers = [1/2, 1, 2, 5]
zeroVals = [7/10, 45/50, 99/100]
Cs = [1/100, 1/2, 1, 4]
As = [0]
argList = list(product(Cs, As, powers, zeroVals))

for c, a, p, z in argList:
    print(f"C = {c}, power = {p} and zero_val = {z}")
    plot_reward_xi_obj(c, a, p, z, save_obj=False)

# d = generate_linear_drift(1/2, 0)
# r = generate_reward_func(1, 7/10)
# diffPros = DiffusionProcess(d, sigma)
# opStrat = OptimalStrategy(diffPros, r)
# dataStrat = DataDrivenImpulseControl(r, sigma)

# T = 1000
# dataStrat.bandwidth = 1/np.sqrt(T)

# data, t = diffPros.EulerMaruymaMethod(T, 0.01, 0)
# print(f"Optimal threshold = {opStrat.y_star}")

# dataStrat.fit(data)

# y_est = dataStrat.estimate_threshold()

# print(f"Estimated threshold = {y_est}")


# Ts = [100*i for i in range(1,3)]
# sims = 2
# r = generate_reward_func(2, 7/10)
# d = generate_linear_drift(2, 0)
# sig = sigma

# start = time()
# simulate_dataDriven_vs_optimal(Ts, sims, d, sig, r)
# print(f"simulation took = {time()-start}")




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





# r = generate_reward_func(2, 99/100)

# y1, zeta = get_y1_and_zeta(r)
# print(f"y1 = {y1}")
# print(f"zeta = {zeta}")

#plot_reward_xi_obj(1, 0.75, 2, 7/10)

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










