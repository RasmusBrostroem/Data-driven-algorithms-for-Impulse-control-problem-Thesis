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


# diffPros = DiffusionProcess(b=b, sigma=sigma)
# opStrat = OptimalStrategy(diffusionProcess=diffPros, rewardFunc=reward)
# thresholdStrat = OptimalStrategy(diffusionProcess=diffPros, rewardFunc=reward)
# dataStrat = DataDrivenImpulseControl(rewardFunc=reward)

y1, zeta = get_y1_and_zeta(reward)

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
# start = time()
# result = Parallel(n_jobs=-1)(delayed(simulate_dataDriven_vs_optimal)(T, sims) for T in Ts)
# print(f"Parallel took: {time()- start}")

# run = neptune.init_run(project='rasmusbrostroem/DiffusionControl', with_id="DIF-27", mode="read-only")
# print(run["Metrics/regret"].fetch_values())


