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

def time_function(func, args_list, repetitions=10):
    total_time = 0
    for _ in range(repetitions):
        start_time = time()
        func(*args_list)
        end_time = time()
        total_time += end_time - start_time
    average_time = total_time / repetitions
    return average_time


def simulate_dataDriven_vs_optimal(Ts, sims, driftFunc, sigmaFunc, rewardFunc):
    run = neptune.init_run(project='rasmusbrostroem/DiffusionControl')
    runId = run["sys/id"].fetch()
    diffusionProcess = DiffusionProcess(b=driftFunc, sigma=sigmaFunc)
    OptimalStrat = OptimalStrategy(diffusionProcess=diffusionProcess, rewardFunc=rewardFunc)
    DataStrat = DataDrivenImpulseControl(rewardFunc=rewardFunc, sigma=sigmaFunc)

    run["AlgoParams"] = {
        "kernelMethod": DataStrat.kernel_method,
        "bandwidthMethod": "1/sqrt(T)",
        "a": DataStrat.a,
        "M1": DataStrat.M1
    }

    run["ModelParams"] = {
        "driftFunc": inspect.getsource(diffusionProcess.b),
        "diffusionCoef": inspect.getsource(diffusionProcess.sigma),
        "rewardFunc": inspect.getsource(DataStrat.g),
        "y_star": OptimalStrat.y_star,
        "y1": DataStrat.y1,
        "zeta": DataStrat.zeta
    }

    for s in range(sims):
        for T in Ts:
            DataStrat.bandwidth = 1/np.sqrt(T)
            diffusionProcess.generate_noise(T, 0.01)
            dataReward, S_T, thresholds_and_t = DataStrat.simulate(diffpros=diffusionProcess, T=T, dt=0.01)
            if len(thresholds_and_t) >= 1:
                thresholds, ts = zip(*thresholds_and_t)
                if len(thresholds) == 1:
                    run[f"Metrics/Sim{s}/Thresholds/{T}"].append(values=thresholds, steps=ts)
                else:
                    run[f"Metrics/Sim{s}/Thresholds/{T}"].extend(values=thresholds, steps=ts)
            opt_reward = OptimalStrat.simulate(diffpros=diffusionProcess, T=T, dt=0.01)

            run[f"Metrics/Sim{s}/T"].append(value=T)
            run[f"Metrics/Sim{s}/simNr"].append(value=s)
            run[f"Metrics/Sim{s}/S_T"].append(value=S_T, step=T)
            run[f"Metrics/Sim{s}/dataDriveReward"].append(value=dataReward, step=T)
            run[f"Metrics/Sim{s}/OptimalStratReward"].append(value=opt_reward, step=T)
            run[f"Metrics/Sim{s}/regret"].append(value=opt_reward-dataReward, step=T)



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


def plot_reward_xi_obj(C, A, power, zeroVal):
    driftFunc = generate_linear_drift(C, A)
    rewardFunc = generate_reward_func(power, zeroVal)
    sigmaFunc = sigma
    difPros = DiffusionProcess(driftFunc, sigmaFunc)
    optStrat = OptimalStrategy(difPros, rewardFunc)
    y1, zeta = get_y1_and_zeta(g=rewardFunc)
    print(f"y1 = {y1} and zeta = {zeta}")

    y = np.linspace(y1, zeta*2, 100)
    gs = rewardFunc(y)

    bs = np.fromiter(map(driftFunc,y), dtype=float)

    absDrift = np.abs(bs)
    driftLine = C*(1+np.abs(y))
    driftY = np.linspace(-A-1, A+1, 100)
    drifs = np.fromiter(map(driftFunc, driftY), dtype=float)
    sgnDrift = (drifs/sigma(driftY)**2)*np.sign(driftY)

    xis = difPros.xi(y)
    vals = gs/xis

    y_star = optStrat.get_optimal_threshold()
    print(f"Optimal threshold = {y_star}")

    plt.plot(y, absDrift, label = "Absolute drift")
    plt.plot(y, driftLine, label = "Drift Line under")
    plt.legend()
    plt.title("First condition on drift")
    plt.show()

    plt.plot(driftY, sgnDrift)
    plt.title("Second condition on drift")
    plt.show()

    plt.plot(y, bs)
    plt.title("Drift function from y1 to 2*zeta")
    plt.show()

    plt.plot(y, gs)
    plt.title("Reward function")
    plt.show()

    plt.plot(y, xis)
    plt.title("Expected hitting times")
    plt.show()

    plt.plot(y, vals)
    plt.title("Objective function")
    plt.show()
    return


# r = generate_reward_func(2, 99/100)

# y1, zeta = get_y1_and_zeta(r)
# print(f"y1 = {y1}")
# print(f"zeta = {zeta}")

plot_reward_xi_obj(10, 1/10, 2, 99/100)

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










