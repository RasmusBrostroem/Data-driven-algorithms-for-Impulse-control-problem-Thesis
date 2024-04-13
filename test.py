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


diff = DiffusionProcess(b=generate_linear_drift(0.1, 0), sigma=sigma)

dataStrat = DataDrivenImpulseControl(rewardFunc=generate_reward_func(1, 0.7), sigma=sigma)
dataStrat.bandwidth = 1/np.sqrt(300**(3/2))
dataStrat.kernel_method = "linear"
ST = 100
band = 1/np.sqrt(ST**(3/2))
data, t = diff.EulerMaruymaMethod(ST, 0.01, 0)
data = list(data)
y1, zeta = get_y1_and_zeta(generate_reward_func(1, 0.7))
vals = np.linspace(-10, 10, 5000)

dataStrat.fit(data=data)
f = lambda x: (dataStrat.pdf_eval(x)-diff.invariant_density(x))**2
M = [f(v) for v in vals]
print(sum(M)/(5000/20))
m3 = dataStrat.MISE_eval_pdf(diff)
print(m3)
M1 = quad(f, -100, 100, limit=2500, epsabs=1e-3, points=np.linspace(-5, 5, 1000))
print(M1)
plt.plot(vals, M)
plt.show()



# t1 = time()
# pdfVals = [dataStrat.pdf_eval(v) for v in vals]
# t2 = time()
# pdfInterVals = [dataStrat.pdf_eval_interpolate(v) for v in vals]
# t3 = time()
# fs = [diff.invariant_density(v) for v in vals]

# print(f"It took {t2-t1} to evaluate with normal pdf")
# print(f"It took {t3-t2} to evaluate with interpolation")

# plt.plot(vals, pdfVals, label="pdf")
# plt.plot(vals, pdfInterVals, label="interpolation")
# plt.legend()
# plt.show()
# # plt.plot(vals, pdfInterVals, label="Sklearn")
# # plt.plot(vals, fs, label="True")
# # plt.legend()
# # plt.show()

# print(f"Evaluation at 1 for pdf = {dataStrat.pdf_eval(0.5)}")
# print(f"Evaluation at 1 for interpolation = {dataStrat.pdf_eval_interpolate(0.5)}")
# print(f"True value at 1 = {diff.invariant_density(1)}")


#cumulativeReward, S_t, thresholds_and_Sts, nrDecisions = dataStrat.simulate(diff, 200, 0.01)

# powers = [1/2, 1, 2, 5]
# zeroVals = [7/10, 45/50, 99/100]
# Cs = [1/5, 1/2, 1, 4]
# As = [0]
# argList = list(product(Cs, As, powers, zeroVals))

# plot_reward_xi_obj(1/2, 2, 0.2, 1, 7/10)

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










