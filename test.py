from scipy.integrate import odeint, quad
from diffusionProcess import b, sigma, DiffusionProcess
from strategies import DataDrivenImpulseControl, reward, get_y1_and_zeta, OptimalStrategy
from time import time
import numpy as np
from collections.abc import Iterable
from functools import partial
import cProfile
import matplotlib.pyplot as plt

from sklearn.neighbors import KernelDensity
from statsmodels.nonparametric.kde import KDEUnivariate
from scipy.stats import gaussian_kde

difPros = DiffusionProcess(b, sigma)
optStrat = OptimalStrategy(diffusionProcess=difPros, rewardFunc=reward)

Ts = [10*i for i in range(1,11)]
sims = 10
STs = []
regrets = []
for T in Ts:
    print(T)
    regret = 0
    STsims = 0
    for i in range(sims):
        print(i)
        dataStrat = DataDrivenImpulseControl(rewardFunc=reward, bandwidth=1/np.sqrt(T))
        cumReward, S_T = dataStrat.simulate(diffpros=difPros, T=T, dt=0.01)
        optReward = optStrat.simulate(diffpros=difPros, T=T, dt=0.01)
        STsims += S_T
        regret += optReward-cumReward
    
    regrets.append(regret/sims)
    STs.append(STsims/sims)

plt.plot(Ts, regrets)
plt.show()
plt.plot(Ts, STs)
plt.show()
# x, t = difPros.EulerMaruymaMethod(100, 0.01, 0)
# x = list(x)
# y1, zeta = get_y1_and_zeta(reward)
# vals = np.linspace(y1, zeta*2, 20)

# dataStrat.fit(x)

# y_star = optStrat.get_optimal_threshold()
# y_hat = dataStrat.estimate_threshold()

# xis = dataStrat.xi_eval(vals)
# xis_theo = difPros.xi_theoretical(vals)

# fig, (ax1, ax2) = plt.subplots(1,2, sharey=True)
# ax1.plot(vals, xis)
# ax1.set_title("Estimated expected hitting times")
# ax2.plot(vals, xis_theo)
# ax2.set_title("Theoretical expected hitting times")
# fig.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.show()


# print(f"Optimal threshold = {y_star}")
# print(f"Estimated threshold = {y_hat}")


# start_sklearn = time()
# kde = KernelDensity(kernel="gaussian", bandwidth=1/np.sqrt(10))
# kde.fit(x[:, None])
# logprob = kde.score_samples(vals[:, None])
# probs_sklearn = np.exp(logprob)
# end_sklearn = time()

# start_stats = time()
# kde = KDEUnivariate(x)
# kde.fit(kernel="gau", bw=1/np.sqrt(10), fft=False)
# densities = kde.evaluate(vals)
# end_stats = time()

# start_scipy = time()
# kde = gaussian_kde(x, bw_method=1/np.sqrt(10), kernel=K)
# dens = kde(vals)
# end_scipy = time()

# print(f"Time for sklearn = {end_sklearn-start_sklearn}")
# print(f"Time for stats = {end_stats-start_stats}")
# print(f"Time for scipy = {end_scipy-start_scipy}")

# fig, (ax1, ax2, ax3) = plt.subplots(1,3)
# fig.suptitle("Density estimation")
# ax1.plot(vals,probs_sklearn)
# ax1.set_title("Sklearn")
# ax2.plot(vals, densities)
# ax2.set_title("Stats")
# ax3.plot(vals, dens)
# ax3.set_title("scipy")
# plt.show()


