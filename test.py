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

from sklearn.neighbors import KernelDensity
from statsmodels.nonparametric.kde import KDEUnivariate
from scipy.stats import gaussian_kde

difPros = DiffusionProcess(b, sigma)
optStrat = OptimalStrategy(diffusionProcess=difPros, rewardFunc=reward)

x, t = difPros.EulerMaruymaMethod(100, 0.01, 0)

T=100
dataStrat = DataDrivenImpulseControl(rewardFunc=reward, bandwidth=1/np.sqrt(T))
y1, zeta = get_y1_and_zeta(reward)


dataStrat.fit(x)

print(dataStrat.estimate_threshold())
cProfile.run("dataStrat.estimate_threshold()", "threshold_stats")
p = pstats.Stats("threshold_stats")
p.sort_stats("cumulative").print_stats()


# vals = np.linspace(y1, zeta, 20000)

# sklearn = KernelDensity(kernel="gaussian", bandwidth=1/np.sqrt(T))
# stats = KDEUnivariate(list(x))

# start = time()
# stats.fit(kernel="gau", bw=1/np.sqrt(T))
# end = time()
# print(f"Statsmodel fit time: {end-start}")

# xs = np.array(x)[:, None]
# start = time()
# sklearn.fit(xs)
# end = time()
# print(f"sklearn fit time: {end-start}")

# statsEval = np.zeros(len(vals))
# start = time()
# for i, v in enumerate(vals):
#     statsEval[i] = stats.evaluate(v)[0]
# end = time()
# print(f"Statsmodel evaluation time: {end-start}")

# skpdfs = np.zeros(len(vals))
# start = time()
# for i, v in enumerate(vals):
#     skEval = sklearn.score_samples([[v]])
#     skpdfs[i] = np.exp(skEval)[0]
# end = time()
# print(f"sklearn evaluation time: {end-start}")

# fig, (ax1, ax2) = plt.subplots(1,2, sharey=True)
# ax1.plot(vals, statsEval)
# ax1.set_title("Statsmodel")
# ax2.plot(vals, skpdfs)
# ax2.set_title("Sklearn")
# fig.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.show()


#optStrat.simulate(diffpros=difPros, T=T, dt=0.01)



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


