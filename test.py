from scipy.integrate import odeint, quad
from diffusionProcess import b, sigma, DiffusionProcess
from time import time
import numpy as np
from collections.abc import Iterable
from functools import partial
import cProfile
import matplotlib.pyplot as plt

from sklearn.neighbors import KernelDensity
from statsmodels.nonparametric.kde import KDEUnivariate
from scipy.stats import gaussian_kde

disPros = DiffusionProcess(b, sigma)

def K(x):
    return np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)

x, t = disPros.EulerMaruymaMethod(100, 0.01, 0)
vals = np.linspace(-2,2,20)

start_sklearn = time()
kde = KernelDensity(kernel="gaussian", bandwidth=1/np.sqrt(10))
kde.fit(x[:, None])
logprob = kde.score_samples(vals[:, None])
probs_sklearn = np.exp(logprob)
end_sklearn = time()

start_stats = time()
kde = KDEUnivariate(x)
kde.fit(kernel="gau", bw=1/np.sqrt(10), fft=False)
densities = kde.evaluate(vals)
end_stats = time()

start_scipy = time()
kde = gaussian_kde(x, bw_method=1/np.sqrt(10), kernel=K)
dens = kde(vals)
end_scipy = time()

print(f"Time for sklearn = {end_sklearn-start_sklearn}")
print(f"Time for stats = {end_stats-start_stats}")
print(f"Time for scipy = {end_scipy-start_scipy}")

fig, (ax1, ax2, ax3) = plt.subplots(1,3)
fig.suptitle("Density estimation")
ax1.plot(vals,probs_sklearn)
ax1.set_title("Sklearn")
ax2.plot(vals, densities)
ax2.set_title("Stats")
ax3.plot(vals, dens)
ax3.set_title("scipy")
plt.show()


