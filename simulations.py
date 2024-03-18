import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from itertools import chain

from joblib import Parallel, delayed

from strategies import OptimalStrategy, reward, get_y1_and_zeta, DataDrivenImpulseControl
from diffusionProcess import DiffusionProcess, b, sigma


def simulate_optimal_strategy(T=10, dt=0.01):
    difPros = DiffusionProcess(b, sigma)
    optStrat = OptimalStrategy(difPros, reward)
    print(f"Optimal threshold: {optStrat.y_star}")
    t = [0]
    X_strat = [0]
    x_plot = []
    t_plot = []
    i = 0
    while t[i] < T:
        if optStrat.take_decision(x=X_strat[i]):
            t_plot.append(t)
            x_plot.append(X_strat)
            t = [t[i]]
            X_strat = [0]
            i = 0

        t.append(t[i] + dt)
        X_strat.append(difPros.step(X_strat[i], t[i], dt))
        i += 1

    total_reward = optStrat.reward
    print(f"Total reward was: {total_reward}")
    for t, x in zip(t_plot, x_plot):
        plt.plot(t,x, color="b")
        
    plt.show()
    return

def plot_uncontrolled_diffusion(T=100, dt=0.01, x0=0):
    difPros = DiffusionProcess(b, sigma)
    x, t = difPros.EulerMaruymaMethod(T, dt, x0)
    plt.plot(t, x)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.show()
    return

def plot_reward_xi_obj():
    difPros = DiffusionProcess(b, sigma)
    optStrat = OptimalStrategy(difPros, reward)
    y1, zeta = get_y1_and_zeta(g=reward)
    print(f"y1 = {y1} and zeta = {zeta}")

    y = np.linspace(y1, zeta*2, 20)
    gs = reward(y)

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


diffPros = DiffusionProcess(b=b, sigma=sigma)
opStrat = OptimalStrategy(diffusionProcess=diffPros, rewardFunc=reward)
thresholdStrat = OptimalStrategy(diffusionProcess=diffPros, rewardFunc=reward)

y1, zeta = get_y1_and_zeta(reward)

sims = 100
Ts = [200*i for i in range(1,51)]
thresholds = np.linspace(y1, zeta, 7)

data = []

def simulate_threshold_vs_optimal(tau, Ts, sims, diffusionProcess, OptimalStrat, ThresholdStrat):
    output = []
    for T in Ts:
        for s in range(sims):
            diffusionProcess.generate_noise(T, 0.01)
            ThresholdStrat.y_star = tau
            threshold_reward = ThresholdStrat.simulate(diffpros=diffusionProcess, T=T, dt=0.01)
            opt_reward = OptimalStrat.simulate(diffpros=diffusionProcess, T=T, dt=0.01)

            output.append({
                "threshold": tau,
                "T": T,
                "simNr": s,
                "threshold_reward": threshold_reward,
                "optimal_reward": opt_reward,
                "regret": opt_reward-threshold_reward
            })
    
    return output

result = Parallel(n_jobs=-1)(delayed(simulate_threshold_vs_optimal)(tau, Ts, sims, diffPros, opStrat, thresholdStrat) for tau in thresholds)

data_df = pd.DataFrame(list(chain.from_iterable(result)))
data_df.to_csv(path_or_buf="./SimulationData/ThresholdData.csv", encoding="utf-8", header=True, index=False)


# if __name__ == "__main__":
#     # plot_uncontrolled_diffusion()

#     #simulate_optimal_strategy()

#     # plot_reward_xi_obj()