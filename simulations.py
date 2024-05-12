import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from itertools import chain
from scipy.integrate import IntegrationWarning

from joblib import Parallel, delayed

import inspect

import neptune

import warnings
import os

from itertools import product

import inspect

from diffusionProcess import DiffusionProcess, drift, sigma, generate_linear_drift
from strategies import OptimalStrategy, reward, get_y1_and_zeta, DataDrivenImpulseControl, generate_reward_func, get_bandwidth



def simulate_optimal_strategy(T=10, dt=0.01):
    difPros = DiffusionProcess(drift, sigma)
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
    nrDecisions = optStrat.nrDecisions
    print(f"Total reward was: {total_reward}")
    print(f"Number of decisions made: {nrDecisions}")
    plt.rcParams["figure.figsize"] = [12,6]
    for t, x in zip(t_plot, x_plot):
        plt.plot(t,x, color="k", linewidth=1.0)
        plt.plot(t, [optStrat.y_star for i in range(len(t))], "k--",linewidth=1.0)
    
    plt.xlabel("time (t)")
    plt.ylabel("X")
    plt.xticks([]) 
    plt.yticks(np.arange(round(min(map(min, x_plot))*2)/2, round(max(map(max, x_plot))*2)/2+0.5, 0.5))
    
    plt.savefig("test.png",bbox_inches='tight')
    plt.show()
    return

def plot_uncontrolled_diffusion(T=100, dt=0.01, x0=0):
    difPros = DiffusionProcess(drift, sigma)
    x, t = difPros.EulerMaruymaMethod(T, dt, x0)
    plt.plot(t, x)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.show()
    return

def plot_reward_xi_obj(C, A, power, zeroVal, save_obj=False):
    driftFunc = generate_linear_drift(C, A)
    rewardFunc = generate_reward_func(power, zeroVal)
    sigmaFunc = sigma
    difPros = DiffusionProcess(driftFunc, sigmaFunc)
    optStrat = OptimalStrategy(difPros, rewardFunc)
    y1, zeta = get_y1_and_zeta(g=rewardFunc)
    print(f"y1 = {y1} and zeta = {zeta}")

    y = np.linspace(y1-0.00001, zeta*2, 100)
    gs = rewardFunc(y)

    bs = np.fromiter(map(driftFunc,y), dtype=float)

    absDrift = np.abs(bs)
    driftLine = C*(1+np.abs(y))
    driftY = np.linspace(-A-1, A+1, 100)
    drifs = np.fromiter(map(driftFunc, driftY), dtype=float)
    sgnDrift = (drifs/sigma(driftY)**2)*np.sign(driftY)
    xis = difPros.xi(y)
    vals = gs/xis

    y_star = optStrat.y_star
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
    if save_obj:
        Cstr = str(C).replace(".", ",")
        Astr = str(A).replace(".", ",")
        Powstr = str(power).replace(".", ",")
        zeroValstr = str(zeroVal).replace(".", ",")
        plt.savefig(f"./Images/ObjectiveFunctions/C{Cstr}A{Astr}pow{Powstr}zeroVal{zeroValstr}.png")
        plt.close()
    plt.show()
    return


# diffPros = DiffusionProcess(b=drift, sigma=sigma)
# opStrat = OptimalStrategy(diffusionProcess=diffPros, rewardFunc=reward)
# thresholdStrat = OptimalStrategy(diffusionProcess=diffPros, rewardFunc=reward)
# dataStrat = DataDrivenImpulseControl(rewardFunc=reward, sigma=sigma)


#y1, zeta = get_y1_and_zeta(reward)

def simulate_MISE(STs,
                  sims,
                  C,
                  A,
                  power,
                  zeroVal,
                  a=0.000001,
                  M1=0.000001,
                  kernel_method="gaussian",
                  bandwidth_a_p_log=[1, -1/2, False],
                  neptune_tags=["MISE"]):
    
    driftFunc = generate_linear_drift(C, A)
    rewardFunc = generate_reward_func(power, zeroVal)
    sigmaFunc = sigma

    run = neptune.init_run(project='rasmusbrostroem/DiffusionControl', tags=neptune_tags)
    runId = run["sys/id"].fetch()
    diffusionProcess = DiffusionProcess(b=driftFunc, sigma=sigmaFunc)
    OptimalStrat = OptimalStrategy(diffusionProcess=diffusionProcess, rewardFunc=rewardFunc)
    DataStrat = DataDrivenImpulseControl(rewardFunc=rewardFunc, sigma=sigmaFunc)
    DataStrat.a = a
    DataStrat.M1 = M1
    DataStrat.kernel_method = kernel_method

    run["AlgoParams"] = {
        "kernelMethod": DataStrat.kernel_method,
        "bandwidthMethod": inspect.getsource(get_bandwidth),
        "a": DataStrat.a,
        "M1": DataStrat.M1,
        "bandwidth_a": bandwidth_a_p_log[0],
        "bandwidth_p": bandwidth_a_p_log[1],
        "bandiwdth_log": bandwidth_a_p_log[2]
    }

    run["ModelParams"] = {
        "driftFunc": inspect.getsource(diffusionProcess.b),
        "C": C,
        "A": A,
        "diffusionCoef": inspect.getsource(diffusionProcess.sigma),
        "rewardFunc": inspect.getsource(DataStrat.g),
        "power": power,
        "zeroVal": zeroVal,
        "y_star": OptimalStrat.y_star,
        "y1": DataStrat.y1,
        "zeta": DataStrat.zeta
    }

    for s in range(sims):
        run[f"Metrics/Sim{s}/ST"].extend(values=STs)
        run[f"Metrics/Sim{s}/simNr"].extend(values=[s for _ in STs])
        MISE_pdf_list = []
        # MISE_cdf_list = []
        for ST in STs:
            DataStrat.bandwidth = get_bandwidth(ST=ST, a=bandwidth_a_p_log[0], p=bandwidth_a_p_log[1], log=bandwidth_a_p_log[2])
            data, t = diffusionProcess.EulerMaruymaMethod(ST, 0.01, 0)
            DataStrat.fit(data)
            MISE_pdf = DataStrat.MISE_eval_pdf(diffusionProcess)
            MISE_pdf_list.append(MISE_pdf)
            # MISE_cdf = DataStrat.MISE_eval_cdf(diffusionProcess)
            # MISE_cdf_list.append(MISE_cdf)
    
        run[f"Metrics/Sim{s}/MISEPdf"].extend(values=MISE_pdf_list, steps=STs)
        # run[f"Metrics/Sim{s}/MISECdf"].extend(values=MISE_cdf_list, steps=STs)
    
    run.stop()


# sims = 50
# Ts = [100*i for i in range(1,21)]

# result = Parallel(n_jobs=7)(delayed(simulate_MISE)(T, sims, diffPros, dataStrat) for T in Ts)
# data_df = pd.DataFrame(list(chain.from_iterable(result)))
# data_df.to_csv(path_or_buf="./SimulationData/MISE3.csv", encoding="utf-8", header=True, index=False)

def simulate_KL(T, sims, diffusionProcess, dataStrategy):
    output = []
    dataStrategy.bandwidth = 1/np.sqrt(T)
    for s in range(sims):
        data, t = diffusionProcess.EulerMaruymaMethod(T, 0.01, 0)
        dataStrategy.fit(data)
        KL = dataStrategy.KL_eval(diffusionProcess)
        output.append({
            "T": T,
            "s": s,
            "KL": KL
        })
    
    return output

# sims = 50
# Ts = [100*i for i in range(1,21)]

# result = Parallel(n_jobs=7)(delayed(simulate_KL)(T, sims, diffPros, dataStrat) for T in Ts)
# data_df = pd.DataFrame(list(chain.from_iterable(result)))
# data_df.to_csv(path_or_buf="./SimulationData/KL2.csv", encoding="utf-8", header=True, index=False)

def simulate_threshold_estimation(T, sims, diffusionProcess: DiffusionProcess, ystar, dataStrategy: DataDrivenImpulseControl):
    output = []
    dataStrategy.bandwidth = 1/np.sqrt(T)
    for s in range(sims):
        data, t = diffusionProcess.EulerMaruymaMethod(T, 0.01, 0)
        dataStrategy.fit(data)
        threshold = dataStrategy.estimate_threshold()
        output.append({
            "T": T,
            "s": s,
            "SquareDiff": (threshold-ystar)**2
        })
    return output

# sims = 50
# Ts = [100*i for i in range(1,21)]

# result = Parallel(n_jobs=7)(delayed(simulate_threshold_estimation)(T, sims, diffPros, opStrat.y_star, dataStrat) for T in Ts)
# data_df = pd.DataFrame(list(chain.from_iterable(result)))
# data_df.to_csv(path_or_buf="./SimulationData/ThresholdDiff.csv", encoding="utf-8", header=True, index=False)

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

# sims = 5
# Ts = [100*i for i in range(1,51)]
# thresholds = np.linspace(y1, zeta, 7)

# result = Parallel(n_jobs=-1)(delayed(simulate_threshold_vs_optimal)(tau, Ts, sims, diffPros, opStrat, thresholdStrat) for tau in thresholds)
# data_df = pd.DataFrame(list(chain.from_iterable(result)))
# data_df.to_csv(path_or_buf="./SimulationData/ThresholdData.csv", encoding="utf-8", header=True, index=False)



def simulate_dataDriven_vs_optimal(Ts,
                                   sims,
                                   C,
                                   A,
                                   power,
                                   zeroVal,
                                   a=0.000001,
                                   M1=0.000001,
                                   ST_form_and_text=(lambda t: t**(2/3), "T^(2/3)"),
                                   kernel_method="gaussian",
                                   bandwidth_func = lambda t: 1/np.sqrt(t),
                                   neptune_tags=["StrategyVsOptimal"]):
    
    driftFunc = generate_linear_drift(C, A)
    rewardFunc = generate_reward_func(power, zeroVal)
    sigmaFunc = sigma
    run = neptune.init_run(project='rasmusbrostroem/DiffusionControl', tags=neptune_tags)
    runId = run["sys/id"].fetch()
    diffusionProcess = DiffusionProcess(b=driftFunc, sigma=sigmaFunc)
    OptimalStrat = OptimalStrategy(diffusionProcess=diffusionProcess, rewardFunc=rewardFunc)
    DataStrat = DataDrivenImpulseControl(rewardFunc=rewardFunc, sigma=sigmaFunc)
    DataStrat.a = a
    DataStrat.M1 = M1
    DataStrat.kernel_method = kernel_method
    DataStrat.bandwidthFunc = bandwidth_func
    DataStrat.ST_form = ST_form_and_text[0]

    run["AlgoParams"] = {
        "kernelMethod": DataStrat.kernel_method,
        "bandwidthMethod": inspect.getsource(bandwidth_func),
        "a": DataStrat.a,
        "M1": DataStrat.M1,
        "ST_form": ST_form_and_text[1]
    }

    run["ModelParams"] = {
        "driftFunc": inspect.getsource(diffusionProcess.b),
        "C": C,
        "A": A,
        "diffusionCoef": inspect.getsource(diffusionProcess.sigma),
        "rewardFunc": inspect.getsource(DataStrat.g),
        "power": power,
        "zeroVal": zeroVal,
        "y_star": OptimalStrat.y_star,
        "y1": DataStrat.y1,
        "zeta": DataStrat.zeta
    }

    for s in range(sims):
        run[f"Metrics/Sim{s}/T"].extend(values=Ts)
        run[f"Metrics/Sim{s}/simNr"].extend(values=[s for _ in Ts])
        S_Ts = []
        dataRewards = []
        dataNrDecisionsList = []
        optRewards = []
        optNrdecisionsList = []
        regrets = []
        for T in Ts:
            diffusionProcess.generate_noise(T, 0.01)
            dataReward, S_T, thresholds_and_Sts, DataStratNrDecisions = DataStrat.simulate(diffpros=diffusionProcess, T=T, dt=0.01)
            if len(thresholds_and_Sts) >= 1:
                thresholds, Sts = zip(*thresholds_and_Sts)
                if len(thresholds) == 1:
                    run[f"Metrics/Sim{s}/Thresholds/{T}"].append(value=thresholds[0], step=Sts[0])
                else:
                    run[f"Metrics/Sim{s}/Thresholds/{T}"].extend(values=thresholds, steps=Sts)

            opt_reward, optNrDecisions = OptimalStrat.simulate(diffpros=diffusionProcess, T=T, dt=0.01)

            S_Ts.append(S_T)
            dataRewards.append(dataReward)
            dataNrDecisionsList.append(DataStratNrDecisions)
            optRewards.append(opt_reward)
            optNrdecisionsList.append(optNrDecisions)
            regrets.append(opt_reward-dataReward)
        
        run[f"Metrics/Sim{s}/S_T"].extend(values=S_Ts, steps=Ts)
        run[f"Metrics/Sim{s}/dataDriveReward"].extend(values=dataRewards, steps=Ts)
        run[f"Metrics/Sim{s}/OptimalStratReward"].extend(values=optRewards, steps=Ts)
        run[f"Metrics/Sim{s}/regret"].extend(values=regrets, steps=Ts)
        run[f"Metrics/Sim{s}/optNrDecisions"].extend(values=optNrdecisionsList, steps=Ts)
        run[f"Metrics/Sim{s}/dataStratNrDecisions"].extend(values=dataNrDecisionsList, steps=Ts)
    
    run.stop()

def simulate_dataDriven_vs_optimal_using_exploitation_data(Ts,
                                                           sims,
                                                           C,
                                                           A,
                                                           power,
                                                           zeroVal,
                                                           a=0.000001,
                                                           M1=0.000001,
                                                           ST_form_and_text=(lambda t: t**(2/3), "T^(2/3)"),
                                                           kernel_method="gaussian",
                                                           bandwidth_func = lambda t: 1/np.sqrt(t),
                                                           p_of_exploitation_data = 0.1,
                                                           neptune_tags=["StrategyVsOptimal"]):
    
    driftFunc = generate_linear_drift(C, A)
    rewardFunc = generate_reward_func(power, zeroVal)
    sigmaFunc = sigma
    run = neptune.init_run(project='rasmusbrostroem/DiffusionControl', tags=neptune_tags)
    runId = run["sys/id"].fetch()
    diffusionProcess = DiffusionProcess(b=driftFunc, sigma=sigmaFunc)
    OptimalStrat = OptimalStrategy(diffusionProcess=diffusionProcess, rewardFunc=rewardFunc)
    DataStrat = DataDrivenImpulseControl(rewardFunc=rewardFunc, sigma=sigmaFunc)
    DataStrat.a = a
    DataStrat.M1 = M1
    DataStrat.kernel_method = kernel_method
    DataStrat.bandwidthFunc = bandwidth_func
    DataStrat.ST_form = ST_form_and_text[0]

    run["AlgoParams"] = {
        "kernelMethod": DataStrat.kernel_method,
        "bandwidthMethod": inspect.getsource(bandwidth_func),
        "a": DataStrat.a,
        "M1": DataStrat.M1,
        "ST_form": ST_form_and_text[1],
        "Exploitation_data_percentage": p_of_exploitation_data
    }

    run["ModelParams"] = {
        "driftFunc": inspect.getsource(diffusionProcess.b),
        "C": C,
        "A": A,
        "diffusionCoef": inspect.getsource(diffusionProcess.sigma),
        "rewardFunc": inspect.getsource(DataStrat.g),
        "power": power,
        "zeroVal": zeroVal,
        "y_star": OptimalStrat.y_star,
        "y1": DataStrat.y1,
        "zeta": DataStrat.zeta
    }

    for s in range(sims):
        run[f"Metrics/Sim{s}/T"].extend(values=Ts)
        run[f"Metrics/Sim{s}/simNr"].extend(values=[s for _ in Ts])
        S_Ts = []
        dataRewards = []
        dataNrDecisionsList = []
        optRewards = []
        optNrdecisionsList = []
        regrets = []
        MISE_pdf_list = []
        MISE_cdf_list = []
        for T in Ts:
            diffusionProcess.generate_noise(T, 0.01)
            strategyData = DataStrat.simulate_using_exploitation_data(diffpros=diffusionProcess, T=T, dt=0.01, p_of_exploitation=p_of_exploitation_data)
            dataReward = strategyData[0]
            S_T = strategyData[1]
            thresholds_and_Sts = strategyData[2]
            DataStratNrDecisions = strategyData[3]
            MISE_pdf = strategyData[4]
            MISE_cdf = strategyData[5]

            if len(thresholds_and_Sts) >= 1:
                thresholds, Sts = zip(*thresholds_and_Sts)
                if len(thresholds) == 1:
                    run[f"Metrics/Sim{s}/Thresholds/{T}"].append(value=thresholds[0], step=Sts[0])
                else:
                    run[f"Metrics/Sim{s}/Thresholds/{T}"].extend(values=thresholds, steps=Sts)

            opt_reward, optNrDecisions = OptimalStrat.simulate(diffpros=diffusionProcess, T=T, dt=0.01)

            S_Ts.append(S_T)
            dataRewards.append(dataReward)
            dataNrDecisionsList.append(DataStratNrDecisions)
            optRewards.append(opt_reward)
            optNrdecisionsList.append(optNrDecisions)
            regrets.append(opt_reward-dataReward)
            
            MISE_pdf_list.append(MISE_pdf) if MISE_pdf else MISE_pdf_list.append(-1)
            MISE_cdf_list.append(MISE_cdf) if MISE_cdf else MISE_cdf_list.append(-1)
        
        run[f"Metrics/Sim{s}/S_T"].extend(values=S_Ts, steps=Ts)
        run[f"Metrics/Sim{s}/dataDriveReward"].extend(values=dataRewards, steps=Ts)
        run[f"Metrics/Sim{s}/OptimalStratReward"].extend(values=optRewards, steps=Ts)
        run[f"Metrics/Sim{s}/regret"].extend(values=regrets, steps=Ts)
        run[f"Metrics/Sim{s}/optNrDecisions"].extend(values=optNrdecisionsList, steps=Ts)
        run[f"Metrics/Sim{s}/dataStratNrDecisions"].extend(values=dataNrDecisionsList, steps=Ts)
        run[f"Metrics/Sim{s}/MISE_pdf"].extend(values=MISE_pdf_list, steps=Ts)
        run[f"Metrics/Sim{s}/MISE_cdf"].extend(values=MISE_cdf_list, steps=Ts)

    run.stop()


if __name__ == "__main__":
    ### Simulating the robustness for changing models and reward function
    # Ts = [100*i for i in range(1,51)]
    # sims = 100
    # powers = [3/4, 1, 2, 5]
    # zeroVals = [7/10, 45/50, 99/100]
    # Cs = [1/10, 1/2, 4]
    # As = [0]
    # argList = list(product(Cs, As, powers, zeroVals))

    # #simulate_dataDriven_vs_optimal(Ts=Ts, sims=sims, C=1/2, A=0, power=1, zeroVal=7/10)
    # Parallel(n_jobs=6)(delayed(simulate_dataDriven_vs_optimal)(Ts=Ts,
    #                                                            sims=sims,
    #                                                            C=C,
    #                                                            A=A,
    #                                                            power=p,
    #                                                            zeroVal=z,
    #                                                            neptune_tags=["Fixed DataStratVsOptimal", "ModelRobustness"]) for C, A, p, z in argList)

    ### Simulating MISE for different kernels and different drift functions
    # STs = [10*i for i in range(1,31)]
    # sims = 100
    # kernels = ["gaussian", "epanechnikov", "linear", "tophat"]
    # Cs = [1/10, 1/2, 2, 4]
    # powers = [1]
    # zeroVals = [7/10]
    # As = [0]
    # argList = list(product(Cs, As, powers, zeroVals, kernels))
    # Parallel(n_jobs=6)(delayed(simulate_MISE)(STs=STs,
    #                                           sims=sims,
    #                                           C=C,
    #                                           A=A,
    #                                           power=p,
    #                                           zeroVal=z,
    #                                           kernel_method=kernel,
    #                                           neptune_tags=["Fixed MISE", "Kernel Methods"]) for C, A, p, z, kernel in argList)

    ### Simulating the robustness for different kernels
    # Ts = [100*i for i in range(1,51)]
    # sims = 100
    # kernels = ["gaussian", "epanechnikov", "linear", "tophat"]
    # powers = [1, 5]
    # zeroVals = [0.9]
    # Cs = [1/2, 4]
    # As = [0]
    # argList = list(product(Cs, As, powers, zeroVals, kernels))

    # Parallel(n_jobs=6)(delayed(simulate_dataDriven_vs_optimal)(Ts=Ts,
    #                                                            sims=sims,
    #                                                            C=C,
    #                                                            A=A,
    #                                                            power=p,
    #                                                            zeroVal=z,
    #                                                            kernel_method=kernel,
    #                                                            neptune_tags=["Fixed StrategyVsOptimal", "Kernel Methods"]) for C, A, p, z, kernel in argList)

    ### Simulating MISE for different bandwidths and drift functions
    # STs = [10*i for i in range(1,31)]
    # sims = 100
    # kernels = ["gaussian"]
    # Cs = [1/10, 1/2, 2, 4]
    # bandwidths = [[1, -1/2, False],
    #               [5, -1/2, False],
    #               [10, -1/2, False],
    #               [1, -1/4, False],
    #               [1, -1/8, False],
    #               ["scott", -1/2, False],
    #               ["silverman", -1/2, False],
    #               ["", "", True]]

    # argList = list(product(Cs, bandwidths))
    # Parallel(n_jobs=6)(delayed(simulate_MISE)(STs=STs,
    #                                           sims=sims,
    #                                           C=C,
    #                                           A=0,
    #                                           power=1,
    #                                           zeroVal=7/10,
    #                                           kernel_method="gaussian",
    #                                           bandwidth_a_p_log=bandwidth_a_p_log,
    #                                           neptune_tags=["Fixed MISE", "Kernel Bandwidths"]) for C, bandwidth_a_p_log in argList)
    
    ### Simulating different exploration times
    # Ts = [100*i for i in range(1,51)]
    # sims = 100
    # Cs = [1/2, 4]
    # powers = [1, 5]
    # ST_forms = [(lambda t: t**(1/4), "T^(1/4)"),
    #             (lambda t: t**(1/3), "T^(1/3)"),
    #             (lambda t: t**(1/2), "T^(1/2)"),
    #             (lambda t: t**(2/3), "T^(2/3)"),
    #             (lambda t: t**(3/4), "T^(3/4)"),
    #             (lambda t: 2*(np.sqrt(2*np.sqrt(t)+1) + np.sqrt(t) + 1), "2*(sqrt(2*sqrt(T)+1) + sqrt(T) + 1)")]

    # argList = list(product(Cs, powers, ST_forms))
    # Parallel(n_jobs=6)(delayed(simulate_dataDriven_vs_optimal)(Ts=Ts,
    #                                                            sims=sims,
    #                                                            C=C,
    #                                                            A=0,
    #                                                            power=p,
    #                                                            zeroVal=0.9,
    #                                                            ST_form_and_text=st_form,
    #                                                            neptune_tags=["Fixed Exploration Forms", "DataDrivenVsOptimal"]) for C, p, st_form in argList)

    ### Simulate using exploration data
    Ts = [100*i for i in range(1,51)]
    sims = 100
    powers = [1, 5]
    zeroVals = [0.9]
    Cs = [0.5, 4]
    As = [0]
    exploration_percentages = [0, 0.1, 0.25, 0.50, 0.75, "all"]
    argList = list(product(Cs, As, powers, zeroVals, exploration_percentages))

    Parallel(n_jobs=6)(delayed(simulate_dataDriven_vs_optimal_using_exploitation_data)(Ts=Ts,
                                                                                       sims=sims,
                                                                                       C=C,
                                                                                       A=A,
                                                                                       power=p,
                                                                                       zeroVal=z,
                                                                                       p_of_exploitation_data=exploi_percent,
                                                                                       neptune_tags=["DataStratUsingExploitationData"]) for 
                                                                                                        C, A, p, z, exploi_percent in argList)
    