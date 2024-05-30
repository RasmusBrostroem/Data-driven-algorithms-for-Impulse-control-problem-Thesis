import neptune
import pandas as pd


def download_neptune_data(ids: list[str], file_name: str):
    dataDfs = []
    for id in ids:
        neptuneRun = neptune.init_run(project="rasmusbrostroem/DiffusionControl", with_id=id, mode="read-only")

        runDfs = []
        for key in neptuneRun.get_structure()["Metrics"].keys():
            params = list(neptuneRun.get_structure()["Metrics"][key].keys())
            tmpDf = pd.DataFrame()
            for param in params:
                if param == "Thresholds":
                    continue
                
                paramString = f"Metrics/{key}/{param}"
                paramDf = neptuneRun[paramString].fetch_values()
                tmpDf[param] = paramDf["value"]
            
            runDfs.append(tmpDf)

        runDf = pd.concat(runDfs, ignore_index=True)

        for key in neptuneRun.get_structure()["ModelParams"].keys():
            modelParam = neptuneRun["ModelParams"][key].fetch()
            runDf[key] = modelParam

        for key in neptuneRun.get_structure()["AlgoParams"].keys():
            algoParam = neptuneRun["AlgoParams"][key].fetch()
            runDf[key] = algoParam
        

        dataDfs.append(runDf)
        neptuneRun.stop()
    
    dataDf = pd.concat(dataDfs, ignore_index=True)
    dataDf.to_csv(f"./SimulationData/{file_name}.csv")

def download_threshold_data(ids: list[str], file_name: str):
    dataDfs = []
    for id in ids:
        neptuneRun = neptune.init_run(project="rasmusbrostroem/DiffusionControl", with_id=id, mode="read-only")

        runDfs = []
        for key in neptuneRun.get_structure()["Metrics"].keys():
            Ts = list(neptuneRun.get_structure()["Metrics"][key]["Thresholds"].keys())
            for T in Ts:              
                tmpDf = pd.DataFrame()
                paramString = f"Metrics/{key}/Thresholds/{T}"
                paramDf = neptuneRun[paramString].fetch_values()
                if paramDf.empty:
                    continue
                tmpDf["ST"] = paramDf["step"]
                tmpDf["yhat"] = paramDf["value"]
                tmpDf["T"] = T
                tmpDf["Sim"] = key

                runDfs.append(tmpDf)
            
        runDf = pd.concat(runDfs, ignore_index=True)

        for key in neptuneRun.get_structure()["ModelParams"].keys():
            modelParam = neptuneRun["ModelParams"][key].fetch()
            runDf[key] = modelParam

        for key in neptuneRun.get_structure()["AlgoParams"].keys():
            algoParam = neptuneRun["AlgoParams"][key].fetch()
            runDf[key] = algoParam
        

        dataDfs.append(runDf)
        neptuneRun.stop()
    
    dataDf = pd.concat(dataDfs, ignore_index=True)
    dataDf.to_csv(f"./SimulationData/{file_name}.csv")



if __name__ == "__main__":
    ### Download DriftsAndReward data
    # id_list = [f"DIF-{i}" for i in range(335, 371)]
    # download_neptune_data(ids=id_list, file_name="DriftsAndRewards")
    # download_threshold_data(id_list, "DriftsAndRewardsThresholds")

    # ### Download Kernel methods MISE data
    # id_list = [f"DIF-{i}" for i in range(371, 387)]
    # download_neptune_data(ids=id_list, file_name="MiseKernelFunctions")

    # ### Download Optimal vs data-start with different Kernel strategies
    # id_list = [f"DIF-{i}" for i in range(387, 403)]
    # download_neptune_data(ids=id_list, file_name="KernelStrategiesData")

    # ### Download Kernel bandwidths MISE data
    # id_list = [f"DIF-{i}" for i in range(409, 441)]
    # download_neptune_data(ids=id_list, file_name="MiseKernelBandwidth")

    ### Download Optimal vs data-strat with different exploration forms
    id_list = [f"DIF-{i}" for i in range(441, 465)]
    # download_neptune_data(ids=id_list, file_name="ExplorationFormsStrategiesData")
    download_threshold_data(id_list, "ExplorationFormsStrategiesThresholdData")

    # ### Download Diffusion coefficient data
    # id_list = [f"DIF-{i}" for i in range(509, 521)]
    # download_neptune_data(ids=id_list, file_name="DiffusionCoefficientData")
    # download_threshold_data(id_list, "DiffusionCoefficientThresholdData")

    ### Download Misspecification data
    #id_list = [f"DIF-{i}" for i in range(545, 557)]
    #download_neptune_data(ids=id_list, file_name="MisspecificationData")
    #download_threshold_data(id_list, "MisspecificationThresholdData")

    ### Download Diffusion a and M1 data
    # id_list = [f"DIF-{i}" for i in range(585, 617)]
    # download_neptune_data(ids=id_list, file_name="aAndM1Data")
    # download_threshold_data(id_list, "aAndM1ThresholdsData")








