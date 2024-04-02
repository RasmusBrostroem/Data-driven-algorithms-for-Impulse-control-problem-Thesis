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

download_neptune_data(ids=["DIF-73", "DIF-74"], file_name="test")









