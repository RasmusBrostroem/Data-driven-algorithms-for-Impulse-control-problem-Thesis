import neptune
import pandas as pd

neptuneRun = neptune.init_run(project="rasmusbrostroem/DiffusionControl", with_id="DIF-38", mode="read-only")

dataDfs = []
for key in neptuneRun.get_structure()["Metrics"].keys():
    params = list(neptuneRun.get_structure()["Metrics"][key].keys())
    tmpDf = pd.DataFrame()
    for param in params:
        if param == "Thresholds":
            continue
        
        paramString = f"Metrics/{key}/{param}"
        paramDf = neptuneRun[paramString].fetch_values()
        tmpDf[param] = paramDf["value"]
    
    dataDfs.append(tmpDf)

dataDf = pd.concat(dataDfs, ignore_index=True)

print(dataDf)






