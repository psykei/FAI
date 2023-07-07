import pandas as pd
from analysis import get_files_from_parameters, get_final_metrics_from_file
from analysis.our import PATH as ANALYSIS_PATH
from images.our import PATH as IMAGES_PATH
from configuration import LAMBDAS
from images import plot_fairness_metric

IDX = 8
CUSTOM_METRICS = ["demographic_parity", "disparate_impact", "equalized_odds"]
FAIRNESS_METRIC_SHORT_NAMES = {
    "demographic_parity": "dp",
    "disparate_impact": "di",
    "equalized_odds": "eo",
}

for CUSTOM_METRIC in CUSTOM_METRICS:
    accs, dps, dis, eos, lambdas, file_names = [], [], [], [], [], []
    for LAMBDA in LAMBDAS:
        if LAMBDA == 1.0:
            LAMBDA = 1
        files = get_files_from_parameters(custom_metric=CUSTOM_METRIC, l=LAMBDA, idx=IDX)
        for file in files:
            loss, acc, dp, di, eo = get_final_metrics_from_file(file)
            accs.append(acc)
            dps.append(dp)
            dis.append(di)
            eos.append(eo)
            lambdas.append(LAMBDA)
            file_names.append(file.name)
    df = pd.DataFrame({"file name": file_names, "lambda": lambdas, "acc": accs, "dp": dps, "di": dis, "eo": eos})
    filename = f"{CUSTOM_METRIC}.csv"
    df.to_csv(ANALYSIS_PATH / filename, index=False)
    plot_fairness_metric(ANALYSIS_PATH / filename, IMAGES_PATH, FAIRNESS_METRIC_SHORT_NAMES[CUSTOM_METRIC])

