import os

import pandas as pd
from configuration import *
from analysis import get_files_from_parameters, get_final_metrics_from_file
from images import plot_fairness_metric
from fairness.jiang import PATH as JIANG_PATH
from images.jiang import PATH as IMAGES_PATH
from analysis.jiang import PATH as ANALYSIS_PATH

PATH = JIANG_PATH / LOG
FAIRNESS_METRIC_SHORT_NAMES = {
    "demographic_parity": "dp",
    "disparate_impact": "di",
    "equalized_odds": "eo",
}

for CUSTOM_METRIC in JIANG_METRICS:
    for IDX in IDXS:
        accs, dps, dis, eos, lambdas, file_names = [], [], [], [], [], []
        for LAMBDA in jiang_lambdas(IDX):
            files = get_files_from_parameters(
                custom_metric=CUSTOM_METRIC, l=LAMBDA, idx=IDX, path=PATH
            )
            for file in files:
                loss, acc, dp, di, eo = get_final_metrics_from_file(file)
                accs.append(acc)
                dps.append(dp)
                dis.append(di)
                eos.append(eo)
                lambdas.append(LAMBDA)
                file_names.append(file.name)
                # if IDX == 8:
                # os.remove(file)
        df = pd.DataFrame(
            {
                "file name": file_names,
                "lambda": lambdas,
                "acc": accs,
                "dp": dps,
                "di": dis,
                "eo": eos,
            }
        )
        filename = f"{CUSTOM_METRIC}_{IDX_TO_NAME[IDX]}.csv"
        df.to_csv(ANALYSIS_PATH / filename, index=False)
        plot_fairness_metric(
            ANALYSIS_PATH / filename,
            IMAGES_PATH,
            FAIRNESS_METRIC_SHORT_NAMES[CUSTOM_METRIC],
            IDX,
        )
