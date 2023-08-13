import os

import pandas as pd
from analysis import get_files_from_parameters, get_final_metrics_from_file
from analysis.our import PATH as ANALYSIS_PATH
from images.our import PATH as IMAGES_PATH
from configuration import IDX_TO_NAME, our_lambdas, IDXS, LOG, CUSTOM_METRICS
from images import plot_fairness_metric
from fairness.our import PATH as OUR_PATH

OUR_PATH /= LOG

FAIRNESS_METRIC_SHORT_NAMES = {
    "demographic_parity": "dp",
    "disparate_impact": "di",
    "equalized_odds": "eo",
}

for CUSTOM_METRIC in CUSTOM_METRICS:
    for IDX in IDXS:
        accs, dps, dis, eos, lambdas, file_names = [], [], [], [], [], []
        for LAMBDA in our_lambdas(IDX, CUSTOM_METRIC):
            files = get_files_from_parameters(
                path=OUR_PATH, custom_metric=CUSTOM_METRIC, l=LAMBDA, idx=IDX
            )
            for file in files:
                loss, acc, dp, di, eo = get_final_metrics_from_file(file)
                accs.append(acc)
                dps.append(dp)
                dis.append(di)
                eos.append(eo)
                lambdas.append(LAMBDA)
                file_names.append(file.name)
                # os.remove(file)
                # if CUSTOM_METRIC == "equalized_odds" and IDX == 0:
                #     os.remove(file)
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
