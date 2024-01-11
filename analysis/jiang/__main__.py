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
        accs, precs, recs, f1s, dps, dis, eos, lambdas, file_names = [], [], [], [], [], [], [], [], []
        for LAMBDA in jiang_lambdas(IDX):
            files = get_files_from_parameters(
                custom_metric=CUSTOM_METRIC, l=LAMBDA, idx=IDX, path=PATH
            )
            for file in files:
                if False:
                    if os.path.isfile(file):
                        os.remove(file)
                    continue
                acc, prec, rec, f1, dp, di, eo = get_final_metrics_from_file(file)
                accs.append(acc)
                precs.append(prec)
                recs.append(rec)
                f1s.append(f1)
                dps.append(dp)
                dis.append(di)
                eos.append(eo)
                lambdas.append(LAMBDA)
                file_names.append(file.name)
        df = pd.DataFrame(
            {
                "file name": file_names,
                "lambda": lambdas,
                "acc": accs,
                "prec": precs,
                "rec": recs,
                "f1": f1s,
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
            'acc'
        )
        plot_fairness_metric(
            ANALYSIS_PATH / filename,
            IMAGES_PATH,
            FAIRNESS_METRIC_SHORT_NAMES[CUSTOM_METRIC],
            IDX,
            'f1'
        )
