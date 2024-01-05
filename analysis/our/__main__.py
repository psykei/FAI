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
        accs, precs, recs, f1s, dps, dis, eos, lambdas, file_names = [], [], [], [], [], [], [], [], []
        for LAMBDA in our_lambdas(IDX, CUSTOM_METRIC):
            files = get_files_from_parameters(
                path=OUR_PATH, custom_metric=CUSTOM_METRIC, l=LAMBDA, idx=IDX
            )
            for file in files:
                # if CUSTOM_METRIC == "disparate_impact" and IDX == 7:
                #     if os.path.isfile(file):
                #         os.remove(file)
                #     continue
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
