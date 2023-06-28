import pandas as pd
from analysis import get_files_from_parameters, get_final_metrics_from_file, PATH as ANALYSIS_PATH


CUSTOM_METRICS = ["demographic_parity", "disparate_impact", "equalized_odds"]

for CUSTOM_METRIC in CUSTOM_METRICS:
    accs, dps, dis, eos, lambdas = [], [], [], [], []
    for LAMBDA in [(10 - i) / 10 for i in range(0, 10)]:
        if LAMBDA == 1.0:
            LAMBDA = 1
        files = get_files_from_parameters(custom_metric=CUSTOM_METRIC, l=LAMBDA)
        for file in files:
            loss, acc, dp, di, eo = get_final_metrics_from_file(file)
            accs.append(acc)
            dps.append(dp)
            dis.append(di)
            eos.append(eo)
            lambdas.append(LAMBDA)
    df = pd.DataFrame({"lambda": lambdas, "acc": accs, "dp": dps, "di": dis, "eo": eos})
    df.to_csv(ANALYSIS_PATH / f"{CUSTOM_METRIC}.csv", index=False)

