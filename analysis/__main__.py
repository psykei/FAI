from configuration import IDX_TO_NAME
from images import plot_fairness_comparison, PATH as IMAGES_PATH
from analysis.our import PATH as OUR_ANALYSIS_PATH
from analysis.cho import PATH as CHO_ANALYSIS_PATH
from analysis.jiang import PATH as JIANG_ANALYSIS_PATH


METRICS = ["demographic_parity"]
IDXS = [0, 7, 8]
# METRICS = ["demographic_parity", "disparate_impact", "equalized_odds"]
DATA_PATHS = [OUR_ANALYSIS_PATH, CHO_ANALYSIS_PATH, JIANG_ANALYSIS_PATH]

for METRIC in METRICS:
    for IDX in IDXS:
        data_files = [path / (METRIC + "_" + IDX_TO_NAME[IDX] + ".csv") for path in DATA_PATHS]
        plot_fairness_comparison(data_files, IMAGES_PATH, METRIC, IDX)
