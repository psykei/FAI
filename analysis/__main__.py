import os
from images import plot_fairness_comparison, PATH as IMAGES_PATH
from analysis.our import PATH as OUR_ANALYSIS_PATH
from analysis.cho import PATH as CHO_ANALYSIS_PATH


METRICS = ["equalized_odds"]
# METRICS = ["demographic_parity", "disparate_impact", "equalized_odds"]
DATA_PATHS = [OUR_ANALYSIS_PATH, CHO_ANALYSIS_PATH]

for METRIC in METRICS:
    data_files = [path / (METRIC + ".csv") for path in DATA_PATHS]
    plot_fairness_comparison(data_files, IMAGES_PATH, METRIC)
