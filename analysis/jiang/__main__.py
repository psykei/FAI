from configuration import *
from analysis import perform_analysis
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
        perform_analysis(IDX, CUSTOM_METRIC, jiang_lambdas(IDX), PATH, ANALYSIS_PATH, IMAGES_PATH)
