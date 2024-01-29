from configuration import *
from analysis import perform_analysis
from fairness.cho import PATH as CHO_PATH
from images.cho import PATH as IMAGES_PATH
from analysis.cho import PATH as ANALYSIS_PATH

PATH = CHO_PATH / LOG
FAIRNESS_METRIC_SHORT_NAMES = {
    "demographic_parity": "dp",
    "disparate_impact": "di",
    "equalized_odds": "eo",
}

for CUSTOM_METRIC in CHO_METRICS:
    for IDX in IDXS:
        perform_analysis(IDX, CUSTOM_METRIC, cho_lambdas(IDX, CUSTOM_METRIC), PATH, ANALYSIS_PATH, IMAGES_PATH)
