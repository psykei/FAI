from analysis import perform_analysis
from analysis.our import PATH as ANALYSIS_PATH
from images.our import PATH as IMAGES_PATH
from configuration import our_lambdas, IDXS, LOG, CUSTOM_METRICS
from fairness.fauci import PATH

PATH /= LOG

FAIRNESS_METRIC_SHORT_NAMES = {
    "demographic_parity": "dp",
    "disparate_impact": "di",
    "equalized_odds": "eo",
}

for CUSTOM_METRIC in CUSTOM_METRICS:
    for IDX in IDXS:
        perform_analysis(IDX, CUSTOM_METRIC, our_lambdas(IDX, CUSTOM_METRIC), PATH, ANALYSIS_PATH, IMAGES_PATH)
