from configuration import IDX_TO_NAME
from images import plot_fairness_comparison, PATH as IMAGES_PATH
from analysis.our import PATH as OUR_ANALYSIS_PATH
from analysis.cho import PATH as CHO_ANALYSIS_PATH
from analysis.jiang import PATH as JIANG_ANALYSIS_PATH


METRICS = ["demographic_parity", "disparate_impact", "equalized_odds"]
IDXS = [0, 7, 8]
DATA_PATHS = [CHO_ANALYSIS_PATH, OUR_ANALYSIS_PATH, JIANG_ANALYSIS_PATH]
METHOD_NAMES = ["Cho", "Our", "Jiang"]
COLORS = ["#09B5E6", "#E63F14", "#20E635"]
SHAPES = ["d", "*", "."]

for METRIC in METRICS:
    for IDX in IDXS:
        data_files = [
            path / (METRIC + "_" + IDX_TO_NAME[IDX] + ".csv") for path in DATA_PATHS
        ]
        plot_fairness_comparison(
            data_files, METHOD_NAMES, COLORS, SHAPES, IMAGES_PATH, METRIC, IDX
        )
