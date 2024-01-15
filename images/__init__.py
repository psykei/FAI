import os
from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt
from configuration import IDX_TO_NAME

PATH = Path(__file__).parents[0]
FAIRNESS_METRIC_LONG_NAMES = {
    "dp": "Demographic parity",
    "di": "Disparate impact",
    "eo": "Equalized odds",
}
FAIRNESS_METRIC_SHORT_NAMES = {
    "demographic_parity": "dp",
    "disparate_impact": "di",
    "equalized_odds": "eo",
}


def plot_fairness_metric(
    data_file: Path, image_path: Path, fairness_metric: str, idx: int, ml_metric: str = 'acc'
) -> None:
    """
    Plot fairness metric.
    X-axis is the corresponding fairness metric.
    Y-axis is the accuracy.
    :param data_file:
    :param image_path:
    :param fairness_metric:
    :param idx:
    :param ml_metric:
    """
    data = pd.read_csv(data_file)
    # Exclude experiments with accuracy < 0.75
    # And f1 < 0.35
    threshold = 0.8 if ml_metric == 'acc' else 0.55
    # threshold = 0.3
    data = data[data[ml_metric] >= threshold]
    acc = data[ml_metric]
    fairness = data[fairness_metric]
    # use a gradient based on lambda
    plt.scatter(fairness, acc, c=data["lambda"], cmap="viridis")
    plt.xlabel(FAIRNESS_METRIC_LONG_NAMES[fairness_metric])
    plt.ylabel("Accuracy" if ml_metric == 'acc' else "F1")
    # add colorbar
    plt.colorbar()
    # save to file
    complete_path = image_path / ml_metric
    plt.savefig(complete_path / f"{fairness_metric}_{IDX_TO_NAME[idx]}.pdf")
    plt.clf()


def plot_fairness_comparison(
    data_files: list[Path],
    method_names: list[str],
    colors: list[str],
    shapes: list[str],
    image_path: Path,
    fairness_metric: str,
    idx: int,
    ml_metric: str = 'acc'
) -> None:
    """
    Plot fairness metric.
    X-axis is the corresponding fairness metric.
    Y-axis is the accuracy.
    :param shapes:
    :param colors:
    :param method_names:
    :param data_files:
    :param image_path:
    :param fairness_metric:
    :param idx:
    :param ml_metric:
    """
    method_names = method_names.copy()
    colors = colors.copy()
    shapes = shapes.copy()
    for data_file in data_files:
        if os.path.isfile(data_file):
            data = pd.read_csv(data_file)
            # Exclude experiments with accuracy < 0.75
            # And f1 < 0.35
            threshold = 0.8 if ml_metric == 'acc' else 0.55
            # threshold = 0.3
            data = data[data[ml_metric] >= threshold]
            acc = data[ml_metric]
            fairness = data[FAIRNESS_METRIC_SHORT_NAMES[fairness_metric]]
            if fairness_metric == "disparate_impact":
                colors.pop(0)
                shapes.pop(0)
            plt.scatter(fairness, acc, color=colors.pop(0), marker=shapes.pop(0))

    if "Our" in method_names:
        # add non constrained network results
        data_file = data_files[method_names.index("Our")]
        data = pd.read_csv(data_file)
        # Take only the non-constrained network, i.e., lambda = 0
        data = data[data["lambda"] == 0.0]
        acc = data[ml_metric]
        fairness = data[FAIRNESS_METRIC_SHORT_NAMES[fairness_metric]]
        # use a bigger font
        plt.scatter(fairness, acc, color="black", marker="X", s=100)
        method_names.append("Vanilla")

    if "demographic_parity" in fairness_metric and idx == 8 and ml_metric == 'acc':
        # FNNC
        plt.scatter([0.01], [0.84], color="purple", marker="s", s=100)
        method_names.append("FNNC")
        # Wagner
        plt.scatter([0.01], [0.877], color="orange", marker="P", s=100)
        method_names.append("Wagner")

    if "equalized_odds" in fairness_metric:
        method_names.remove("Jiang")
        if idx == 8 and ml_metric == 'acc':
            # FNNC
            plt.scatter([0.02], [0.839], color="purple", marker="s", s=100)
            method_names.append("FNNC")

    if "disparate_impact" in fairness_metric:
        method_names.remove("Jiang")
        method_names.remove("Cho")
        if idx == 8 and ml_metric == 'acc':
            # FNNC
            plt.scatter([0.8], [0.822], color="purple", marker="s", s=100)
            method_names.append("FNNC")
            # Wagner
            plt.scatter([0.8], [0.877], color="orange", marker="P", s=100)
            method_names.append("Wagner")

    plt.xlabel(
        FAIRNESS_METRIC_LONG_NAMES[FAIRNESS_METRIC_SHORT_NAMES[fairness_metric]],
        fontsize=18,
    )
    plt.ylabel("Accuracy" if ml_metric == 'acc' else "F1", fontsize=18)

    # add legend
    method_names = [method_name.replace("Our", "FaUCI") for method_name in method_names]
    plt.legend(method_names)

    # save to file
    complete_path = image_path / ml_metric
    plt.savefig(complete_path / f"{fairness_metric}_{IDX_TO_NAME[idx]}.pdf")
    plt.clf()
