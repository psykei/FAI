from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt


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


def plot_fairness_metric(data_file: Path, image_path: Path, fairness_metric: str) -> None:
    """
    Plot fairness metric.
    X-axis is the corresponding fairness metric.
    Y-axis is the accuracy.
    :param data_file:
    :param image_path:
    :param fairness_metric:
    """
    data = pd.read_csv(data_file)
    # Exclude experiments with accuracy < 0.5
    data = data[data['acc'] >= 0.5]
    acc = data['acc']
    fairness = data[fairness_metric]
    # use a gradient based on lambda
    plt.scatter(fairness, acc, c=data['lambda'], cmap='viridis')
    plt.xlabel(FAIRNESS_METRIC_LONG_NAMES[fairness_metric])
    plt.ylabel('Accuracy')
    # add colorbar
    plt.colorbar()
    # save to file
    plt.savefig(image_path / f"{fairness_metric}.pdf")
    plt.clf()


def plot_fairness_comparison(data_files: list[Path], image_path: Path, fairness_metric: str) -> None:
    """
    Plot fairness metric.
    X-axis is the corresponding fairness metric.
    Y-axis is the accuracy.
    :param data_files:
    :param image_path:
    :param fairness_metric:
    """
    colors = ['blue', 'red']
    for data_file in data_files:
        data = pd.read_csv(data_file)
        # Exclude experiments with accuracy < 0.5
        data = data[data['acc'] >= 0.5]
        acc = data['acc']
        fairness = data[FAIRNESS_METRIC_SHORT_NAMES[fairness_metric]]
        plt.scatter(fairness, acc, color=colors.pop())
    plt.xlabel(FAIRNESS_METRIC_LONG_NAMES[FAIRNESS_METRIC_SHORT_NAMES[fairness_metric]])
    plt.ylabel('Accuracy')
    # save to file
    plt.savefig(image_path / f"{fairness_metric}.pdf")
    plt.clf()