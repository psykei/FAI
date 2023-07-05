from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt


PATH = Path(__file__).parents[0]
FAIRNESS_METRIC_LONG_NAMES = {
    "dp": "Demographic parity",
    "di": "Disparate impact",
    "eo": "Equalized odds",
}


def plot_fairness_metric(file: Path, fairness_metric: str) -> None:
    """
    Plot fairness metric.
    X-axis is the corresponding fairness metric.
    Y-axis is the accuracy.
    :param file:
    :param fairness_metric:
    """
    data = pd.read_csv(file)
    acc = data['acc']
    fairness = data[fairness_metric]
    plt.scatter(fairness, acc)
    plt.xlabel(FAIRNESS_METRIC_LONG_NAMES[fairness_metric])
    plt.ylabel('Accuracy')
    # save to file
    plt.savefig(PATH / f"{fairness_metric}.pdf")
    plt.clf()
