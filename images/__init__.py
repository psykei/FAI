from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

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


def plot_fairness_metric(data_file: Path, image_path: Path, fairness_metric: str, idx: int) -> None:
    """
    Plot fairness metric.
    X-axis is the corresponding fairness metric.
    Y-axis is the accuracy.
    :param data_file:
    :param image_path:
    :param fairness_metric:
    :param idx:
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
    plt.savefig(image_path / f"{fairness_metric}_{IDX_TO_NAME[idx]}.pdf")
    plt.clf()


def plot_fairness_comparison(data_files: list[Path], image_path: Path, fairness_metric: str, idx: int) -> None:
    """
    Plot fairness metric.
    X-axis is the corresponding fairness metric.
    Y-axis is the accuracy.
    :param data_files:
    :param image_path:
    :param fairness_metric:
    """
    methods_name = ['Our', 'Cho', 'Jiang']
    colors = ['#20E635', '#09B5E6', '#E63F14']
    shapes = ['o', 'd', '*']
    for data_file in data_files:
        data = pd.read_csv(data_file)
        # Exclude experiments with accuracy < 0.5
        data = data[data['acc'] >= 0.5]
        acc = data['acc']
        fairness = data[FAIRNESS_METRIC_SHORT_NAMES[fairness_metric]]
        # plt.scatter(fairness, acc, color=colors.pop(), marker=shapes.pop())
        step = 5
        x = [np.mean(acc[int(i*step):int((i+1)*step)]) for i in range(int(acc.shape[0] / step))]
        y = [np.mean(fairness[int(i*step):int((i+1)*step)]) for i in range(int(fairness.shape[0] / step))]
        hidden_y = np.array(fairness).reshape(-1, 1)
        color = colors.pop()
        plt.plot(y, x, color=color, label=data_file.stem)

    plt.xlabel(FAIRNESS_METRIC_LONG_NAMES[FAIRNESS_METRIC_SHORT_NAMES[fairness_metric]])
    plt.ylabel('Accuracy')
    # add legend
    plt.legend(methods_name)

    colors = ['#20E635', '#09B5E6', '#E63F14']
    for data_file in data_files:
        data = pd.read_csv(data_file)
        # Exclude experiments with accuracy < 0.5
        data = data[data['acc'] >= 0.5]
        acc = data['acc']
        fairness = data[FAIRNESS_METRIC_SHORT_NAMES[fairness_metric]]
        # plt.scatter(fairness, acc, color=colors.pop(), marker=shapes.pop())
        step = 5
        x = [np.mean(acc[int(i*step):int((i+1)*step)]) for i in range(int(acc.shape[0] / step))]
        y = [np.mean(fairness[int(i*step):int((i+1)*step)]) for i in range(int(fairness.shape[0] / step))]
        hidden_y = np.array(fairness).reshape(-1, 1)
        std_devs = [np.std(hidden_y[int(i*step):int((i+1)*step)]) for i in range(int(hidden_y.shape[0] / step))]
        color = colors.pop()
        plt.fill_between(y, np.array(x) - np.array(std_devs), np.array(x) + np.array(std_devs), color=color, alpha=0.2)

    # save to file
    plt.savefig(image_path / f"{fairness_metric}_{IDX_TO_NAME[idx]}.pdf")
    plt.clf()
