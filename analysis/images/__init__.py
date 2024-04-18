from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt


PATH = Path(__file__).parents[0]
FAIRNESS_METRIC_PRETTY_NAMES = {
    "dp": "Demographic Parity",
    "di": "Disparate Impact",
    "eo": "Equalized Odds",
}
FAIRNESS_METRIC_LONG_NAMES = {
    "dp": "demographic_parity",
    "di": "disparate_impact",
    "eo": "equalized_odds",
}
FAIRNESS_METRIC_SHORT_NAMES = {
    "demographic_parity": "dp",
    "disparate_impact": "di",
    "equalized_odds": "eo",
}
METHOD_COLORS = {
    "fauci": "red",
    "cho": "azure",
    "jiang": "green",
    "vanilla": "black",
    "fnnc": "purple",
    "wagner": "orange",
}
METHOD_SHAPES = {
    "fauci": "s",
    "cho": "D",
    "jiang": "o",
    "vanilla": "X",
    "fnnc": "s",
    "wagner": "P",
}


def plot_fairness_comparison(
    dfs: list[pd.DataFrame],
    dataset: str,
    feature: str,
    fairness_metric: str,
    ml_metric: str,
    methods: list[str],
) -> None:
    """
    Plot fairness comparison.
    :param dfs: data
    :param dataset: the name of the dataset
    :param feature: the name of the protected feature
    :param fairness_metric: the fairness metric
    :param ml_metric: the machine learning metric
    :param methods: the methods
    :return:
    """
    for df, method in zip(dfs, methods):
        # don't plot when lambda is 0 (this is like using a vanilla model)
        df_copy = df[df["lambda"] != 0]
        plt.scatter(df_copy[fairness_metric], df_copy[ml_metric], s=80, label=method, color=METHOD_COLORS[method], marker=METHOD_SHAPES[method])
    # Add Vanilla, i.e. Fauci when lambda is 0
    vanilla_df = dfs[0][dfs[0]["lambda"] == 0]
    plt.scatter(vanilla_df[fairness_metric], vanilla_df[ml_metric], s=80, label="Vanilla", color=METHOD_COLORS["vanilla"], marker=METHOD_SHAPES["vanilla"])
    plt.ylabel(ml_metric)
    plt.xlabel(fairness_metric)
    plt.title(f"{ml_metric} vs {FAIRNESS_METRIC_PRETTY_NAMES[FAIRNESS_METRIC_SHORT_NAMES[fairness_metric]]} ({dataset}, {feature})")
    plt.legend()
    # plt.show()
    # Save to file
    file_name = f"{dataset}_{feature}_{FAIRNESS_METRIC_SHORT_NAMES[fairness_metric]}_{ml_metric}.pdf"
    plt.savefig(PATH / file_name)
    plt.clf()
