import re
from pathlib import Path
import numpy as np
import pandas as pd
from dataset import get_feature_name
from experiments import CACHE_PATH
from experiments._logging import exp_name
from analysis.images import FAIRNESS_METRIC_SHORT_NAMES, plot_fairness_comparison, FAIRNESS_METRIC_LONG_NAMES
from experiments.configuration import PATH as CONF_PATH, from_yaml_file_to_dict, K, get_lambda_values


PATH = Path(__file__).parents[0]
IDX_TO_NAME = {
    0: "age",
    7: "ethnicity",
    8: "sex",
}
HEADER = ["accuracy", "precision", "recall", "f1", "auc", "demographic_parity", "disparate_impact", "equalized_odds", "lambda"]


def get_list_of_files() -> list[Path]:
    return [f for f in CACHE_PATH.iterdir() if f.is_file() and f.suffix == ".yml"]


def get_results_from_setup(
    dataset: str,
    method: str,
    metric: str,
    lambda_value: float,
    seed: int,
    k: int,
    protected: int,
) -> list[float]:
    experiment_log_files = get_list_of_files()
    names = [exp_name(dataset, method, metric, protected, lambda_value, exp_number, seed) for exp_number in range(k)]
    files = [CACHE_PATH / name for name in names if CACHE_PATH / name in experiment_log_files]
    accs, precs, recs, f1s, aucs, dps, dis, eos, lambdas = [], [], [], [], [], [], [], [], []
    for file in files:
        results = from_yaml_file_to_dict(file)["metrics"]
        accs.append(results["accuracy"])
        precs.append(results["precision"])
        recs.append(results["recall"])
        f1s.append(results["f1"])
        aucs.append(results["auc"])
        dps.append(results["demographic_parity"])
        dis.append(results["disparate_impact"])
        eos.append(results["equalized_odds"])
        lambdas.append(lambda_value)
    results = [accs, precs, recs, f1s, aucs, dps, dis, eos, lambdas]
    # prevent mean of empty slice
    if not results[0]:
        return [0.0 for _ in range(len(HEADER))]
    return [np.mean(result).mean() for result in results]


def get_results_from_configuration(configuration_file_name: str) -> list[pd.DataFrame]:
    configuration = from_yaml_file_to_dict(CONF_PATH / configuration_file_name)
    dataset = configuration["dataset"]
    method = configuration["method"]
    metric = configuration["metric"]
    protected = configuration["protected"]
    exp_seed = configuration["seed"]
    lambda_values_lists = get_lambda_values(configuration["lambda"])
    results = []
    for feature, lambda_values in zip(protected, lambda_values_lists):
        single_setup_results = []
        for lambda_value in lambda_values:
            single_setup_results.append(
                get_results_from_setup(
                    dataset=dataset,
                    method=method,
                    metric=metric,
                    lambda_value=lambda_value,
                    seed=exp_seed,
                    k=K,
                    protected=feature,
                )
            )
        df_result = pd.DataFrame(single_setup_results, columns=HEADER)
        results.append(df_result)
    return results


def get_all_results(configuration_file_names: list[str]) -> dict[str, dict[str, dict[str, dict[str, pd.DataFrame]]]]:
    results = {}
    tmp_results = {}
    for configuration_file_name in configuration_file_names:
        config_info = from_yaml_file_to_dict(CONF_PATH / configuration_file_name)
        dataset = config_info["dataset"]
        protected = config_info["protected"]
        method = config_info["method"]
        metric = config_info["metric"]
        results_dfs = get_results_from_configuration(configuration_file_name)
        for feature, result in zip(protected, results_dfs):
            feature_name = get_feature_name(dataset, feature)
            short_metric = FAIRNESS_METRIC_SHORT_NAMES[metric]
            tmp_results[f"{dataset}_{method}_{short_metric}_{feature_name}"] = result
    for key, value in tmp_results.items():
        dataset, method, metric, feature = re.split(r"_", key)
        if dataset not in results:
            results[dataset] = {}
        if metric not in results[dataset]:
            results[dataset][metric] = {}
        if feature not in results[dataset][metric]:
            results[dataset][metric][feature] = {}
        results[dataset][metric][feature][method] = value
    return results


def perform_analysis(configuration_file_names: list[str]):
    results = get_all_results(configuration_file_names)
    for dataset in results.keys():
        for metric in results[dataset].keys():
            for feature in results[dataset][metric].keys():
                dfs = []
                for method in results[dataset][metric][feature].keys():
                    df = results[dataset][metric][feature][method]
                    dfs.append(df)
                metric_long_name = FAIRNESS_METRIC_LONG_NAMES[metric]
                # plot_fairness_comparison(dfs, dataset, feature, metric_long_name, "accuracy", list(results[dataset][metric][feature].keys()))
                plot_fairness_comparison(dfs, dataset, feature, metric_long_name, "f1", list(results[dataset][metric][feature].keys()))