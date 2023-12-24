from pathlib import Path
import re
from fairness import PATH as FAIRNESS_PATH


PATH = Path(__file__).parents[0]
LOG = "log"
LOG_PATH = FAIRNESS_PATH / LOG
MINIMUM_AMOUNT_OF_ROWS = 4
SEED_ROW = 1
K_ROW = 2
EPOCHS_ROW = 3
BATCH_SIZE_ROW = 4
NEURONS_PER_LAYER_ROW = 5
IDX_ROW = 6
CUSTOM_METRIC_ROW = 7
LAMBDA_ROW = 8


def get_list_of_files(path: Path, extension: str = ".txt") -> list[Path]:
    return [f for f in path.iterdir() if f.is_file() and f.suffix == extension]


def is_seed_equal(file: Path, seed: int) -> bool:
    with open(file, "r") as f:
        lines = f.readlines()
        return int(lines[SEED_ROW].split("=")[1]) == seed


def is_k_equal(file: Path, k: int) -> bool:
    with open(file, "r") as f:
        lines = f.readlines()
        return int(lines[K_ROW].split("=")[1]) == k


def is_epochs_equal(file: Path, epochs: int) -> bool:
    with open(file, "r") as f:
        lines = f.readlines()
        return int(lines[EPOCHS_ROW].split("=")[1]) == epochs


def is_batch_size_equal(file: Path, batch_size: int) -> bool:
    with open(file, "r") as f:
        lines = f.readlines()
        return int(lines[BATCH_SIZE_ROW].split("=")[1]) == batch_size


def is_neurons_per_layer_equal(file: Path, neurons_per_layer: list[int]) -> bool:
    with open(file, "r") as f:
        lines = f.readlines()
        return lines[NEURONS_PER_LAYER_ROW].split("=")[1].strip() == str(
            neurons_per_layer
        )


def is_idx_equal(file: Path, idx: int) -> bool:
    with open(file, "r") as f:
        lines = f.readlines()
        return int(re.match(r"[0-9]*", lines[IDX_ROW].split("=")[1])[0]) == idx


def is_custom_metric_equal(file: Path, custom_metric: str) -> bool:
    with open(file, "r") as f:
        lines = f.readlines()
        return lines[CUSTOM_METRIC_ROW].split("=")[1].strip() == custom_metric


def is_lambda_equal(file: Path, l: float) -> bool:
    with open(file, "r") as f:
        lines = f.readlines()
        return float(lines[LAMBDA_ROW].split("=")[1]) == l


def is_complete(file: Path) -> bool:
    import os

    def is_value_present(line: str) -> bool:
        return len(line.split(":")) == 2

    with open(file, "r") as f:
        lines = f.readlines()
        condition = len(lines) >= MINIMUM_AMOUNT_OF_ROWS
        for idx in range(1, MINIMUM_AMOUNT_OF_ROWS + 1):
            condition &= is_value_present(lines[-idx])
        # if not condition:
        #     print(f"Removing {file} because it is incomplete.")
        #     os.remove(file)
        return condition


def get_files_from_parameters(
    custom_metric: str = "None",
    path=LOG_PATH,
    l: float = 1,
    batch_size: int = 500,
    epochs: int = 5000,
    neurons_per_layer=None,
    seed: int = 0,
    k: int = 5,
    idx: int = 3,
) -> list[Path]:
    if neurons_per_layer is None:
        neurons_per_layer = [100, 50]
    experiment_log_files = get_list_of_files(path)
    experiment_log_files = [f for f in experiment_log_files if is_seed_equal(f, seed)]
    experiment_log_files = [f for f in experiment_log_files if is_k_equal(f, k)]
    experiment_log_files = [
        f for f in experiment_log_files if is_epochs_equal(f, epochs)
    ]
    experiment_log_files = [
        f for f in experiment_log_files if is_batch_size_equal(f, batch_size)
    ]
    experiment_log_files = [
        f
        for f in experiment_log_files
        if is_neurons_per_layer_equal(f, neurons_per_layer)
    ]
    experiment_log_files = [f for f in experiment_log_files if is_idx_equal(f, idx)]
    experiment_log_files = [
        f for f in experiment_log_files if is_custom_metric_equal(f, custom_metric)
    ]
    experiment_log_files = [f for f in experiment_log_files if is_lambda_equal(f, l)]
    experiment_log_files = [f for f in experiment_log_files if is_complete(f)]
    return experiment_log_files


def get_final_metrics_from_file(file: Path):
    with open(file, "r") as f:
        lines = f.readlines()
        results = []
        for i in range(0, 7):
            results.append(float(lines[i - 7].split(":")[1]))
        return results
