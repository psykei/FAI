# General experiments configuration
import torch
from sklearn.model_selection import train_test_split, KFold
from dataset.adult_data_pipeline import AdultLoader
from fairness import enable_logging, logger, enable_file_logging

LOG = "log"
SEED = 0
K = 5
EPOCHS = 5000
BATCH_SIZE = 500
NEURONS_PER_LAYER = [100, 50]
VERBOSE = 0
IDXS = [8, 7, 0]  # index of the sensitive attribute [0 = age, 7 = ethnicity, 8 = sex,...]
# IDXS = [7]
CUSTOM_METRICS = ["demographic_parity", "disparate_impact", "equalized_odds"]
# CUSTOM_METRICS = ["demographic_parity", "disparate_impact", "equalized_odds"]
ONE_HOT = False
IDX_TO_NAME = {
    0: "age",
    7: "ethnicity",
    8: "sex",
}

IDX_TO_IDX = {
    0: 0,
    7: 1,
    8: 2,
}

# Pytorch method configuration
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Target fairness metrics and accuracy
TARGET_ACCURACY = 0.9
TARGET_FAIRNESS_METRIC = 0.01
TARGET_DISPARATE_IMPACT = 0.99


def generate_lambdas(max_lambda: float, steps: float, min_lambda: float = 0.):
    result = [((max_lambda * (1 / steps)) - i) / (1 / steps) for i in range(0, int(max_lambda * (1 / steps)) + 1)]
    return [r for r in result if r >= min_lambda]


# Hyperparameters of our method
OUR_MAX_LAMBDAS_DP = [10, 100, 4]
OUR_MIN_LAMBDAS_DP = [0, 0, 0]  # [0, 10, 0]
OUR_STEPS_DP = [0.1, 1, 0.04]

OUR_MAX_LAMBDAS_DI = [0.5, 0.1, 0.3]  # [1, 1, 4]
OUR_MIN_LAMBDAS_DI = [0, 0, 0]
OUR_STEPS_DI = [0.01, 0.001, 0.01]  # [0.1, 0.01, 0.5]

OUR_MAX_LAMBDAS_EO = [5, 3, 12]
OUR_MIN_LAMBDAS_EO = [0, 0, 0]
OUR_STEPS_EO = [0.1, 0.1, 0.1]


def our_lambdas(index: int, metric: str = "demographic_parity"):
    if metric == "demographic_parity":
        result = generate_lambdas(OUR_MAX_LAMBDAS_DP[IDX_TO_IDX[index]], OUR_STEPS_DP[IDX_TO_IDX[index]], OUR_MIN_LAMBDAS_DP[IDX_TO_IDX[index]])
    elif metric == "equalized_odds":
        result = generate_lambdas(OUR_MAX_LAMBDAS_EO[IDX_TO_IDX[index]], OUR_STEPS_EO[IDX_TO_IDX[index]], OUR_MIN_LAMBDAS_EO[IDX_TO_IDX[index]])
    elif metric == "disparate_impact":
        result = generate_lambdas(OUR_MAX_LAMBDAS_DI[IDX_TO_IDX[index]], OUR_STEPS_DI[IDX_TO_IDX[index]], OUR_MIN_LAMBDAS_DI[IDX_TO_IDX[index]])
    else:
        raise ValueError(f"Unknown metric: {metric}")
    return result if 0. in result else result + [0.]


# Cho's method configuration
# Lambda is always bounded between 0 and 1
CHO_H = 0.1
CHO_DELTA = 1.0

CHO_MAX_LAMBDAS_DP = [1, 1, 1]
CHO_MIN_LAMBDAS_DP = [0, 0.99, 0]  # [0, 0.985, 0]
CHO_STEPS_DP = [0.01, 0.0001, 0.01]

CHO_MAX_LAMBDAS_EO = [1, 0.999, 1]
CHO_MIN_LAMBDAS_EO = [0.85, 0.995, 0]
CHO_STEPS_EO = [0.001, 0.0001, 0.05]


def cho_lambdas(index: int, metric: str = "demographic_parity"):
    if metric == "demographic_parity":
        if index == 7:
            return generate_lambdas(0.98, 0.02, 0.952) + generate_lambdas(0.95, 0.05, 0.5) + generate_lambdas(CHO_MAX_LAMBDAS_DP[1], CHO_STEPS_DP[1], CHO_MIN_LAMBDAS_DP[1])
        else:
            return generate_lambdas(CHO_MAX_LAMBDAS_DP[IDX_TO_IDX[index]], CHO_STEPS_DP[IDX_TO_IDX[index]], CHO_MIN_LAMBDAS_DP[IDX_TO_IDX[index]])
    elif metric == "equalized_odds":
        if index == 0:
            return generate_lambdas(CHO_MAX_LAMBDAS_EO[0], 0.0001, 0.999) + generate_lambdas(0.8, 0.05, 0) + generate_lambdas(CHO_MAX_LAMBDAS_EO[0], CHO_STEPS_EO[0], CHO_MIN_LAMBDAS_EO[0])
        if index == 8:
            return generate_lambdas(CHO_MAX_LAMBDAS_EO[0], 0.001, 0.98) + generate_lambdas(CHO_MAX_LAMBDAS_EO[2], CHO_STEPS_EO[2], CHO_MIN_LAMBDAS_EO[2])
        else:
            return generate_lambdas(CHO_MAX_LAMBDAS_EO[IDX_TO_IDX[index]], CHO_STEPS_EO[IDX_TO_IDX[index]], CHO_MIN_LAMBDAS_EO[IDX_TO_IDX[index]])
    else:
        raise ValueError(f"Unknown metric: {metric}")


CHO_METRICS = ["demographic_parity", "equalized_odds"]
# CHO_METRICS = ["demographic_parity", "disparate_impact", "equalized_odds"]

# Jiang's method configuration
JIANG_MAX_LAMBDA = [100, 120, 3]
JIANG_MIN_LAMBDAS = [0, 0, 0]  # [0, 20, 0]
JIANG_STEPS = [1, 1, 0.05]


def jiang_lambdas(index: int):
    return generate_lambdas(JIANG_MAX_LAMBDA[IDX_TO_IDX[index]], JIANG_STEPS[IDX_TO_IDX[index]], JIANG_MIN_LAMBDAS[IDX_TO_IDX[index]])


JIANG_METRICS = ["demographic_parity"]


def initialize_experiment(filename: str, metric: str, idx: int, l: float, preprocess: bool = True,
                          min_max: bool = False):
    enable_logging()
    logger.info(f"Logging to {filename}")
    enable_file_logging(filename)
    logger.info(
        f"Parameters:"
        f"\n\tSEED={SEED}"
        f"\n\tK={K}"
        f"\n\tEPOCHS={EPOCHS}"
        f"\n\tBATCH_SIZE={BATCH_SIZE}"
        f"\n\tNEURONS_PER_LAYER={NEURONS_PER_LAYER}"
        f"\n\tIDX={idx}"
        f"\n\tCUSTOM_METRIC={metric}"
        f"\n\tLAMBDA={l}"
        f"\n\tONE_HOT={ONE_HOT}"
    )

    loader = AdultLoader()
    dataset = loader.load_preprocessed(all_datasets=True, one_hot=False, preprocess=preprocess, min_max=min_max)
    train, test = train_test_split(dataset, test_size=0.2, random_state=SEED, stratify=dataset["income"])
    kfold = KFold(n_splits=K, shuffle=True, random_state=SEED)
    return dataset, train, test, kfold
