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
IDX = 8  # index of the sensitive attribute [0 = age, 7 = ethnicity, 8 = sex,...]
CUSTOM_METRICS = ["demographic_parity"]
# CUSTOM_METRICS = ["demographic_parity", "disparate_impact", "equalized_odds"]
ONE_HOT = False
IDX_TO_NAME = {
    0: "age",
    7: "ethnicity",
    8: "sex",
}

# Hyperparameters of our method
MAX_LAMBDA = 1
STEPS = 0.1
LAMBDAS = [((MAX_LAMBDA * (1 / STEPS)) - i)/(1/STEPS) for i in range(0, int(MAX_LAMBDA * (1 / STEPS)))]
LAMBDAS = sorted(LAMBDAS, reverse=False)

# Target fairness metrics and accuracy
TARGET_ACCURACY = 0.9
TARGET_FAIRNESS_METRIC = 0.01
TARGET_DISPARATE_IMPACT = 0.99

# Pytorch method configuration
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Cho's method configuration
CHO_H = 0.1
CHO_DELTA = 1.0
CHO_MAX_LAMBDA = 1
CHO_STEPS = 0.02
CHO_LAMBDAS = [((CHO_MAX_LAMBDA * (1 / CHO_STEPS)) - i)/(1/CHO_STEPS) for i in range(0, int(CHO_MAX_LAMBDA * (1 / CHO_STEPS)))]
CHO_METRICS = ["demographic_parity"]
# CHO_METRICS = ["demographic_parity", "disparate_impact", "equalized_odds"]

# Jiang's method configuration
JIANG_MAX_LAMBDA = 1
JIANG_STEPS = 0.1
JIANG_LAMBDAS = [((JIANG_MAX_LAMBDA * (1 / JIANG_STEPS)) - i)/(1/JIANG_STEPS) for i in range(0, int(JIANG_MAX_LAMBDA * (1 / JIANG_STEPS)))]
JIANG_METRICS = ["demographic_parity"]


def initialize_experiment(filename: str, metric: str, l: float, preprocess: bool = True, min_max: bool = False):
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
        f"\n\tIDX={IDX}"
        f"\n\tCUSTOM_METRIC={metric}"
        f"\n\tLAMBDA={l}"
        f"\n\tONE_HOT={ONE_HOT}"
    )

    loader = AdultLoader()
    dataset = loader.load_preprocessed(all_datasets=True, one_hot=False, preprocess=preprocess, min_max=min_max)
    train, test = train_test_split(dataset, test_size=0.2, random_state=SEED, stratify=dataset["income"])
    kfold = KFold(n_splits=K, shuffle=True, random_state=SEED)
    return dataset, train, test, kfold
