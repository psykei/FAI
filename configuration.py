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
IDX = 8  # index of the sensitive attribute [8 = sex, 7 = ethnicity, ...]
CUSTOM_METRICS = ["equalized_odds"]
# CUSTOM_METRICS = ["demographic_parity", "disparate_impact", "equalized_odds"]
ONE_HOT = False

# Hyperparameters of our method
MAX_LAMBDA = 1
STEPS = 0.01
LAMBDAS = [((MAX_LAMBDA * (1 / STEPS)) - i)/(1/STEPS) for i in range(0, int(MAX_LAMBDA * (1 / STEPS)))]

# Target fairness metrics and accuracy
TARGET_ACCURACY = 0.9
TARGET_FAIRNESS_METRIC = 0.01
TARGET_DISPARATE_IMPACT = 0.99

# Pytorch method configuration
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Cho's method configuration
CHO_H = 0.1
CHO_DELTA = 1.0
MAX_LAMBDA = 1
STEPS = 0.01
CHO_LAMBDAS = [((MAX_LAMBDA * (1 / STEPS)) - i)/(1/STEPS) for i in range(0, int(MAX_LAMBDA * (1 / STEPS)))]
CHO_METRICS = ["demographic_parity"]  # , "equalized_odds"


def initialize_experiment(filename: str, metric: str, l: float, preprocess: bool = True):
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
    dataset = loader.load_preprocessed(all_datasets=True, one_hot=False, preprocess=preprocess)
    train, test = train_test_split(dataset, test_size=0.2, random_state=SEED, stratify=dataset["income"])
    kfold = KFold(n_splits=K, shuffle=True, random_state=SEED)
    return dataset, train, test, kfold
