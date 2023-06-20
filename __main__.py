import numpy as np
from dataset.adult_data_pipeline import AdultLoader
from fairness import enable_logging, logger
from fairness.metric import (
    is_demographic_parity,
    is_disparate_impact,
    is_equalized_odds,
)

enable_logging()
loader = AdultLoader()
dataset = loader.load_preprocessed()
train, valid, test = loader.load_preprocessed_split()

male_data = dataset[dataset["Sex"] == 0]
female_data = dataset[dataset["Sex"] == 1]
male_loan_rate = sum(male_data.income) / male_data.shape[0]
female_loan_rate = sum(female_data.income) / female_data.shape[0]
logger.info(f"Male percentage is {male_data.shape[0] / dataset.shape[0] * 100:.1f}%")
logger.info(f"Male loan rate is {male_loan_rate:.3f}")
logger.info(f"Female percentage is {female_data.shape[0] / dataset.shape[0] * 100:.1f}%")
logger.info(f"Female loan rate is {female_loan_rate:.3f}")


dp = is_demographic_parity(np.array(train["Sex"]), np.array(train["income"]))
di = is_disparate_impact(np.array(train["Sex"]), np.array(train["income"]))
predicted = np.array(train["income"]) + np.random.randint(-1, 1, len(train["income"]))
predicted = np.clip(predicted, 0, 1)
eo = is_equalized_odds(np.array(train["Sex"]), np.array(train["income"]), predicted)
logger.info(f"Demographic parity: {dp}")
logger.info(f"Disparate impact: {di}")
logger.info(f"Equalized odds: {eo}")
