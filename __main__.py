import numpy as np
from dataset.tf_data_pipeline import AdultLoader, Processor
from fairness import enable_logging
from fairness.metric import is_demographic_parity, is_disparate_impact, is_equalized_odds

enable_logging()
loader = AdultLoader()
processor = Processor()
dataset = loader.load()
dataset = processor.setup(dataset)
train, valid, test = loader.split(dataset)

dp = is_demographic_parity(np.array(train["Sex"]), np.array(train["income"]))
di = is_disparate_impact(np.array(train["Sex"]), np.array(train["income"]))
predicted = np.array(train["income"]) + np.random.randint(-1, 1, len(train["income"]))
predicted = np.clip(predicted, 0, 1)
eo = is_equalized_odds(np.array(train["Sex"]), np.array(train["income"]), predicted)
print("Demographic parity: ", dp)
print("Disparate impact: ", di)
print("Equalized odds: ", eo)

