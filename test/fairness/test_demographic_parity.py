import unittest
from dataset.adult_data_pipeline import AdultLoader
from fairness import enable_logging, LOG_INFO, logger
from fairness.metric import is_demographic_parity


class TestDemographicParity(unittest.TestCase):

    adult_train_p = None
    adult_train_y = None
    protected_attribute = "Sex"
    high_epsilon = 0.4

    @classmethod
    def setUpClass(cls) -> None:
        enable_logging(LOG_INFO)
        logger.info("Setting up TestDemographicParity")

    @classmethod
    def tearDownClass(cls) -> None:
        logger.info("Tearing down TestDemographicParity")

    def setUp(self):
        loader = AdultLoader()
        adult_train_dataset, _, _ = loader.load_preprocessed_split()
        self.adult_train_p = adult_train_dataset[self.protected_attribute].values
        self.adult_train_y = adult_train_dataset.iloc[:, -1].values

    def test_unfair_model(self):
        self.assertTrue(is_demographic_parity(self.adult_train_p, self.adult_train_y))

    def test_unfair_model_with_high_epsilon(self):
        self.assertFalse(is_demographic_parity(self.adult_train_p, self.adult_train_y, epsilon=self.high_epsilon))
