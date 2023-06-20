import unittest
from dataset.adult_data_pipeline import AdultLoader
from fairness import enable_logging, LOG_INFO, logger
from fairness.metric import is_disparate_impact


class TestDisparateImpact(unittest.TestCase):
    adult_train_p = None
    adult_train_y = None
    protected_attribute = "Sex"
    low_threshold = 0.2

    @classmethod
    def setUpClass(cls) -> None:
        enable_logging(LOG_INFO)
        logger.info("Setting up TestDisparateImpact")

    @classmethod
    def tearDownClass(cls) -> None:
        logger.info("Tearing down TestDisparateImpact")

    def setUp(self):
        loader = AdultLoader()
        adult_train_dataset, _, _ = loader.load_preprocessed_split()
        self.adult_train_p = adult_train_dataset[self.protected_attribute].values
        self.adult_train_y = adult_train_dataset.iloc[:, -1].values

    def test_unfair_model(self):
        self.assertTrue(is_disparate_impact(self.adult_train_p, self.adult_train_y))

    def test_unfair_model_with_low_threshold(self):
        self.assertFalse(is_disparate_impact(self.adult_train_p, self.adult_train_y, threshold=self.low_threshold))
