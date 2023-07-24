import unittest

from sklearn.tree import DecisionTreeClassifier

from dataset.adult_data_pipeline import AdultLoader
from fairness import enable_logging, LOG_INFO, logger
from fairness.metric import equalized_odds


class TestEqualizedOdds(unittest.TestCase):
    seed = 0
    adult_train_p = None
    adult_train_y = None
    adult_train_x = None
    adult_test_p = None
    adult_test_y = None
    protected_attribute = "Sex"
    high_epsilon = 0.3
    low_epsilon = 1e-3

    @classmethod
    def setUpClass(cls) -> None:
        enable_logging(LOG_INFO)
        logger.info("Setting up TestEqualizedOdds")

    @classmethod
    def tearDownClass(cls) -> None:
        logger.info("Tearing down TestEqualizedOdds")

    def _get_prediction_from_decision_tree(self, train_x, train_y, test_x):
        model = DecisionTreeClassifier(max_depth=10, random_state=self.seed)
        model.fit(train_x, train_y)
        adult_prediction = model.predict(test_x)
        return adult_prediction

    def setUp(self):
        loader = AdultLoader()
        adult_train_dataset, _, adult_test_dataset = loader.load_preprocessed_split()
        self.adult_train_p = adult_train_dataset[self.protected_attribute].values
        self.adult_train_x = adult_train_dataset.iloc[:, :-1].values
        self.adult_train_y = adult_train_dataset.iloc[:, -1].values
        self.adult_test_p = adult_test_dataset[self.protected_attribute].values
        self.adult_test_x = adult_test_dataset.iloc[:, :-1].values
        self.adult_test_y = adult_test_dataset.iloc[:, -1].values

    def test_perfect_model(self):
        self.assertTrue(
            equalized_odds(
                self.adult_train_p, self.adult_train_y, self.adult_train_y, numeric=False
            )
        )

    def test_model_with_low_epsilon(self):
        adult_prediction = self._get_prediction_from_decision_tree(
            self.adult_train_x, self.adult_train_y, self.adult_test_x
        )
        self.assertFalse(
            equalized_odds(
                self.adult_test_p, self.adult_test_y, adult_prediction, self.low_epsilon, numeric=False
            )
        )

    def test_model_with_high_epsilon(self):
        adult_prediction = self._get_prediction_from_decision_tree(
            self.adult_train_x, self.adult_train_y, self.adult_test_x
        )
        self.assertTrue(
            equalized_odds(
                self.adult_test_p,
                self.adult_test_y,
                adult_prediction,
                epsilon=self.high_epsilon,
                numeric=False
            )
        )


if __name__ == "__main__":
    unittest.main()
