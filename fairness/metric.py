import numpy as np
import fairness

EPSILON: float = 1e-2
DISPARATE_IMPACT_THRESHOLD: float = 0.8


def is_demographic_parity(p: np.array, y: np.array, epsilon=EPSILON, numeric: bool = False) -> bool or float:
    """
    Demographic parity is a measure of fairness that measures if a value of a protected feature impacts the outcome of a
    prediction. In other words, it measures if the outcome is independent of the protected feature.
    The protected feature must be binary.
    The output must be binary.
    :param p: protected feature
    :param y: output
    :param epsilon: threshold for demographic parity
    :return: True if demographic parity is less than epsilon, False otherwise
    """
    assert len(np.unique(p)) <= 2, "Demographic parity is only defined for binary protected features"
    parity = np.abs(np.mean(y[p == 0]) - np.mean(y)) + np.abs(np.mean(y[p == 1]) - np.mean(y))
    fairness.logger.info(f"Demographic parity is {parity:.4f}")
    return parity < epsilon if numeric else parity


def is_disparate_impact(
        p: np.array, y: np.array, threshold: float = DISPARATE_IMPACT_THRESHOLD
) -> bool:
    """
    Disparate impact is a measure of fairness that measures if a protected feature impacts the outcome of a prediction.
    It has been defined on binary classification problems as the ratio of the probability of a positive outcome given
    the protected feature to the probability of a positive outcome given the complement of the protected feature.
    If the ratio is less than a threshold (usually 0.8), then the prediction is considered to be unfair.
    The protected feature must be binary.
    The output must be binary.
    :param p: protected feature
    :param y: output
    :param threshold: threshold for disparate impact
    :return: True if disparate impact is less than threshold, False otherwise
    """
    protected_feature_values = np.unique(p)
    assert (
            len(protected_feature_values) <= 2
    ), "Disparate impact is only defined for binary protected features"
    first_impact = np.mean(y[p == 0]) / np.mean(y[p == 1])
    assert first_impact > 0, "Cannot divide by zero"
    impact = np.min([first_impact, 1 / first_impact])
    fairness.logger.info(f"Disparate impact is {impact:.4f}")
    return impact > threshold


def is_equalized_odds(
        p: np.array, y_true: np.array, y_pred: np.array, epsilon: float = EPSILON
) -> bool:
    """
    Equalized odds is a measure of fairness that measures if the output is independent of the protected feature given
    the label Y.
    The protected feature must be binary.
    The output must be binary.
    :param p: protected feature
    :param y_true: ground truth
    :param y_pred: prediction
    :param epsilon: threshold for equalized odds
    :return: True if equalized odds is satisfied, False otherwise
    """
    conditional_prob_zero = np.mean(y_pred[y_true == 0])
    conditional_prob_one = np.mean(y_pred[y_true == 1])
    double_conditional_prob_zero_zero = np.mean(y_pred[(p == 0) & (y_true == 0)])
    double_conditional_prob_zero_one = np.mean(y_pred[(p == 1) & (y_true == 0)])
    double_conditional_prob_one_zero = np.mean(y_pred[(p == 0) & (y_true == 1)])
    double_conditional_prob_one_one = np.mean(y_pred[(p == 1) & (y_true == 1)])
    equalized_odds = np.abs(double_conditional_prob_zero_zero - conditional_prob_zero) + np.abs(
        double_conditional_prob_zero_one - conditional_prob_zero
    ) + np.abs(double_conditional_prob_one_zero - conditional_prob_one) + np.abs(
        double_conditional_prob_one_one - conditional_prob_one
    )
    fairness.logger.info(f"Equalized odds is {equalized_odds:.4f}")
    return equalized_odds < epsilon
