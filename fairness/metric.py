import numpy as np
import fairness

EPSILON: float = 1e-2
DISPARATE_IMPACT_THRESHOLD: float = 0.8


def is_demographic_parity(p: np.array, y: np.array, epsilon=EPSILON, numeric: bool = True) -> bool or float:
    """
    Demographic parity is a measure of fairness that measures if a value of a protected feature impacts the outcome of a
    prediction. In other words, it measures if the outcome is independent of the protected feature.
    The protected feature must be binary.
    The output must be binary.
    :param p: protected feature
    :param y: output
    :param epsilon: threshold for demographic parity
    :param numeric: if True, return the value of demographic parity instead of a boolean
    :return: True if demographic parity is less than epsilon, False otherwise
    """
    assert len(np.unique(p)) <= 2, "Demographic parity is only defined for binary protected features"
    conditional_prob_zero = y[p == 0]
    if conditional_prob_zero.shape[0] == 0:
        conditional_prob_zero = 0
    else:
        conditional_prob_zero = np.mean(y[p == 0])
    conditional_prob_one = y[p == 1]
    if conditional_prob_one.shape[0] == 0:
        conditional_prob_one = 0
    else:
        conditional_prob_one = np.mean(y[p == 1])
    parity = np.abs(conditional_prob_zero - np.mean(y)) + np.abs(conditional_prob_one - np.mean(y))
    fairness.logger.info(f"Demographic parity: {parity:.4f}")
    return parity < epsilon if not numeric else parity


def is_disparate_impact(
        p: np.array, y: np.array, threshold: float = DISPARATE_IMPACT_THRESHOLD, numeric: bool = True
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
    :param numeric: if True, return the value of disparate impact instead of a boolean
    :return: True if disparate impact is less than threshold, False otherwise
    """
    protected_feature_values = np.unique(p)
    assert (
            len(protected_feature_values) <= 2
    ), "Disparate impact is only defined for binary protected features"
    first_impact = np.mean(y[p == 0]) / np.mean(y[p == 1])
    assert first_impact > 0, "Cannot divide by zero"
    impact = np.min([first_impact, 1 / first_impact])
    fairness.logger.info(f"Disparate impact: {impact:.4f}")
    return impact > threshold if not numeric else impact


def is_equalized_odds(
        p: np.array, y_true: np.array, y_pred: np.array, epsilon: float = EPSILON, numeric: bool = True
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
    :param numeric: if True, return the value of equalized odds instead of a boolean
    :return: True if equalized odds is satisfied, False otherwise
    """
    conditional_prob_zero = np.mean(y_pred[y_true == 0])
    conditional_prob_one = np.mean(y_pred[y_true == 1])
    double_conditional_prob_zero_zero = np.mean(y_pred[(p == 0) & (y_true == 0)])
    double_conditional_prob_zero_one = np.mean(y_pred[(p == 1) & (y_true == 0)])
    double_conditional_prob_one_zero = np.mean(y_pred[(p == 0) & (y_true == 1)])
    double_conditional_prob_one_one = np.mean(y_pred[(p == 1) & (y_true == 1)])
    equalized_odds = np.sum([
        np.abs(double_conditional_prob_zero_zero - conditional_prob_zero),
        np.abs(double_conditional_prob_zero_one - conditional_prob_zero),
        np.abs(double_conditional_prob_one_zero - conditional_prob_one),
        np.abs(double_conditional_prob_one_one - conditional_prob_one)
    ])
    fairness.logger.info(f"Equalized odds: {equalized_odds:.4f}")
    return equalized_odds < epsilon if not numeric else equalized_odds
