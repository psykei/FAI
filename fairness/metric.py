import numpy as np
import fairness

EPSILON: float = 1e-2
DISPARATE_IMPACT_THRESHOLD: float = 0.8


def single_conditional_probability(predicted: np.array, protected: np.array, value: int) -> float:
    """
    Calculate the estimated conditioned output distribution of a model.
    The protected attribute can be binary or categorical.
    :param predicted: the predicted labels.
    :param protected: the protected attribute.
    :param value: the value of the protected attribute.
    :return: the conditional probability.
    """
    mask = predicted[protected == value]
    return mask.mean()


def is_demographic_parity(p: np.array, y: np.array, epsilon=EPSILON, numeric: bool = True) -> bool or float:
    """
    Demographic parity is a measure of fairness that measures if a value of a protected feature impacts the outcome of a
    prediction. In other words, it measures if the outcome is independent of the protected feature.
    The protected feature must be binary or categorical.
    The output must be binary.
    :param p: protected feature
    :param y: output
    :param epsilon: threshold for demographic parity
    :param numeric: if True, return the value of demographic parity instead of a boolean
    :return: True if demographic parity is less than epsilon, False otherwise
    """
    unique_p = np.unique(p)
    absolute_probability = np.mean(y)
    parity = 0
    for p_value in unique_p:
        conditional_probability = single_conditional_probability(y, p, p_value)
        if conditional_probability == 0:
            continue
        parity += np.abs(conditional_probability - absolute_probability)
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
    The protected feature must be binary or categorical.
    The output must be binary.
    :param p: protected feature
    :param y: output
    :param threshold: threshold for disparate impact
    :param numeric: if True, return the value of disparate impact instead of a boolean
    :return: True if disparate impact is less than threshold, False otherwise
    """
    unique_protected = np.unique(p)
    probabilities_a = np.array([np.mean(y[p == x]) for x in unique_protected])
    probabilities_not_a = np.array([np.mean(y[p != x]) for x in unique_protected])
    first_impact = np.nan_to_num(probabilities_a / probabilities_not_a)
    if np.any(first_impact <= EPSILON):
        impact = 0
    else:
        impact = np.min([first_impact, np.nan_to_num(1 / first_impact)])
    fairness.logger.info(f"Disparate impact: {impact:.4f}")
    return impact > threshold if not numeric else impact


def is_equalized_odds(
        p: np.array, y_true: np.array, y_pred: np.array, epsilon: float = EPSILON, numeric: bool = True
) -> bool:
    """
    Equalized odds is a measure of fairness that measures if the output is independent of the protected feature given
    the label Y.
    The protected feature must be binary or categorical.
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
    unique_protected = np.unique(p)
    probabilities_a_0 = np.array([np.mean(y_pred[(p == x) & (y_true == 0)]) for x in unique_protected])
    probabilities_a_1 = np.array([np.mean(y_pred[(p == x) & (y_true == 1)]) for x in unique_protected])
    equalized_odds = np.sum(np.abs(np.concatenate([probabilities_a_0 - conditional_prob_zero, probabilities_a_1 - conditional_prob_one])))
    equalized_odds = np.nan_to_num(equalized_odds)
    fairness.logger.info(f"Equalized odds: {equalized_odds:.4f}")
    return equalized_odds < epsilon if not numeric else equalized_odds
