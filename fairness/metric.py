import numpy as np
import fairness

EPSILON: float = 1e-2
DELTA: float = 1e-2
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

def single_conditional_probability_in_range(predicted: np.array, protected: np.array, min_value: float, max_value: float, negate: bool = False) -> float:
    """
    Calculate the estimated conditioned output distribution of a model.
    The protected attribute can be binary or categorical.
    :param predicted: the predicted labels.
    :param protected: the protected attribute.
    :param min_value: the minimum value of the protected attribute.
    :param max_value: the maximum value of the protected attribute.
    :param negate: if True, return the conditional probability of the negated range.
    :return: the conditional probability.
    """
    if negate:
        mask = predicted[np.logical_or(protected < min_value, protected >= max_value)]
    else:
        mask = predicted[np.logical_and(protected >= min_value, protected < max_value)]
    return mask.mean() if len(mask) > 0 else 0.0


def is_demographic_parity(p: np.array, y: np.array, epsilon: float = EPSILON, continuous: bool = False, numeric: bool = True, delta: float = DELTA) -> bool or float:
    """
    Demographic parity is a measure of fairness that measures if a value of a protected feature impacts the outcome of a
    prediction. In other words, it measures if the outcome is independent of the protected feature.
    The protected feature must be binary or categorical.
    The output must be binary.
    :param p: protected feature
    :param y: output
    :param epsilon: threshold for demographic parity
    :param delta: approximation parameter for the calculus of continuous demographic parity
    :param continuous: if True, calculate the continuous demographic parity
    :param numeric: if True, return the value of demographic parity instead of a boolean
    :return: True if demographic parity is less than epsilon, False otherwise
    """
    unique_p = np.unique(p)
    absolute_probability = np.mean(y)
    parity = 0

    def _continuous_demographic_parity() -> float:
        result = 0
        min_protected = np.min(p)
        max_protected = np.max(p)
        interval = max_protected - min_protected
        step_width = interval * delta
        number_of_steps = int(interval / step_width)
        for i in range(number_of_steps):
            min_value = min_protected + i * step_width
            max_value = min_protected + (i + 1) * step_width
            conditional_probability = single_conditional_probability_in_range(y, p, min_value, max_value)
            if conditional_probability == 0:
                continue
            number_of_sample = np.sum(np.logical_and(p >= min_value, p < max_value))
            result += np.abs(conditional_probability - absolute_probability) * number_of_sample
        return result / len(p)

    if continuous:
        parity = _continuous_demographic_parity()
    else:
        for p_value in unique_p:
            conditional_probability = single_conditional_probability(y, p, p_value)
            if conditional_probability == 0:
                continue
            number_of_sample = np.sum(p == p_value)
            parity += np.abs(conditional_probability - absolute_probability) * number_of_sample
        parity /= len(p)
    fairness.logger.info(f"Demographic parity: {parity:.4f}")
    return parity < epsilon if not numeric else parity


def is_disparate_impact(
        p: np.array, y: np.array, threshold: float = DISPARATE_IMPACT_THRESHOLD, continuous: bool = False, numeric: bool = True, delta: float = DELTA
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
    :param continuous: if True, calculate the continuous disparate impact
    :param numeric: if True, return the value of disparate impact instead of a boolean
    :param delta: approximation parameter for the calculus of continuous disparate impact
    :return: True if disparate impact is less than threshold, False otherwise
    """
    unique_protected = np.unique(p)

    def _continuous_disparate_impact() -> float:
        result = 0
        min_protected = np.min(p)
        max_protected = np.max(p)
        interval = max_protected - min_protected
        step_width = interval * delta
        number_of_steps = int(interval / step_width)
        for i in range(number_of_steps):
            min_value = min_protected + i * step_width
            max_value = min_protected + (i + 1) * step_width
            conditional_probability_in = single_conditional_probability_in_range(y, p, min_value, max_value)
            conditional_probability_out = single_conditional_probability_in_range(y, p, min_value, max_value, negate=True)
            if conditional_probability_in <= EPSILON or conditional_probability_out <= EPSILON:
                pass
            else:
                number_of_sample = np.sum(np.logical_and(p >= min_value, p < max_value))
                ratio = conditional_probability_in / conditional_probability_out
                inverse_ratio = conditional_probability_out / conditional_probability_in
                result += min(ratio, inverse_ratio) * number_of_sample
        return result / len(p)

    if continuous:
        impact = _continuous_disparate_impact()
    else:
        probabilities_a = np.array([np.mean(y[p == x]) for x in unique_protected])
        probabilities_not_a = np.array([np.mean(y[p != x]) for x in unique_protected])
        first_impact = np.nan_to_num(probabilities_a / probabilities_not_a)
        second_impact = np.nan_to_num(probabilities_not_a / probabilities_a)
        number_of_samples = np.array([np.sum(p == x) for x in unique_protected])
        pair_wise_weighted_min = np.min(np.vstack((first_impact, second_impact)), axis=0) * number_of_samples
        impact = np.sum(pair_wise_weighted_min) / len(p)
    fairness.logger.info(f"Disparate impact: {impact:.4f}")
    return impact > threshold if not numeric else impact


def is_equalized_odds(
        p: np.array, y_true: np.array, y_pred: np.array, epsilon: float = EPSILON, continuous: bool = False, numeric: bool = True
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

    def _continuous_equalized_odds() -> float:
        min_protected = np.min(p)
        max_protected = np.max(p)
        interval = max_protected - min_protected
        step_width = interval * DELTA
        number_of_steps = int(interval / step_width)
        result = 0
        for i in range(number_of_steps):
            probabilities_a_0 = np.array([np.mean(y_pred[(p >= min_protected + i * step_width) & (p < min_protected + (i + 1) * step_width) & (y_true == 0)])])
            probabilities_a_1 = np.array([np.mean(y_pred[(p >= min_protected + i * step_width) & (p < min_protected + (i + 1) * step_width) & (y_true == 1)])])
            number_of_samples = np.array([np.sum((p >= min_protected + i * step_width) & (p < min_protected + (i + 1) * step_width) & (y_true == y)) for y in [0, 1]])
            partial = np.abs(np.concatenate([probabilities_a_0 - conditional_prob_zero, probabilities_a_1 - conditional_prob_one]))
            partial = np.nan_to_num(partial)
            partial = np.sum(partial * number_of_samples)
            result += partial
        return result / len(y_true)

    if continuous:
        equalized_odds = _continuous_equalized_odds()
    else:
        probabilities_a_0 = np.array([np.mean(y_pred[(p == x) & (y_true == 0)]) for x in unique_protected])
        probabilities_a_1 = np.array([np.mean(y_pred[(p == x) & (y_true == 1)]) for x in unique_protected])
        number_of_samples = np.array([np.sum(p == x and y_true == y) for x in unique_protected for y in [0, 1]])
        equalized_odds = np.abs(np.concatenate([probabilities_a_0 - conditional_prob_zero, probabilities_a_1 - conditional_prob_one]))
        equalized_odds = np.nan_to_num(equalized_odds)
        equalized_odds = np.sum(equalized_odds * number_of_samples) / np.sum(number_of_samples)
    fairness.logger.info(f"Equalized odds: {equalized_odds:.4f}")
    return equalized_odds < epsilon if not numeric else equalized_odds
