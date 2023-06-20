import numpy as np
import fairness

EPSILON: float = 1e-2
DISPARATE_IMPACT_THRESHOLD: float = 0.8


def is_demographic_parity(p: np.array, y: np.array, epsilon=EPSILON) -> bool:
    """
    Demographic parity is a measure of fairness that measures if a value of a protected feature impacts the outcome of a
    prediction. In other words, it measures if the outcome is independent of the protected feature.
    :param p: protected feature
    :param y: output
    :param epsilon: threshold for demographic parity
    :return: False if demographic parity is less than epsilon, True otherwise
    """
    result = True
    output_values = np.unique(y)
    for output_value in output_values:
        output_indices = np.where(y == output_value)
        probability = np.sum(y[output_indices]) / len(y)
        protected_feature_values = np.unique(p[output_indices])
        for protected_feature_value in protected_feature_values:
            protected_feature_indices = np.where(p == protected_feature_value)
            predictions = y[protected_feature_indices]
            conditional_probability = np.sum(predictions) / len(predictions)
            prob_abs_diff = abs(probability - conditional_probability)
            if prob_abs_diff > epsilon:
                result = False
                fairness.logger.info(
                    f"Demographic parity violated for output value {output_value} and protected feature value {protected_feature_value} with probability absolute difference {prob_abs_diff}"
                )
                break
            else:
                fairness.logger.info(
                    f"Demographic parity satisfied for output value {output_value} and protected feature value {protected_feature_value} with probability absolute difference {prob_abs_diff}"
                )
    return result


def is_disparate_impact(
    p: np.array, y: np.array, threshold: float = DISPARATE_IMPACT_THRESHOLD
) -> bool:
    """
    Disparate impact is a measure of fairness that measures if a protected feature impacts the outcome of a prediction.
    It has been defined on binary classification problems as the ratio of the probability of a positive outcome given
    the protected feature to the probability of a positive outcome given the complement of the protected feature.
    If the ratio is less than a threshold (usually 0.8), then the prediction is considered to be unfair.
    :param p: protected feature
    :param y: output
    :param threshold: threshold for disparate impact
    :return: True if disparate impact is less than threshold, False otherwise
    """
    result = False
    protected_feature_values, counts = np.unique(p, return_counts=True)
    most_common_protected_feature_value = protected_feature_values[np.argmax(counts)]
    minority_values = protected_feature_values[
        protected_feature_values != most_common_protected_feature_value
    ]
    majority_indices = np.where(p == most_common_protected_feature_value)
    majority_predictions = y[majority_indices]
    majority_probability = np.sum(majority_predictions) / len(majority_predictions)
    for value in minority_values:
        minority_indices = np.where(p == value)
        minority_predictions = y[minority_indices]
        minority_probability = np.sum(minority_predictions) / len(minority_predictions)
        ratio = minority_probability / majority_probability
        if ratio <= threshold:
            result = True
            fairness.logger.info(
                f"There is disparate impact for protected feature value {value} with ratio {ratio}"
            )
            break
    return result


def is_equalized_odds(
    p: np.array, y_true: np.array, y_pred: np.array, epsilon: float = EPSILON
) -> bool:
    """
    Equalized odds is a measure of fairness that measures if the output is independent of the protected feature given
    the label Y.
    :param p: protected feature
    :param y_true: ground truth
    :param y_pred: prediction
    :param epsilon: threshold for equalized odds
    :return: True if equalized odds is satisfied, False otherwise
    """
    result = True
    true_output_values = np.unique(y_true)
    assert (
        len(true_output_values) <= 2
    ), "Equalized odds is only defined for binary classification problems"
    for output_value in true_output_values:
        true_output_indices = np.where(y_true == output_value)
        protected_feature_values = np.unique(p[true_output_indices])
        assert (
            len(protected_feature_values) <= 2
        ), "Equalized odds is only defined for binary protected features"
        conditional_probability = np.sum(y_pred[true_output_indices]) / len(
            y_pred[true_output_indices]
        )
        for protected_feature_value in protected_feature_values:
            protected_feature_indices = np.where(p == protected_feature_value)
            indices = np.intersect1d(true_output_indices, protected_feature_indices)
            predictions = y_pred[indices]
            double_conditional_probability = np.sum(predictions) / len(predictions)
            probability_abs_diff = abs(
                conditional_probability - double_conditional_probability
            )
            if probability_abs_diff > epsilon:
                result = False
                fairness.logger.info(
                    f"Equalized odds violated for output value {output_value} and protected feature value {protected_feature_value} with probability absolute difference {probability_abs_diff}"
                )
                break
            else:
                fairness.logger.info(
                    f"Equalized odds satisfied for output value {output_value} and protected feature value {protected_feature_value} with probability absolute difference {probability_abs_diff}"
                )
    return result
