import tensorflow as tf

EPSILON: float = 1e-9
INFINITY: float = 1e9


def single_conditional_probability(predicted: tf.Tensor, protected: tf.Tensor, value: int, equal: bool = True) -> tf.Tensor:
    """
    Calculate the estimated conditioned output distribution of a model.
    The protected attribute can be binary or categorical.

    @param predicted: the predicted labels.
    @param protected: the protected attribute.
    @param value: the value of the protected attribute.
    @param equal: if True, filter rows whose protected attribute is equal to value, otherwise filter rows whose protected
    attribute is not equal to value.
    @return: the conditional probability.
    :param equal:
    """
    mask = tf.cond(
        tf.convert_to_tensor(equal),
        lambda: tf.boolean_mask(predicted, tf.equal(protected, value)),
        lambda: tf.boolean_mask(predicted, tf.not_equal(protected, value))
    )
    return tf.cond(
        tf.equal(tf.size(mask), 0),
        lambda: tf.constant(0.0),
        lambda: tf.math.reduce_mean(mask)
    )


def double_conditional_probability(predicted: tf.Tensor, protected: tf.Tensor, ground_truth: tf.Tensor, first_value: int, second_value: int) -> tf.Tensor:
    """
    Calculate the estimated conditioned output distribution of a model.
    The protected attribute can be binary or categorical.

    @param predicted: the predicted labels.
    @param protected: the protected attribute.
    @param ground_truth: the ground truth.
    @param first_value: the value of the protected attribute.
    @param second_value: the value of the ground truth.
    @return: the conditional probability.
    :param equal:
    """
    mask = tf.boolean_mask(predicted, tf.logical_and(tf.equal(protected, first_value), tf.equal(ground_truth, second_value)))
    return tf.cond(
        tf.equal(tf.size(mask), 0),
        lambda: tf.constant(0.0),
        lambda: tf.math.reduce_mean(mask)
    )


def tf_demographic_parity(index: int, x: tf.Tensor, predicted: tf.Tensor, threshold: float = EPSILON) -> tf.Tensor:
    """
    Calculate the demographic parity of a model.
    The protected attribute can be binary or categorical.

    @param index: the index of the protected attribute.
    @param x: the input data.
    @param predicted: the predicted labels.
    @param threshold: the target threshold for demographic parity.
    @return: the demographic impact error.
    """
    protected = x[:, index]
    unique_protected, _ = tf.unique(protected)
    absolute_probability = tf.math.reduce_mean(predicted)

    def _single_conditional_probability(value: int) -> tf.Tensor:
        return single_conditional_probability(predicted, protected, value)

    probabilities = tf.map_fn(_single_conditional_probability, unique_protected)
    result = tf.reduce_sum(tf.abs(probabilities - absolute_probability))
    return tf.cond(
        tf.less(result, threshold),
        lambda: tf.constant(0.0),
        lambda: result
    )


def tf_disparate_impact(index: int, x: tf.Tensor, predicted: tf.Tensor, threshold: float = 0.8) -> tf.Tensor:
    """
    Calculate the disparate impact of a model.
    The protected attribute is must be binary or categorical.

    @param index: the index of the protected attribute.
    @param x: the input data.
    @param predicted: the predicted labels.
    @param threshold: the target threshold for disparate impact.
    @return: the disparate impact error.
    """
    protected = x[:, index]
    unique_protected, _ = tf.unique(protected)
    masks_a = tf.map_fn(lambda value: single_conditional_probability(predicted, protected, value), unique_protected)
    masks_not_a = tf.map_fn(lambda value: single_conditional_probability(predicted, protected, value, equal=False), unique_protected)
    probabilities_a = tf.map_fn(lambda mask: tf.math.reduce_mean(mask), masks_a)
    probabilities_not_a = tf.map_fn(lambda mask: tf.math.reduce_mean(mask), masks_not_a)
    impacts = tf.math.divide_no_nan(probabilities_a, tf.math.reduce_mean(predicted))
    inverse_impacts = tf.math.divide_no_nan(probabilities_not_a, tf.math.reduce_mean(predicted))
    result = 1 - tf.reduce_min(tf.concat([impacts, inverse_impacts], axis=0))

    return tf.cond(
        tf.less(result, 1.0 - threshold),
        lambda: tf.constant(0.0),
        lambda: result
    )


def tf_equalized_odds(index: int, x: tf.Tensor, y: tf.Tensor, predicted: tf.Tensor,
                      threshold: float = EPSILON) -> tf.Tensor:
    """
    Calculate the equalized odds of a model.
    The protected attribute is must be binary or categorical.

    :param index: the index of the protected attribute.
    :param x: the input data.
    :param y: the ground truth labels.
    :param predicted: the predicted labels.
    :param threshold: the target threshold for equalized odds.
    :return: the equalized odds error.
    """
    protected = x[:, index]
    unique_protected, _ = tf.unique(protected)
    masks_a_0 = tf.map_fn(lambda value: double_conditional_probability(predicted, protected, y[:, 0], value, 0), unique_protected)
    masks_a_1 = tf.map_fn(lambda value: double_conditional_probability(predicted, protected, y[:, 0], value, 1), unique_protected)
    mask_0 = tf.math.reduce_mean(single_conditional_probability(predicted, y[:, 0], 0))
    mask_1 = tf.math.reduce_mean(single_conditional_probability(predicted, y[:, 0], 1))
    differences_0 = tf.map_fn(lambda mask: tf.math.abs(mask - mask_0), masks_a_0)
    differences_1 = tf.map_fn(lambda mask: tf.math.abs(mask - mask_1), masks_a_1)
    result = tf.reduce_sum(differences_0) + tf.reduce_sum(differences_1)
    return tf.cond(
        tf.less(result, threshold),
        lambda: tf.constant(0.0),
        lambda: result
    )
