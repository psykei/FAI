import tensorflow as tf


EPSILON: float = 1e-9
INFINITY: float = 1e9


@tf.function
def demographic_parity(index: int, x: tf.Tensor, predicted: tf.Tensor, threshold: float = EPSILON) -> tf.Tensor:
    """
    Calculate the demographic parity of a model.
    The protected attribute is must be binary.

    @param index: the index of the protected attribute.
    @param x: the input data.
    @param predicted: the predicted labels.
    @param threshold: the target threshold for demographic parity.
    @return: the demographic impact error.
    """
    conditional_prob_zero = tf.cond(
        tf.equal(tf.size(tf.boolean_mask(predicted, tf.equal(x[:, index], 0))), 0),
        lambda: tf.constant(0.0),
        lambda: tf.math.reduce_mean(tf.boolean_mask(predicted, tf.equal(x[:, index], 0)))
    )
    conditional_prob_one = tf.cond(
        tf.equal(tf.size(tf.boolean_mask(predicted, tf.equal(x[:, index], 1))), 0),
        lambda: tf.constant(0.0),
        lambda: tf.math.reduce_mean(tf.boolean_mask(predicted, tf.equal(x[:, index], 1)))
    )
    result = tf.abs(tf.math.reduce_mean(conditional_prob_zero) - tf.math.reduce_mean(predicted)) \
        + tf.abs(tf.math.reduce_mean(conditional_prob_one) - tf.math.reduce_mean(predicted))
    return tf.cond(
        tf.less(result, threshold),
        lambda: tf.constant(0.0),
        lambda: result
    )


@tf.function
def disparate_impact(index: int, x: tf.Tensor, predicted: tf.Tensor, threshold: float = 0.8) -> tf.Tensor:
    """
    Calculate the disparate impact of a model.
    The protected attribute is must be binary.

    @param index: the index of the protected attribute.
    @param x: the input data.
    @param predicted: the predicted labels.
    @param threshold: the target threshold for disparate impact.
    @return: the disparate impact error.
    """
    mask_zero = tf.boolean_mask(predicted, tf.equal(x[:, index], 0))
    conditional_prob_zero = tf.cond(
        tf.equal(tf.size(mask_zero), 0),
        lambda: tf.constant(0.0),
        lambda: tf.math.reduce_mean(mask_zero)
    )
    mask_one = tf.boolean_mask(predicted, tf.equal(x[:, index], 1))
    conditional_prob_one = tf.cond(
        tf.equal(tf.size(mask_one), 0),
        lambda: tf.constant(0.0),
        lambda: tf.math.reduce_mean(mask_one)
    )
    first_impact = tf.cond(
        tf.less(conditional_prob_one, EPSILON),
        lambda: tf.constant(INFINITY),
        lambda: conditional_prob_zero / conditional_prob_one
    )
    result = tf.cond(
        tf.less(first_impact, EPSILON),
        lambda: tf.constant(1.0),
        lambda: 1.0 - tf.math.minimum(first_impact, 1 / first_impact)
    )
    return tf.cond(
        tf.less(result, 1.0 - threshold),
        lambda: tf.constant(0.0),
        lambda: result
    )


@tf.function
def equalized_odds(index: int, x: tf.Tensor, y: tf.Tensor, predicted: tf.Tensor, threshold: float = EPSILON) -> tf.Tensor:
    """
    Calculate the equalized odds of a model.
    The protected attribute is must be binary.

    :param index: the index of the protected attribute.
    :param x: the input data.
    :param y: the ground truth labels.
    :param predicted: the predicted labels.
    :param threshold: the target threshold for equalized odds.
    :return: the equalized odds error.
    """
    mask = tf.boolean_mask(predicted, tf.equal(y, 0))
    conditional_prob_zero = tf.cond(
        tf.equal(tf.size(mask), 0),
        lambda: tf.constant(0.0),
        lambda: tf.math.reduce_mean(mask)
    )
    mask = tf.boolean_mask(predicted, tf.equal(y, 1))
    conditional_prob_one = tf.cond(
        tf.equal(tf.size(mask), 0),
        lambda: tf.constant(0.0),
        lambda: tf.math.reduce_mean(mask)
    )
    mask = tf.boolean_mask(predicted, tf.logical_and(tf.equal(x[:, index], 0), tf.equal(y[:, 0], 0)))
    double_conditional_prob_zero_zero = tf.cond(
        tf.equal(tf.size(mask), 0),
        lambda: tf.constant(0.0),
        lambda: tf.math.reduce_mean(mask)
    )
    mask = tf.boolean_mask(predicted, tf.logical_and(tf.equal(x[:, index], 0), tf.equal(y[:, 0], 1)))
    double_conditional_prob_zero_one = tf.cond(
        tf.equal(tf.size(mask), 0),
        lambda: tf.constant(0.0),
        lambda: tf.math.reduce_mean(mask)
    )
    mask = tf.boolean_mask(predicted, tf.logical_and(tf.equal(x[:, index], 1), tf.equal(y[:, 0], 0)))
    double_conditional_prob_one_zero = tf.cond(
        tf.equal(tf.size(mask), 0),
        lambda: tf.constant(0.0),
        lambda: tf.math.reduce_mean(mask)
    )
    mask = tf.boolean_mask(predicted, tf.logical_and(tf.equal(x[:, index], 1), tf.equal(y[:, 0], 1)))
    double_conditional_prob_one_one = tf.cond(
        tf.equal(tf.size(mask), 0),
        lambda: tf.constant(0.0),
        lambda: tf.math.reduce_mean(mask)
    )
    result = tf.reduce_sum([
        tf.abs(double_conditional_prob_zero_zero - conditional_prob_zero),
        tf.abs(double_conditional_prob_zero_one - conditional_prob_zero),
        tf.abs(double_conditional_prob_one_zero - conditional_prob_one),
        tf.abs(double_conditional_prob_one_one - conditional_prob_one)
    ])
    return tf.cond(
        tf.less(result, threshold),
        lambda: tf.constant(0.0),
        lambda: result
    )
