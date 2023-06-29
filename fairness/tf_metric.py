import tensorflow as tf

EPSILON: float = 1e-9
INFINITY: float = 1e9


# @tf.function
def single_conditional_probability(predicted: tf.Tensor, protected: tf.Tensor, value: int) -> tf.Tensor:
    """
    Calculate the estimated conditioned output distribution of a model.
    The protected attribute can be binary or categorical.

    @param predicted: the predicted labels.
    @param protected: the protected attribute.
    @param value: the value of the protected attribute.
    @return: the conditional probability.
    """
    mask = tf.boolean_mask(predicted, tf.equal(protected, value))
    return tf.cond(
        tf.equal(tf.size(mask), 0),
        lambda: tf.constant(0.0),
        lambda: tf.math.reduce_mean(mask)
    )


# @tf.function
def _demographic_parity(index: int, x: tf.Tensor, predicted: tf.Tensor, threshold: float = EPSILON) -> tf.Tensor:
    """
    Calculate the demographic parity of a model.
    The protected attribute can be binary or categorical.

    @param index: the index of the protected attribute.
    @param x: the input data.
    @param predicted: the predicted labels.
    @param threshold: the target threshold for demographic parity.
    @return: the demographic impact error.
    """
    protected = tf.boolean_mask(x[:, index], tf.equal(x[:, index], 0))
    unique_protected, _ = tf.unique(protected)
    # unique_protected = tf.cast(unique_protected, tf.float32)
    # tf.debugging.assert_integer(unique_protected)
    absolute_probability = tf.math.reduce_mean(predicted)

    # def _single_conditional_probability(value: int) -> tf.Tensor:
    #     return single_conditional_probability(predicted, protected, value)
    #
    # probabilities = tf.map_fn(_single_conditional_probability, unique_protected)
    mask = tf.boolean_mask(predicted, tf.equal(protected, 0))
    x0 = tf.cond(
        tf.equal(tf.size(mask), 0),
        lambda: tf.constant(0.0),
        lambda: tf.math.reduce_mean(mask)
    )
    mask = tf.boolean_mask(predicted, tf.equal(protected, 1))
    x1 = tf.cond(
        tf.equal(tf.size(mask), 0),
        lambda: tf.constant(0.0),
        lambda: tf.math.reduce_mean(mask)
    )
    result = tf.reduce_sum([
        tf.abs(x0 - absolute_probability),
        tf.abs(x1 - absolute_probability)
    ])
    return tf.cond(
        tf.less(result, threshold),
        lambda: tf.constant(0.0),
        lambda: result
    )


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
    absolute_probability = tf.math.reduce_mean(predicted)
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
    result = tf.abs(tf.math.reduce_mean(conditional_prob_zero) - absolute_probability) \
        + tf.abs(tf.math.reduce_mean(conditional_prob_one) - absolute_probability)
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
def equalized_odds(index: int, x: tf.Tensor, y: tf.Tensor, predicted: tf.Tensor,
                   threshold: float = EPSILON) -> tf.Tensor:
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
