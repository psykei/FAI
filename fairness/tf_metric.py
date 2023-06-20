import tensorflow as tf


@tf.function
def demographic_parity(index: int, x: tf.Tensor, predicted: tf.Tensor) -> tf.Tensor:
    """
    Calculate the demographic parity of a model.
    The protected attribute is must be binary.

    @param index: the index of the protected attribute.
    @param x: the input data.
    @param predicted: the predicted labels.
    @return: the demographic impact error.
    """
    return tf.abs(tf.math.reduce_mean(
        tf.boolean_mask(predicted, tf.equal(x[:, index], 0))
    ) - tf.math.reduce_mean(predicted)) + tf.abs(tf.math.reduce_mean(
        tf.boolean_mask(predicted, tf.equal(x[:, index], 1))
    ) - tf.math.reduce_mean(predicted))


@tf.function
def disparate_impact(index: int, x: tf.Tensor, predicted: tf.Tensor) -> tf.Tensor:
    """
    Calculate the disparate impact of a model.
    The protected attribute is must be binary.

    @param index: the index of the protected attribute.
    @param x: the input data.
    @param predicted: the predicted labels.
    @return: the disparate impact error.
    """
    first_impact = tf.cond(
        tf.math.reduce_mean(tf.boolean_mask(predicted, tf.equal(x[:, index], 0))) == 0,
        lambda: tf.constant(0.0),
        lambda: tf.cond(
            tf.math.reduce_mean(tf.boolean_mask(predicted, tf.equal(x[:, index], 1))) == 0,
            lambda: tf.constant(0.0),
            lambda: tf.math.reduce_mean(
                tf.boolean_mask(predicted, tf.equal(x[:, index], 0))
            ) / tf.math.reduce_mean(
                tf.boolean_mask(predicted, tf.equal(x[:, index], 1))
            )
        )
    )
    return tf.cond(
        first_impact == 0,
        lambda: tf.constant(0.0),
        lambda: 1 - tf.math.minimum(first_impact, 1 / first_impact)
    )


@tf.function
def equalized_odds(index: int, x: tf.Tensor, y: tf.Tensor, predicted: tf.Tensor) -> tf.Tensor:
    """
    Calculate the equalized odds of a model.
    The protected attribute is must be binary.

    :param index: the index of the protected attribute.
    :param x: the input data.
    :param y: the ground truth labels.
    :param predicted: the predicted labels.
    :return: the equalized odds error.
    """
    conditional_prob_zero = tf.math.reduce_mean(tf.boolean_mask(predicted, tf.equal(y, 0)))
    conditional_prob_one = tf.math.reduce_mean(tf.boolean_mask(predicted, tf.equal(y, 1)))
    double_conditional_prob_zero_zero = tf.math.reduce_mean(tf.boolean_mask(predicted, tf.equal(x[:, index], 0) and tf.equal(y, 0)))
    double_conditional_prob_zero_one = tf.cond(tf.math.reduce_mean(tf.boolean_mask(predicted, tf.equal(x[:, index], 0) and tf.equal(y, 1))))
    double_conditional_prob_one_zero = tf.math.reduce_mean(tf.boolean_mask(predicted, tf.equal(x[:, index], 1) and tf.equal(y, 0)))
    double_conditional_prob_one_one = tf.cond(tf.math.reduce_mean(tf.boolean_mask(predicted, tf.equal(x[:, index], 1) and tf.equal(y, 1))))

    return tf.abs(double_conditional_prob_zero_zero - conditional_prob_zero) + tf.abs(
        double_conditional_prob_zero_one - conditional_prob_zero) + tf.abs(
        double_conditional_prob_one_zero - conditional_prob_one) + tf.abs(
        double_conditional_prob_one_one - conditional_prob_one)
