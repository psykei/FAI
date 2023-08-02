import hashlib
import os
from configuration import *
import tensorflow as tf
from tensorflow.python.compat.v2_compat import disable_v2_behavior
from tensorflow.python.framework.ops import disable_eager_execution
from tensorflow.python.framework.random_seed import set_seed
from tensorflow.python.keras.losses import binary_crossentropy
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tqdm.keras import TqdmCallback
from fairness import enable_logging, logger, disable_file_logging
from fairness.metric import demographic_parity, equalized_odds, disparate_impact
from fairness.tf_metric import continuous_demographic_parity, continuous_disparate_impact, continuous_equalized_odds, \
    discrete_demographic_parity, discrete_equalized_odds, discrete_disparate_impact
from utils import create_fully_connected_nn, Conditions
import numpy as np
from fairness.our import PATH as FAIRNESS_PATH


disable_v2_behavior()
disable_eager_execution()


def cost_combiner(first_cost: tf.Tensor, second_cost: tf.Tensor) -> tf.Tensor:
    # return tf.minimum(first_cost + second_cost, tf.constant(2, dtype=tf.float32) * first_cost)
    return first_cost + second_cost


def compute_experiments_given_fairness_metric(metric: str = None, IDX: int = 0):

    def compute_experiments_given_lambda(l: float = 1.0):
        continuous = True if IDX == 0 else False
        idf = '_'.join([str(x) for x in [SEED, K, EPOCHS, BATCH_SIZE, NEURONS_PER_LAYER, IDX, metric, l]])
        if not ONE_HOT:
            idf += "_" + str(ONE_HOT)
        filename = str(FAIRNESS_PATH) + os.sep + LOG + os.sep + hashlib.md5(str(idf).encode()).hexdigest() + ".txt"
        if not os.path.exists(filename):
            dataset, train, test, kfold = initialize_experiment(filename, metric, IDX, l)
            l_tf = tf.constant(l, dtype=tf.float32)

            mean_accuracy = 0
            mean_demographic_parity = 0
            mean_disparate_impact = 0
            mean_equalized_odds = 0
            for fold, (train_idx, valid_idx) in enumerate(kfold.split(train)):
                logger.info(f"Fold {fold + 1}")
                train = dataset.iloc[train_idx]
                valid = dataset.iloc[valid_idx]
                valid_x, valid_y = valid.drop("income", axis=1), valid["income"]
                np.random.seed(SEED + fold)
                set_seed(SEED + fold)
                model = create_fully_connected_nn(train.shape[1] - 1, 1, NEURONS_PER_LAYER)
                x = model.layers[0].input

                if metric == "demographic_parity":
                    custom_loss = lambda y_true, y_pred: cost_combiner(binary_crossentropy(y_true, y_pred), l_tf * continuous_demographic_parity(IDX, x, y_pred))
                elif metric == "disparate_impact":
                    custom_loss = lambda y_true, y_pred: cost_combiner(binary_crossentropy(y_true, y_pred), l_tf * continuous_disparate_impact(IDX, x, y_pred))
                elif metric == "equalized_odds":
                    custom_loss = lambda y_true, y_pred: cost_combiner(binary_crossentropy(y_true, y_pred), l_tf * continuous_equalized_odds(IDX, x, y_true, y_pred))
                else:
                    custom_loss = binary_crossentropy

                if continuous:
                    def tf_demographic_parity(y_true, y_pred):
                        return continuous_demographic_parity(IDX, x, y_pred)

                    def tf_disparate_impact(y_true, y_pred):
                        return continuous_disparate_impact(IDX, x, y_pred)

                    def tf_equalized_odds(y_true, y_pred):
                        return continuous_equalized_odds(IDX, x, y_true, y_pred)
                else:
                    def tf_demographic_parity(y_true, y_pred):
                        return discrete_demographic_parity(IDX, x, y_pred)

                    def tf_disparate_impact(y_true, y_pred):
                        return discrete_disparate_impact(IDX, x, y_pred)

                    def tf_equalized_odds(y_true, y_pred):
                        return discrete_equalized_odds(IDX, x, y_true, y_pred)

                fairness_metric = tf_demographic_parity
                if metric == "demographic_parity":
                    fairness_metric = tf_demographic_parity
                elif metric == "disparate_impact":
                    fairness_metric = tf_disparate_impact
                elif metric == "equalized_odds":
                    fairness_metric = tf_equalized_odds

                metrics = [
                    "accuracy",
                    fairness_metric
                ]
                target_fairness_metric = TARGET_FAIRNESS_METRIC if metric == "demographic_parity" else TARGET_DISPARATE_IMPACT
                early_stopping = Conditions(fairness_metric_name=metric, target_accuracy=TARGET_ACCURACY, target_fairness_metric=target_fairness_metric)
                model.compile(optimizer=Adam(), loss=custom_loss, metrics=metrics)
                model.fit(train.iloc[:, :-1], train.iloc[:, -1], epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=VERBOSE,
                          validation_data=(valid_x, valid_y),callbacks=[TqdmCallback(verbose=VERBOSE), early_stopping])
                # loss, accuracy, _, = model.evaluate(test.iloc[:, :-1], test.iloc[:, -1], verbose=VERBOSE)
                # logger.info(f"Test loss: {loss:.4f}")
                predictions = np.squeeze(np.round(model.predict(test.iloc[:, :-1])))
                accuracy = np.mean(predictions == test.iloc[:, -1].to_numpy())
                logger.info(f"Test accuracy: {accuracy:.4f}")
                # mean_loss += loss
                mean_accuracy += accuracy
                mean_demographic_parity += demographic_parity(test.iloc[:, IDX].to_numpy(), predictions, numeric=True, continuous=continuous)
                mean_disparate_impact += disparate_impact(test.iloc[:, IDX].to_numpy(), predictions, numeric=True, continuous=continuous)
                mean_equalized_odds += equalized_odds(test.iloc[:, IDX].to_numpy(), test.iloc[:, -1].to_numpy(), predictions, numeric=True, continuous=continuous)
                # sleep(60*2)
            # mean_loss /= K
            mean_accuracy /= K
            mean_demographic_parity /= K
            mean_disparate_impact /= K
            mean_equalized_odds /= K
            # logger.info(f"Mean test loss: {mean_loss:.4f}")
            logger.info(f"Mean test accuracy: {mean_accuracy:.4f}")
            logger.info(f"Mean demographic parity: {mean_demographic_parity:.4f}")
            logger.info(f"Mean disparate impact: {mean_disparate_impact:.4f}")
            logger.info(f"Mean equalized odds: {mean_equalized_odds:.4f}")
            disable_file_logging()
        else:
            enable_logging()
            logger.info("Experiment already executed! It is available in the file: " + filename + ".")

    if metric is None:
        compute_experiments_given_lambda()
    else:
        for l in our_lambdas(IDX, metric):
            compute_experiments_given_lambda(l)


# compute_experiments_given_fairness_metric()
for metric in CUSTOM_METRICS:
    for idx in IDXS:
        compute_experiments_given_fairness_metric(metric, idx)
