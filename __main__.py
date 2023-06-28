import hashlib
import os
import tensorflow as tf
from sklearn.model_selection import KFold
from tensorflow.python.compat.v2_compat import disable_v2_behavior
from tensorflow.python.framework.ops import disable_eager_execution
from tensorflow.python.framework.random_seed import set_seed
from tensorflow.python.keras.losses import binary_crossentropy
from tqdm.keras import TqdmCallback
from dataset.adult_data_pipeline import AdultLoader
from fairness import enable_logging, enable_file_logging, logger, disable_file_logging
from fairness.metric import is_demographic_parity, is_disparate_impact, is_equalized_odds
from fairness.tf_metric import demographic_parity, disparate_impact, equalized_odds
from utils import create_fully_connected_nn
import numpy as np
from fairness import PATH as FAIRNESS_PATH


LOG = "log"
SEED = 0
K = 5
EPOCHS = 5000
BATCH_SIZE = 500
NEURONS_PER_LAYER = [100, 50]
VERBOSE = 0
IDX = 3
CUSTOM_METRICS = ["demographic_parity", "disparate_impact", "equalized_odds"]
LAMBDAS = [(10-i)/10 for i in range(0, 10)]
disable_v2_behavior()
disable_eager_execution()

for CUSTOM_METRIC in CUSTOM_METRICS:
    for LAMBDA in LAMBDAS:
        if LAMBDA in (1.0, None):
            LAMBDA = 1
        L = tf.constant(LAMBDA, dtype=tf.float32)
        id = str(SEED) + "_" + str(K) + "_" + str(EPOCHS) + "_" + str(BATCH_SIZE) + "_" + str(NEURONS_PER_LAYER) + "_" + str(
            IDX) + "_" + str(CUSTOM_METRIC)
        if LAMBDA != 1:
            id += "_" + str(LAMBDA)
        filename = str(FAIRNESS_PATH) + os.sep + LOG + os.sep + hashlib.md5(str(id).encode()).hexdigest() + ".txt"
        if not os.path.exists(filename):
            enable_file_logging(filename)
            logger.info(
                f"Parameters:"
                f"\n\tSEED={SEED}"
                f"\n\tK={K}"
                f"\n\tEPOCHS={EPOCHS}"
                f"\n\tBATCH_SIZE={BATCH_SIZE}"
                f"\n\tNEURONS_PER_LAYER={NEURONS_PER_LAYER}"
                f"\n\tIDX={IDX} (protected feature index)"
                f"\n\tCUSTOM_METRIC={CUSTOM_METRIC}"
                f"\n\tLAMBDA={1 if LAMBDA is None else LAMBDA}")

            loader = AdultLoader()
            dataset = loader.load_preprocessed(all_datasets=True)
            kfold = KFold(n_splits=K, shuffle=True, random_state=SEED)

            mean_loss = 0
            mean_accuracy = 0
            mean_demographic_parity = 0
            mean_disparate_impact = 0
            mean_equalized_odds = 0
            for fold, (train_idx, test_idx) in enumerate(kfold.split(dataset)):
                logger.info(f"Fold {fold + 1}")
                train = dataset.iloc[train_idx]
                test = dataset.iloc[test_idx]
                np.random.seed(SEED + fold)
                set_seed(SEED + fold)
                model = create_fully_connected_nn(train.shape[1] - 1, 1, NEURONS_PER_LAYER)
                x = model.layers[0].input

                if CUSTOM_METRIC == "demographic_parity":
                    custom_loss = lambda y_true, y_pred: binary_crossentropy(y_true, y_pred) + L * demographic_parity(IDX, x, y_pred)
                elif CUSTOM_METRIC == "disparate_impact":
                    custom_loss = lambda y_true, y_pred: binary_crossentropy(y_true, y_pred) + L * disparate_impact(IDX, x, y_pred)
                elif CUSTOM_METRIC == "equalized_odds":
                    custom_loss = lambda y_true, y_pred: binary_crossentropy(y_true, y_pred) + L * equalized_odds(IDX, x, y_true, y_pred)
                else:
                    custom_loss = binary_crossentropy

                @tf.function
                def compliant_demo_parity(y_true, y_pred):
                    return demographic_parity(IDX, x, y_pred)

                @tf.function
                def compliant_disparate_impact(y_true, y_pred):
                    return disparate_impact(IDX, x, y_pred)

                @tf.function
                def compliant_equalized_odds(y_true, y_pred):
                    return equalized_odds(IDX, x, y_true, y_pred)

                @tf.function
                def how_many_ones(y_true, y_pred):
                    return tf.reduce_sum(y_pred)/tf.cast(len(y_pred), tf.float32)

                metrics = [
                    "accuracy",
                    compliant_demo_parity,
                    compliant_disparate_impact,
                    compliant_equalized_odds,
                    how_many_ones
                ]
                model.compile(optimizer="adam", loss=custom_loss, metrics=metrics)
                model.fit(train.iloc[:, :-1], train.iloc[:, -1], epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=VERBOSE,
                          callbacks=[TqdmCallback(verbose=VERBOSE)])
                loss, accuracy, _, _, _, _ = model.evaluate(test.iloc[:, :-1], test.iloc[:, -1], verbose=VERBOSE)
                logger.info(f"Test loss: {loss:.4f}")
                logger.info(f"Test accuracy: {accuracy:.4f}")
                predictions = np.squeeze(np.round(model.predict(test.iloc[:, :-1])))
                mean_loss += loss
                mean_accuracy += accuracy
                mean_demographic_parity += is_demographic_parity(test.iloc[:, IDX].to_numpy(), predictions, numeric=True)
                mean_disparate_impact += is_disparate_impact(test.iloc[:, IDX].to_numpy(), predictions, numeric=True)
                mean_equalized_odds += is_equalized_odds(test.iloc[:, IDX].to_numpy(), test.iloc[:, -1].to_numpy(), predictions, numeric=True)
            mean_loss /= K
            mean_accuracy /= K
            mean_demographic_parity /= K
            mean_disparate_impact /= K
            mean_equalized_odds /= K
            logger.info(f"Mean test loss: {mean_loss:.4f}")
            logger.info(f"Mean test accuracy: {mean_accuracy:.4f}")
            logger.info(f"Mean demographic parity: {mean_demographic_parity:.4f}")
            logger.info(f"Mean disparate impact: {mean_disparate_impact:.4f}")
            logger.info(f"Mean equalized odds: {mean_equalized_odds:.4f}")
            disable_file_logging()
        else:
            enable_logging()
            logger.info("Experiment already executed! It is available in the file: " + filename + ".")
