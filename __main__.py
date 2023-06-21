from sklearn.model_selection import KFold
from tensorflow.python.framework.ops import disable_eager_execution
from tensorflow.python.framework.random_seed import set_seed
from tensorflow.python.keras.losses import binary_crossentropy
from tqdm.keras import TqdmCallback
from dataset.adult_data_pipeline import AdultLoader
from fairness import enable_logging, logger
from fairness.metric import is_demographic_parity, is_disparate_impact, is_equalized_odds
from fairness.tf_metric import demographic_parity, disparate_impact
from utils import create_fully_connected_nn
import numpy as np

SEED = 0
K = 5
EPOCHS = 5000
BATCH_SIZE = 500
NEURONS_PER_LAYER = [100, 50]
VERBOSE = 0
IDX = 3

disable_eager_execution()
enable_logging()
loader = AdultLoader()
dataset = loader.load_preprocessed(all_datasets=True)
train, test = loader.load_preprocessed_split(validation=False, all_datasets=True)
kfold = KFold(n_splits=K, shuffle=True, random_state=SEED)

mean_loss = 0
mean_accuracy = 0
mean_demographic_parity = 0
for fold, (train_idx, test_idx) in enumerate(kfold.split(dataset)):
    logger.info(f"Fold {fold + 1}")
    train = dataset.iloc[train_idx]
    test = dataset.iloc[test_idx]
    set_seed(SEED + fold)
    model = create_fully_connected_nn(train.shape[1] - 1, 1, NEURONS_PER_LAYER)
    x = model.layers[0].input
    custom_loss = lambda y_true, y_pred: binary_crossentropy(y_true, y_pred) # + demographic_parity(IDX, x, y_pred)
    # custom_loss = lambda y_true, y_pred: binary_crossentropy(y_true, y_pred) + disparate_impact(idx, x, y_pred)
    model.compile(optimizer="adam", loss=custom_loss, metrics=["accuracy"])
    model.fit(train.iloc[:, :-1], train.iloc[:, -1], epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=VERBOSE, callbacks=[TqdmCallback(verbose=VERBOSE)])
    loss, accuracy = model.evaluate(test.iloc[:, :-1], test.iloc[:, -1], verbose=VERBOSE)
    logger.info(f"Test loss: {loss:.4f}")
    logger.info(f"Test accuracy: {accuracy:.4f}")
    predictions = np.squeeze(np.round(model.predict(test.iloc[:, :-1])))
    mean_loss += loss
    mean_accuracy += accuracy
    mean_demographic_parity += is_demographic_parity(test.iloc[:, IDX].to_numpy(), predictions, numeric=True)
mean_loss /= K
mean_accuracy /= K
mean_demographic_parity /= K
logger.info(f"Mean test loss: {mean_loss:.4f}")
logger.info(f"Mean test accuracy: {mean_accuracy:.4f}")
logger.info(f"Mean demographic parity: {mean_demographic_parity:.4f}")
