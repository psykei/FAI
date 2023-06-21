from tensorflow.python.framework.ops import disable_eager_execution
from tensorflow.python.keras.losses import binary_crossentropy
from dataset.adult_data_pipeline import AdultLoader
from fairness import enable_logging, logger
from fairness.metric import is_demographic_parity
from fairness.tf_metric import demographic_parity
from utils import create_fully_connected_nn
import numpy as np

EPOCHS = 5000
BATCH_SIZE = 500
NEURONS_PER_LAYER = [100, 50]
VERBOSE = 1

disable_eager_execution()
enable_logging()
loader = AdultLoader()
dataset = loader.load_preprocessed()
train, valid, test = loader.load_preprocessed_split()

idx = 3
model = create_fully_connected_nn(train.shape[1] - 1, 1, NEURONS_PER_LAYER)
x = model.layers[0].input
custom_loss = lambda y_true, y_pred: binary_crossentropy(y_true, y_pred) #+ demographic_parity(idx, x, y_pred)

model.compile(optimizer="adam", loss=custom_loss, metrics=["accuracy"])
model.fit(train.iloc[:, :-1], train.iloc[:, -1], epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(valid.iloc[:, :-1], valid.iloc[:, -1]), verbose=VERBOSE)
loss, accuracy = model.evaluate(test.iloc[:, :-1], test.iloc[:, -1], verbose=VERBOSE)
logger.info(f"Test loss: {loss:.4f}")
logger.info(f"Test accuracy: {accuracy:.4f}")
predictions = np.squeeze(np.round(model.predict(test.iloc[:, :-1])))
logger.info(f"Demographic parity: {is_demographic_parity(test.iloc[:, idx].to_numpy(), predictions)}")
