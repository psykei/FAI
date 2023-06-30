from typing import Iterable
from tensorflow.python.keras import Input
from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Model
from fairness import logger


def create_fully_connected_nn(input_size: int, output_size: int, neurons_per_hidden_layer: Iterable[int], activation_function: str = 'relu', latest_activation_function: str = 'sigmoid') -> Model:
    input_layer = Input(shape=(input_size,))
    x = input_layer
    for neurons in neurons_per_hidden_layer:
        x = Dense(neurons, activation=activation_function)(x)
    output_layer = Dense(output_size, activation=latest_activation_function)(x)
    return Model(inputs=input_layer, outputs=output_layer)


class Conditions(Callback):

    def __init__(self,
                 patience: int = 100,
                 target_accuracy: float = 0.83,
                 fairness_metric_name: str = "demographic_parity",
                 target_fairness_metric: float = 0.01):
        super().__init__()
        self.patience = patience
        self.long_patience = patience * 5
        self.target_accuracy = target_accuracy
        self.fairness_metric_name = fairness_metric_name
        self.target_fairness_metric = target_fairness_metric
        self.best_weights = None
        self.best_accuracy = 0
        self.best_fairness_metric = 0 if fairness_metric_name == "disparate_impact" else 1000
        self.wait = 0
        self.long_wait = 0
        self.stopped_epoch = 0

    def on_train_begin(self, logs=None):
        self.best_weights = self.model.get_weights()
        self.best_accuracy = 0
        self.best_fairness_metric = 0 if self.fairness_metric_name == "disparate_impact" else 1000
        self.wait = 0
        self.long_wait = 0
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch, logs=None):

        def end(condition: int = 0):
            self.stopped_epoch = epoch
            self.model.stop_training = True
            logger.info(f"Early stopping at epoch {epoch + 1} because of condition {condition}")

        accuracy = logs.get('acc')
        fairness_metric = logs.get(self.fairness_metric_name)
        if self.fairness_metric_name == "disparate_impact":
            fairness_metric = 1 - fairness_metric
            best_metric_condition = fairness_metric > self.best_fairness_metric
            target_metric_condition = fairness_metric >= self.target_fairness_metric

        else:
            best_metric_condition = self.best_fairness_metric > fairness_metric
            target_metric_condition = self.target_fairness_metric >= fairness_metric

        # Update all if accuracy improves
        if accuracy > self.best_accuracy:
            self.best_weights = self.model.get_weights()
            self.best_accuracy = max(self.best_accuracy, accuracy)
            if self.fairness_metric_name == "disparate_impact":
                self.best_fairness_metric = max(self.best_fairness_metric, fairness_metric)
            else:
                self.best_fairness_metric = min(self.best_fairness_metric, fairness_metric)
            self.wait = 0
            self.long_wait = 0

        # First condition: accuracy and fairness metric are above the targets
        if accuracy >= self.target_accuracy and target_metric_condition:
            end(1)
        # # Second condition: accuracy is above the target but fairness metric is not improving anymore
        # elif accuracy > self.target_accuracy and not best_metric_condition:
        #     self.wait += 1
        #     if self.wait >= self.patience:
        #         end(2)
        # Second condition: accuracy and fairness metric does not improve anymore
        elif accuracy < self.best_accuracy and not best_metric_condition:
            self.long_wait += 1
            if self.wait >= self.long_patience:
                end(2)

        if epoch+1 == self.params['epochs']:
            end()
