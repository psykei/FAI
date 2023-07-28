from typing import Iterable

import torch
from tensorflow.python.keras import Input
from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Model
from fairness import logger


def create_fully_connected_nn(input_size: int, output_size: int, neurons_per_hidden_layer: Iterable[int],
                              activation_function: str = 'relu', latest_activation_function: str = 'sigmoid') -> Model:
    input_layer = Input(shape=(input_size,))
    x = input_layer
    for neurons in neurons_per_hidden_layer:
        x = Dense(neurons, activation=activation_function)(x)
    output_layer = Dense(output_size, activation=latest_activation_function)(x)
    return Model(inputs=input_layer, outputs=output_layer)


class Conditions(Callback):

    def __init__(self,
                 patience: int = 100,
                 target_accuracy: float = 0.85,
                 fairness_metric_name: str = None,
                 target_fairness_metric: float = 0.01):
        super().__init__()
        self.patience = patience
        self.long_patience = patience * 2
        self.target_accuracy = target_accuracy
        self.fairness_metric_name = fairness_metric_name
        self.target_fairness_metric = target_fairness_metric
        self.best_accuracy = 0
        self.best_fairness_metric = 0 if fairness_metric_name == "disparate_impact" else 1000
        self.wait = 0
        self.stopped_epoch = 0

    def on_train_begin(self, logs=None):
        self.best_accuracy = 0
        self.best_fairness_metric = 0 if self.fairness_metric_name == "disparate_impact" else 1000
        self.wait = 0
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch, logs=None):

        def end(condition: int = 0):
            self.stopped_epoch = epoch
            self.model.stop_training = True
            logger.info(f"Early stopping at epoch {epoch + 1} because of condition {condition}")

        accuracy = logs.get('val_acc')
        target_metric_condition = True
        best_metric_condition = True
        fairness_metric = 0
        if self.fairness_metric_name is not None:
            fairness_metric = logs.get("val_tf_" + self.fairness_metric_name)
            if self.fairness_metric_name == "disparate_impact":
                fairness_metric = 1 - fairness_metric
                best_metric_condition = fairness_metric > self.best_fairness_metric
                target_metric_condition = fairness_metric >= self.target_fairness_metric
            else:
                best_metric_condition = self.best_fairness_metric > fairness_metric
                target_metric_condition = self.target_fairness_metric >= fairness_metric

        # Update if both accuracy and fairness metric improve
        if accuracy > self.best_accuracy and best_metric_condition:
            self.best_accuracy = max(self.best_accuracy, accuracy)
            if self.fairness_metric_name is not None:
                if self.fairness_metric_name == "disparate_impact":
                    self.best_fairness_metric = max(self.best_fairness_metric, fairness_metric)
                else:
                    self.best_fairness_metric = min(self.best_fairness_metric, fairness_metric)
            self.wait = 0

        # First condition: accuracy and fairness metric are above the targets
        if accuracy >= self.target_accuracy and target_metric_condition:
            end(1)
        # Second condition: no improvement for n epochs
        else:
            self.wait += 1
            if self.wait >= self.patience:
                end(2)

        # print(f"Epoch {epoch+1}: val accuracy = {accuracy}, val fairness metric = {fairness_metric}, best val accuracy = {self.best_accuracy}, best val fairness metric = {self.best_fairness_metric}, wait = {self.wait}")

        if epoch + 1 == self.params['epochs']:
            end()


class PyTorchConditions:
    def __init__(self,
                 model: torch.nn.Module,
                 patience: int = 100,
                 target_accuracy: float = 0.9,
                 fairness_metric_name: str = None,
                 target_fairness_metric: float = 0.01,
                 max_epochs: int = 5000):
        super().__init__()
        self.model = model
        self.patience = patience
        self.long_patience = patience * 2
        self.target_accuracy = target_accuracy
        self.fairness_metric_name = fairness_metric_name
        self.target_fairness_metric = target_fairness_metric
        self.best_accuracy = 0.0
        self.best_fairness_metric = 0.0 if fairness_metric_name == "disparate_impact" else 1000
        self.wait = 0
        self.stopped_epoch = 0
        self.max_epochs = max_epochs

    def early_stop(self, epoch: int, accuracy: float, fairness_metric: float):
        def end(condition: int = 0):
            self.stopped_epoch = epoch
            logger.info(f"Early stopping at epoch {epoch + 1} because of condition {condition}")
            return True

        target_metric_condition = True
        best_metric_condition = True
        if self.fairness_metric_name is not None:
            if self.fairness_metric_name == "disparate_impact":
                fairness_metric = 1 - fairness_metric
                best_metric_condition = fairness_metric > self.best_fairness_metric
                target_metric_condition = fairness_metric >= self.target_fairness_metric

            else:
                best_metric_condition = self.best_fairness_metric > fairness_metric
                target_metric_condition = self.target_fairness_metric >= fairness_metric

        # Update if both accuracy and fairness metric improve
        if accuracy > self.best_accuracy and best_metric_condition:
            self.best_accuracy = max(self.best_accuracy, accuracy)
            if self.fairness_metric_name is not None:
                if self.fairness_metric_name == "disparate_impact":
                    self.best_fairness_metric = max(self.best_fairness_metric, fairness_metric)
                else:
                    self.best_fairness_metric = min(self.best_fairness_metric, fairness_metric)
            self.wait = 0

        # First condition: accuracy and fairness metric are above the targets
        if accuracy >= self.target_accuracy and target_metric_condition:
            return end(1)
        # Second condition: no improvement for n epochs
        else:
            self.wait += 1
            if self.wait >= self.patience:
                return end(2)

        if epoch + 1 == self.max_epochs:
            return end()

        return False
