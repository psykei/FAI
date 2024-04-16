import math
from logging import Logger
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tensorflow.python.keras import Input
from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Model
from torch import nn

from experiments._logging import INDENT, LOG_FLOAT_PRECISION
from fairness.metric import demographic_parity, equalized_odds, disparate_impact

PATH = Path(__file__).parents[0]
CACHE_DIR_NAME = 'cache'
CACHE_PATH = PATH / CACHE_DIR_NAME
DI_METRIC_NAME = "disparate_impact"
METRIC_LIST_NAMES = [
    'accuracy',
    'precision',
    'recall',
    'f1',
    'auc',
    'demographic_parity',
    'disparate_impact'
    'equalized_odds'
]


def get_feature_data_type(dataset_name: str, index: int) -> str:
    if dataset_name == "adult":
        if index in [7, 8]:
            return "discrete"
        elif index == 0:
            return "continuous"
        else:
            return "unknown"
    elif dataset_name == "compas":
        return "unknown"


class TensorflowConditions(Callback):

    def __init__(self, patience: int, fairness_metric_name: str):
        super().__init__()
        self.patience = patience
        self.fairness_metric_name = fairness_metric_name
        self.best_accuracy = 0
        self.best_fairness_metric = 0 if fairness_metric_name == DI_METRIC_NAME else math.inf
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None

    def on_train_begin(self, logs=None):
        self.best_accuracy = 0
        self.best_fairness_metric = 0 if self.fairness_metric_name == DI_METRIC_NAME else math.inf
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = self.model.get_weights()

    def on_epoch_end(self, epoch, logs=None):

        def end():
            self.stopped_epoch = epoch
            self.model.stop_training = True
            self.model.set_weights(self.best_weights)

        accuracy = logs.get('val_acc')
        fairness_metric = logs.get("val_tf_" + self.fairness_metric_name)
        if self.fairness_metric_name == DI_METRIC_NAME:
            fairness_metric = 1 - fairness_metric
            metric_condition = fairness_metric > self.best_fairness_metric
        else:
            metric_condition = self.best_fairness_metric > fairness_metric

        # First condition: reached the maximum amount of epochs
        if epoch + 1 == self.params['epochs']:
            end()

        # Second condition: accuracy and fairness metric does not improve for patience epochs
        if accuracy > self.best_accuracy and metric_condition:
            self.best_accuracy = max(self.best_accuracy, accuracy)
            self.best_fairness_metric = max(self.best_fairness_metric, fairness_metric)
            self.wait = 0
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                end()


class PyTorchConditions:
    def __init__(self, model: torch.nn.Module, patience: int, fairness_metric_name: str, max_epochs: int):
        super().__init__()
        self.model = model
        self.patience = patience
        self.fairness_metric_name = fairness_metric_name
        self.best_accuracy = 0.0
        self.best_fairness_metric = 0.0 if fairness_metric_name == DI_METRIC_NAME else math.inf
        self.wait = 0
        self.stopped_epoch = 0
        self.max_epochs = max_epochs
        self.best_weights = None

    def early_stop(self, epoch: int, accuracy: float, fairness_metric: float):
        def end():
            self.stopped_epoch = epoch
            self.model.load_state_dict(self.best_weights)
            return True

        if self.fairness_metric_name == DI_METRIC_NAME:
            fairness_metric = 1 - fairness_metric
            metric_condition = fairness_metric > self.best_fairness_metric
        else:
            metric_condition = self.best_fairness_metric > fairness_metric

        # First condition: reached the maximum amount of epochs
        if epoch + 1 == self.max_epochs:
            return end()

        # Second condition: accuracy and fairness metric does not improve for patience epochs
        if accuracy > self.best_accuracy and metric_condition:
            self.best_accuracy = max(self.best_accuracy, accuracy)
            self.best_fairness_metric = max(self.best_fairness_metric, fairness_metric)
            self.wait = 0
            self.best_weights = self.model.state_dict()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                return end()

        return False


class PytorchNN(nn.Module):
    def __init__(self, n_layers, n_inputs, n_hidden_units):
        super(PytorchNN, self).__init__()
        layers = []

        if n_layers == 1:  # Logistic Regression
            layers.append(nn.Linear(n_inputs, 1))
            layers.append(nn.Sigmoid())
        else:
            layers.append(nn.Linear(n_inputs, n_hidden_units[0]))
            layers.append(nn.ReLU())
            for i in range(1, len(n_hidden_units)):
                layers.append(nn.Linear(n_hidden_units[i - 1], n_hidden_units[i]))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(n_hidden_units[-1], 1))
            layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x


def create_fully_connected_nn_tf(
        input_size: int,
        output_size: int,
        neurons_per_hidden_layer: Iterable[int],
        activation_function: str = 'relu',
        latest_activation_function: str = 'sigmoid'
) -> Model:
    input_layer = Input(shape=(input_size,))
    x = input_layer
    for neurons in neurons_per_hidden_layer:
        x = Dense(neurons, activation=activation_function)(x)
    output_layer = Dense(output_size, activation=latest_activation_function)(x)
    return Model(inputs=input_layer, outputs=output_layer)


def evaluate_predictions(protected: np.array, y_pred: np.array, y_true: np.array, logger: Logger) -> None:
    """
    Evaluate the predictions. Compute the following metrics:
    - Accuracy
    - Precision
    - Recall
    - F1 score
    - Statistical parity
    - TPR parity (equal opportunity)
    - FPR parity

    @param protected: protected features
    @param y_pred: predicted labels
    @param y_true: true labels
    @param logger: logger
    """
    binary_predictions = np.squeeze(np.where(y_pred >= 0.5, 1, 0))
    tp = np.sum(np.logical_and(binary_predictions == 1, y_true == 1))
    tn = np.sum(np.logical_and(binary_predictions == 0, y_true == 0))
    fp = np.sum(np.logical_and(binary_predictions == 1, y_true == 0))
    fn = np.sum(np.logical_and(binary_predictions == 0, y_true == 1))
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    auc = roc_auc_score(y_true, y_pred)
    dp = demographic_parity(protected, y_pred)
    eo = equalized_odds(protected, y_true, y_pred)
    di = disparate_impact(protected, y_pred)
    logger.info(f"metrics:")
    for metric, value in zip(METRIC_LIST_NAMES, [accuracy, precision, recall, f1_score, auc, dp, di, eo]):
        logger.info(f"{INDENT}{metric}: {value:.{LOG_FLOAT_PRECISION}f}")
