from typing import Iterable, Callable
import tensorflow as tf
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Model


def create_fully_connected_nn(input_size: int, output_size: int, neurons_per_hidden_layer: Iterable[int], activation_function: str = 'relu', latest_activation_function: str = 'sigmoid') -> Model:
    input_layer = Input(shape=(input_size,))
    x = input_layer
    for neurons in neurons_per_hidden_layer:
        x = Dense(neurons, activation=activation_function)(x)
    output_layer = Dense(output_size, activation=latest_activation_function)(x)
    return Model(inputs=input_layer, outputs=output_layer)
