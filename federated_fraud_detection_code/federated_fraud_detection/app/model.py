import random

import numpy as np
import tensorflow as tf
from tensorflow import keras

from app.config import LEARNING_RATE, RANDOM_STATE


def set_global_seed(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_model(input_dim: int, learning_rate: float = LEARNING_RATE) -> keras.Model:
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(input_dim,), name="transaction_features"),
            keras.layers.Dense(64, activation="relu", name="dense_1"),
            keras.layers.Dropout(0.30, name="dropout_1"),
            keras.layers.Dense(32, activation="relu", name="dense_2"),
            keras.layers.Dropout(0.20, name="dropout_2"),
            keras.layers.Dense(16, activation="relu", name="dense_3"),
            keras.layers.Dense(1, activation="sigmoid", name="fraud_probability"),
        ],
        name="federated_fraud_mlp",
    )
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="binary_crossentropy")
    return model


def get_model_weights(model: keras.Model):
    return model.get_weights()


def set_model_weights(model: keras.Model, weights) -> None:
    model.set_weights(weights)
