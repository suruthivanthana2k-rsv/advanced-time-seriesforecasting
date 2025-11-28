import tensorflow as tf
from tensorflow.keras import layers, Sequential

def build_lstm_model(seq_len: int, n_features: int, units: int = 64) -> tf.keras.Model:
    """
    LSTM forecasting model.
    """
    model = Sequential([
        layers.LSTM(units, return_sequences=False, input_shape=(seq_len, n_features)),
        layers.Dense(32, activation="relu"),
        layers.Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")
    return model
