import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from utils.dataloader import create_sequences

def integrated_gradients(model, baseline, input_sample, steps=50):
    interpolated = np.array([
        baseline + (float(i) / steps) * (input_sample - baseline)
        for i in range(steps + 1)
    ])

    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        preds = model(interpolated)

    grads = tape.gradient(preds, interpolated)
    avg_grads = np.mean(grads, axis=0)
    return (input_sample - baseline) * avg_grads


if _name_ == "_main_":
    df = pd.read_csv("dataset.csv")
    X, y = create_sequences(df)

    model = load_model("lstm_model.h5")

    baseline = np.zeros_like(X[0])
    ig = integrated_gradients(model, baseline, X[10])

    np.save("integrated_gradients.npy", ig)
    print("Saved integrated_gradients.npy")
