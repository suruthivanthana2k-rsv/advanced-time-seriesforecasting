import shap
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from utils.dataloader import create_sequences

def run_shap():
    df = pd.read_csv("dataset.csv")
    X, y = create_sequences(df, seq_len=30)

    model = load_model("lstm_model.h5")

    explainer = shap.DeepExplainer(model, X[:100])
    shap_values = explainer.shap_values(X[:10])

    np.save("shap_values.npy", shap_values)
    print("Saved shap_values.npy")

if _name_ == "_main_":
    run_shap()
