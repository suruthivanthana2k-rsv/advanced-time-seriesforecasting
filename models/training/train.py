import pandas as pd
from utils.dataloader import create_sequences
from models.lstm_model import build_lstm_model
from models.transformer_model import build_transformer

import argparse

def train_model(model_type="lstm"):
    df = pd.read_csv("dataset.csv")
    X, y = create_sequences(df, seq_len=30)

    if model_type == "lstm":
        model = build_lstm_model(30, 5)
    else:
        model = build_transformer(30, 5)

    model.fit(X, y, epochs=20, batch_size=32)
    model.save(f"{model_type}_model.h5")
    print(f"Saved {model_type}_model.h5")


if _name_ == "_main_":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="lstm")
    args = parser.parse_args()

    train_model(args.model)
