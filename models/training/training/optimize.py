import optuna
from models.lstm_model import build_lstm_model
from utils.dataloader import create_sequences
import pandas as pd
import tensorflow as tf

def objective(trial):
    df = pd.read_csv("dataset.csv")
    X, y = create_sequences(df, seq_len=30)

    units = trial.suggest_int("units", 32, 128)

    model = build_lstm_model(30, 5, units)

    model.fit(X, y, epochs=3, batch_size=32, verbose=0)
    loss = model.evaluate(X, y, verbose=0)
    return loss

if _name_ == "_main_":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10)
    print("Best params:", study.best_params)
