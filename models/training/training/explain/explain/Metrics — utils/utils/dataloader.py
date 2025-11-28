import numpy as np
import pandas as pd
from typing import Tuple

def create_sequences(df: pd.DataFrame, seq_len: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts time series DataFrame into sequences for LSTM/Transformer models.
    """
    data = df.values
    X, y = [], []

    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len, :-1])
        y.append(data[i+seq_len, -1])

    return np.array(X), np.array(y)
