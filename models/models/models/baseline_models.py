import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

def sarimax_baseline(df: pd.DataFrame, train_ratio: float = 0.8):
    target = df["target"]
    n = len(target)
    train, test = target[:int(n*train_ratio)], target[int(n*train_ratio):]

    model = SARIMAX(train, order=(2,1,2))
    result = model.fit(disp=False)
    predictions = result.predict(start=len(train), end=n-1)

    return test.values, predictions.values
