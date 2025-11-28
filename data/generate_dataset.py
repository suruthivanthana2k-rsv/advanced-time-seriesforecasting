import numpy as np
import pandas as pd

def generate_synthetic_data(n_samples: int = 1500) -> pd.DataFrame:
    """
    Generate a multivariate time series dataset using a stochastic process.

    Features:
    - Seasonal component
    - Trend
    - Noise
    - Inter-feature dependence
    """
    t = np.arange(n_samples)

    f1 = 0.5 * np.sin(0.02 * t) + 0.01 * t + np.random.normal(0, 0.1, n_samples)
    f2 = 0.3 * np.cos(0.015 * t) + np.random.normal(0, 0.1, n_samples)
    f3 = f1 * 0.4 + f2 * 0.2 + np.random.normal(0, 0.05, n_samples)
    f4 = np.random.normal(0, 1, n_samples).cumsum()  # random walk
    f5 = 0.8 * np.sin(0.01 * t) + np.random.normal(0, 0.1, n_samples)

    df = pd.DataFrame({
        "feat1": f1,
        "feat2": f2,
        "feat3": f3,
        "feat4": f4,
        "feat5": f5,
        "target": f1 * 0.7 + f3 * 0.3 + np.random.normal(0, 0.1, n_samples)
    })

    df.to_csv("dataset.csv", index=False)
    print("Dataset saved to dataset.csv")
    return df


if _name_ == "_main_":
    generate_synthetic_data()
