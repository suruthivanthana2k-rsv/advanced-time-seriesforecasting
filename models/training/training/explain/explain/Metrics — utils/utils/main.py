from data.generate_dataset import generate_synthetic_data
from training.train import train_model
from explain.shap_explain import run_shap
from models.baseline_models import sarimax_baseline
from utils.metrics import compute_metrics
import pandas as pd

def run_pipeline():
    generate_synthetic_data()

    train_model("lstm")

    run_shap()

    df = pd.read_csv("dataset.csv")
    y_true, y_pred = sarimax_baseline(df)

    print("Baseline Metrics:", compute_metrics(y_true, y_pred))


if _name_ == "_main_":
    run_pipeline()
