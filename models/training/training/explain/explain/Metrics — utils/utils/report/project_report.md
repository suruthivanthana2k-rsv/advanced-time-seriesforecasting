# Project Report — Advanced Time Series Forecasting with Neural Networks

## 1. Dataset Generation
A multivariate stochastic process (5 features, 1500 samples) was created using:

- Sinusoidal seasonal components
- Linear trend
- Gaussian noise
- Random walk process
- Cross-feature correlations

Dataset saved as dataset.csv.

## 2. Deep Learning Architecture
Two architectures implemented:

### LSTM
- 64 units
- Dense → ReLU
- Adam optimizer

### Transformer Encoder
- 4-head MultiHeadAttention
- Feedforward block
- Layer Normalization

Both trained with sequence length = 30.

## 3. Hyperparameter Optimization
Optuna was used.
Best parameters (sample):
