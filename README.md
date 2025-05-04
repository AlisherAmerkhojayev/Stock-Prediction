# FTSE 100 Stock Price Forecasting using Deep Learning with Macroeconomic Indicators

## Overview
This repository presents a rigorous and comprehensive research project investigating the predictive power of state-of-the-art deep learning models—Convolutional Neural Networks (CNN), Long Short-Term Memory (LSTM), and a hybrid CNN-LSTM architecture—on the FTSE 100 index. The study uniquely integrates macroeconomic indicators such as interest rates, inflation, GDP growth, unemployment, and exchange rates into the forecasting models, offering a holistic view of financial markets. The models were evaluated using rich historical data (2013–2023) to assess their ability not only to fit past data but also to generalize under extreme market volatility—particularly post-COVID economic turbulence.

This research is the culmination of extensive experimentation, rigorous tuning, and critical reflections. It targets professionals, quants, and academics seeking robust and interpretable deep learning solutions in financial forecasting.

## Objectives
- Model Evaluation: Assess the performance of CNN, LSTM, and CNN-LSTM models in forecasting FTSE 100 daily stock prices.
- Macroeconomic Integration: Evaluate the influence of macroeconomic indicators on forecasting accuracy.
- Generalization Assessment: Examine each model’s robustness to unseen test data, particularly from the volatile post-2020 market.
- Directional Accuracy: Focus not only on prediction accuracy but on the correctness of directional trends, crucial in trading contexts.
- Sliding Window Framework: Implement a sliding window approach to simulate real-world time-series prediction with 20-day historical inputs.

## Dataset
**Timeframe**: January 2013 – December 2023  
**Market Index**: FTSE 100 (UK)  
**Frequency**: Daily data for market variables; monthly/quarterly for macro indicators  

**Market Features**:
- Open, High, Low, Close, Volume

**Macroeconomic Indicators**:
- Inflation Rate (CPI)
- Interest Rate (BOE)
- Real GDP Growth
- Unemployment Rate
- Exchange Rates (GBP/USD, GBP/EUR)

**Normalization**: Applied Min-Max Scaling to normalize features between 0 and 1. Scalers were fit only on training data to avoid leakage.

## Model Architectures

### 1. CNN (Convolutional Neural Network)
CNNs are used for their strength in capturing localized temporal features in time-series data. The architecture involved:
- 3 × Conv1D layers with filter sizes increasing from 32 to 256
- Kernel sizes of 2–3 to capture short-term volatility
- MaxPooling and ReLU activation
- Dense layer (128 units) for output regression
- Dropout (0.3) for regularization

**Strength**: Excellent at short-term feature extraction (e.g., trend reversals, momentum patterns)

### 2. LSTM (Long Short-Term Memory)
LSTM networks capture long-term dependencies via internal memory cells and gates. Architecture highlights:
- Single LSTM layer (128 units)
- Return_sequences = False for next-day price prediction
- Dense output layer (112 units)
- Dropout (0.4) for regularization
- Higher learning rate (0.01) for faster convergence

**Strength**: Superior at learning trends across extended timeframes, effective in stable regimes.

### 3. CNN-LSTM Hybrid
Combines CNN’s localized pattern detection with LSTM’s temporal modeling:
- CNN blocks (32, 64, 256 filters) process 1D input
- LSTM layer (128 units) receives CNN feature maps
- Dense output layer (112 units)
- Dropout (0.4) after LSTM
- Learning rate: 0.01

**Strength**: Captures both short-term volatility and long-term economic patterns.

## Methodology

### Sliding Window Approach
A custom windowing method segmented the data into sequences of 20 days to predict the 21st day. This simulates realistic trading decisions and accommodates model input shapes.

### Hyperparameter Tuning
Used Keras Tuner (Hyperband) for extensive hyperparameter search:
- CNN: filters, kernel sizes, dropout rates, learning rates
- LSTM: units, return_sequences, learning rates
- Optimizer: Adam with learning rate scheduling

### Cross-Validation
Applied TimeSeriesSplit for 5-fold cross-validation preserving temporal order:
- Fold 1: Train (2013–2015) → Validate (2016)
- Fold 2: Train (2013–2016) → Validate (2017)
- ...
- Fold 5: Train (2013–2020) → Validate (2021)

Also included early stopping to reduce overfitting.

## Evaluation Metrics
To thoroughly assess performance, we employed:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Square Error)
- R² (Coefficient of Determination)
- MAPE / SMAPE
- Directional Accuracy (binary trend prediction)
- Accuracy within tolerance (1% band, 50-unit delta)
- Explained Variance Score
- Mean Bias Deviation (MBD)

These were computed on both training and strictly unseen test data (2021–2023).

## Results Summary

### CNN
- Train R²: 0.9304 — 93% variance explained
- Train MAE/RMSE: 99.31 / 142.32
- Train Directional Accuracy: 70.87%
- Test R²: –2.0117 — Negative generalization, worse than mean predictor
- Test MAE/RMSE: 436.21 / 630.59
- Test Directional Accuracy: 67.25%

While the CNN captured market direction well, it struggled with price magnitude under volatile macro conditions—notably inflation shocks and interest rate hikes.

### LSTM
- Train R²: 0.9199
- Train MAE/RMSE: 108.34 / 153.42
- Train Directional Accuracy: 62.71%
- Test R²: –0.8177
- Test MAE/RMSE: 322.05 / 471.81
- Test Directional Accuracy: 57.81%

LSTM generalized better than CNN in terms of absolute error, but underperformed on directional changes, failing to react quickly to short-term events.

### CNN-LSTM
- Train R²: 0.8756
- Train MAE/RMSE: 129.86 / 187.81
- Train Directional Accuracy: 71.55% (highest)
- Test R²: –1.6889
- Test MAE/RMSE: 437.81 / 615.54
- Test Directional Accuracy: 70.24% (best generalization)

CNN-LSTM was the most accurate at predicting trend directions, making it promising for trading signals despite struggles in precise price estimation.

## Key Observations & Insights

1. **Generalization Crisis**  
All models overfit to pre-2020 macroeconomic regimes. Sharp inflation and interest rate surges post-COVID triggered model failure. CNN and CNN-LSTM especially lacked adaptability.

2. **Directional vs. Absolute Prediction**  
Even when R² was negative, directional accuracy remained high (up to 70%). This suggests these models are viable for trend-based trading strategies, if not for price targeting.

3. **LSTM Memory Limits**  
While LSTM excelled in stable periods, it was sluggish to adapt to market shocks. This confirms findings from Fischer & Krauss (2018) that LSTMs require fine-tuning for volatile markets.

4. **Model Complexity vs. Robustness**  
CNN-LSTM's complexity granted it superior directional accuracy but reduced robustness to data drift, emphasizing the trade-off between expressiveness and generalization.

## Lessons Learned
This project was both enlightening and humbling. The financial markets are not just noisy—they’re structurally non-stationary. Key takeaways include:

- Deep learning isn’t magic: even highly-tuned models struggle with macroeconomic regime shifts.
- Model generalization is critical: future models must anticipate structural breaks or incorporate macroeconomic forecasting into their design.
- Directional accuracy is underrated: traders often care more about movement than magnitude—a nuance traditional regression metrics miss.
- Hybrid models show promise, but require sophisticated regularization and validation on diverse economic regimes.

## 🛠Future Work
- **Ensemble Learning**: Combine CNN, LSTM, and classical models (e.g., ARIMA) using model stacking or boosting.
- **Regime-Switching Models**: Train different sub-models for different economic regimes (e.g., high inflation vs. deflation).
- **Attention Mechanisms**: Allow the model to dynamically weight macroeconomic indicators, improving interpretability.
- **Sentiment Analysis**: Integrate news sentiment, volatility indices (e.g., VIX), and social media signals for richer context.
- **Explainability (XAI)**: Apply SHAP or LIME to make predictions transparent for investors.
