# Oilman Sachs — WTI Crude Oil Price Prediction

Group project for **INDENG 242A (Machine Learning and Data Analytics I)**, UC Berkeley, Fall 2025.
Team: Arthur Fang, Yuan Jiang, Jasmine Chen, Haorui Zhang, Wish Wang.

A comparative study of forecasting methods for WTI crude oil, built around two questions that come up constantly in financial machine learning:

1. **Are advanced methods always superior?** We benchmark "black-box" models (Random Forest, XGBoost, LSTM) against a Random Walk, a linear baseline, and classical ARIMA-family models.
2. **Is "garbage in, garbage out" solvable?** Financial feature sets are wide and noisy. We test whether aggressive feature engineering (PCA, Elastic Net) or high-capacity models can extract signal from 400+ raw inputs sampled at different frequencies.

The full write-up is in [`FinalReportINDENG_242A.pdf`](FinalReportINDENG_242A.pdf).

## Key findings

- **For price-level regression, nothing beats the Random Walk.** RW reaches RMSE 1.34 / MAPE 1.48 % / R² 0.93 on the test set; tuned AR, MA and ARIMA land within a few percent of it. Random Forest and XGBoost are an order of magnitude worse (RMSE > 12, negative R²).
- **For directional forecasting, the ranking flips.** Random Walk and ARIMA-family models have essentially no directional skill (AUC ≈ 0.45–0.50), while Random Forest and XGBoost achieve recall ≈ 0.98 and AUC ≈ 0.57.
- **LSTMs are competitive on regression (best single model RMSE 1.78, R² 0.77) but unstable** across hyperparameters; a top-10 ensemble is *worse* than the best single model, and directional AUC is at chance.
- Advanced methods are therefore not uniformly better: which model wins depends on whether the target has exploitable structure (direction) or is close to a martingale (level).

## Data

- **Span:** 2021-01-04 → 2025-11-24 (post-COVID regime only), 1,229 daily observations.
- **Target:** next-day WTI (`CL=F`) price / return.
- **Market data (Yahoo Finance, 18 instruments):** WTI, Brent, RBOB gasoline, heating oil, natural gas, DXY, US 10Y yield, OVX, USD/CAD, copper, gold, S&P 500, XLE, EEM, TIP, IYT, OIH, HYG.
- **Engineered features:** crack spread (3:2:1), Gold/Oil, Copper/Oil, Transport/Oil, Services/Oil ratios; RSI, MACD, Bollinger Bands, ATR, momentum, volatility and lagged features.
- **Weekly fundamentals (EIA Weekly Petroleum Status Report, `data/psw01–07.xls`, 15 sheets):** crude and product stocks, field production, refinery inputs and utilization by PADD region. Forward-filled onto the daily grid and de-duplicated across EIA tables.

The cleaned panel is exported as `data/cleaned_oil_prediction_data.csv` (full) and `data/shortened_oil_data.csv` (reduced feature set used by the time-series and LSTM notebooks).

## Methodology

1. **Baseline** — OLS on all raw features (RMSE 25.5): fails on multicollinearity and dimension.
2. **Feature engineering** — PCA within feature groups (446 features → 10 PCs, 97.8 % reduction) and Elastic Net selection by category (49 features retained; Lasso gave the same set with worse convergence).
3. **Models**
   - Random Walk, plus a *smoothed* variant anchored on the trailing two-week average with EIA fundamentals as a regime indicator.
   - Random Forest (5-fold CV over `max_features`, 500 trees) and XGBoost.
   - AR / MA / ARIMA with grid search on AIC/BIC and rolling one-step-ahead forecasts.
   - LSTM: grid search over 216 configurations (sequence length, units, dropout, learning rate), top-10 ensemble.
4. **Evaluation** — strict chronological 80/20 split. Regression metrics (RMSE, MAPE, AIC, R²) **and** directional-classification metrics (precision, recall, AUC-ROC, log loss), on the argument that trading P&L depends on direction and calibration more than on point accuracy.

### Results (test set)

| Model | RMSE | MAPE (%) | R² | Precision | Recall | AUC-ROC | Log loss |
|---|---|---|---|---|---|---|---|
| Linear baseline (raw) | 25.54 | 33.20 | −25.08 | 0.50 | 0.91 | 0.49 | 0.97 |
| **Random Walk** | **1.34** | **1.48** | **0.93** | 0.00 | 0.00 | 0.50 | 0.69 |
| Smoothed RW (with features) | 1.52 | 1.65 | 0.91 | 0.52 | 0.53 | 0.54 | 0.77 |
| Random Forest | 13.13 | 19.19 | −5.89 | 0.50 | **0.99** | **0.57** | 1.28 |
| XGBoost | 12.08 | 17.46 | −4.83 | 0.50 | 0.98 | **0.57** | 1.18 |
| AR(7), tuned | 1.38 | 1.57 | 0.86 | 0.51 | 0.48 | 0.45 | 0.93 |
| MA(20), tuned | 1.73 | 2.00 | 0.78 | 0.48 | 0.49 | 0.44 | 1.01 |
| ARIMA(5,1,5), tuned | 1.37 | 1.53 | 0.86 | 0.49 | 0.45 | 0.47 | 0.91 |
| LSTM (single best) | 1.78 | 2.14 | 0.77 | 0.49 | 0.46 | 0.47 | 0.86 |
| LSTM (top-10 ensemble) | 3.65 | 5.16 | 0.01 | 0.47 | 0.42 | 0.45 | 0.80 |

Per-model predictions and grid-search logs are in `results/`.

## Repository layout

```
notebooks/
  Data Processing.ipynb        # yfinance download, feature engineering, EIA merge, QA, export
  Time Series Modeling.ipynb   # AR / MA / ARIMA grid search, rolling forecasts, directional metrics
  Neural Network Modeling.py   # LSTM grid search (216 configs) + top-10 ensemble
  Neural Network Metrics.ipynb # regression + directional metrics for the LSTM runs
  Analytics.ipynb              # LSTM result analysis, error analysis, trading-strategy backtest
dashboard/
  dashboard.py                 # builds a self-contained Plotly dashboard
  prediction_dashboard.html    # interactive predictions-vs-actual dashboard (open in a browser)
data/                          # EIA weekly reports (psw01–07.xls) and the cleaned panels
results/                       # grid-search tables, predictions, metric summaries
FinalReportINDENG_242A.pdf     # final report
```

## Reproducing

```bash
pip install yfinance pandas numpy xlrd scikit-learn statsmodels xgboost tensorflow plotly matplotlib seaborn openpyxl
```

Run `notebooks/Data Processing.ipynb` first (it writes the cleaned CSVs), then the modeling notebooks in any order. `python notebooks/Neural\ Network\ Modeling.py` reproduces the LSTM grid search (slow on CPU). `python dashboard/dashboard.py` rebuilds `dashboard/prediction_dashboard.html` from `results/model_predictions_vs_actual.xlsx`.
