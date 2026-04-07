# Oilman Sachs: Crude Oil Price Prediction

This repository contains our INDENG 242A final project on **WTI crude oil price forecasting**.  
We compare classical time-series models with neural-network-based methods and evaluate both statistical fit and practical forecasting quality.

## Final Report

- Full project write-up: [`FinalReportINDENG_242A.pdf`](FinalReportINDENG_242A.pdf)

## Project Goals

- Build an end-to-end forecasting pipeline for crude oil prices.
- Compare baseline, time-series, and neural-network models under a consistent evaluation framework.
- Identify which approaches generalize best on held-out data.

## Repository Structure

- `data/` — raw and processed datasets used for modeling.
- `notebooks/` — data processing, exploratory analysis, and model development notebooks/scripts.
- `results/` — model metrics, grid-search outputs, and prediction exports.
- `dashboard/` — lightweight dashboard artifacts for presenting predictions.
- `FinalReportINDENG_242A.pdf` — final report with methodology, experiments, and conclusions.

## Methods Evaluated

### Baselines & Time Series
- Naive benchmark (yesterday’s price)
- Autoregressive models (AR)
- Moving-average models (MA)
- ARIMA models

### Neural Network Models
- Single neural-network model (best run)
- LSTM hyperparameter search + top-model ensemble analysis

## Key Results (from `results/`)

### Overall model comparison (`results/model_comparison_tuned.csv`)
- Best RMSE: **ARIMA(5,1,5) (Tuned)** with RMSE ≈ **1.35** and R² ≈ **0.87**.
- AR(7) and Naive baseline are close behind, showing strong short-horizon persistence.
- Tuned LSTM underperforms top classical time-series models in this dataset split.

### Time-series metrics (`results/ts_comprehensive_metrics.csv`)
- ARIMA(5,1,5) (Tuned) yields the strongest R² among listed time-series candidates.
- Directional classification metrics (precision/recall/AUC) remain near coin-flip levels, indicating that good point forecasts do not automatically imply strong directional signals.

### Neural-network metrics (`results/nn_summary_metrics.csv`, `results/lstm_summary.csv`)
- Best single NN/LSTM run reaches RMSE around **1.78** with R² around **0.76**.
- Ensemble approach performs worse than the best single model in this experiment setting.

## Reproducibility Notes

- Most analysis is documented in notebooks under `notebooks/`.
- Final result tables are stored in CSV form in `results/` for direct inspection.
- The dashboard in `dashboard/` contains presentation-ready outputs.

## Suggested Next Improvements

- Add an explicit `requirements.txt` or `environment.yml` with pinned package versions.
- Introduce a single reproducible runner script (e.g., `scripts/run_pipeline.py`) to execute preprocessing + training + evaluation end to end.
- Add walk-forward validation utilities and transaction-cost-aware backtesting for strategy-oriented interpretation.
