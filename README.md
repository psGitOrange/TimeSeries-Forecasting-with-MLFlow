# TimeSeries-Forecasting-with-MLFlow
Timeseries forecasting with statsmodels SARIMAX, Facebook Prophet and PatchTST using tsai library. Experiment tracking, versioning and comparing model performance with MLFlow.  

# Time Series Forecasting with Experiment Tracking

> *“All models are wrong, but some are useful.” – George Box*
>
> In this project, we explore multiple time series forecasting approaches - from classical SARIMAX and Prophet to deep learning-based PatchTST - and use **MLflow** for systematic experiment tracking and model comparison.

## Project Overview

The goal of this project is to forecast **webpage views** for multiple languages over time, using both historical trends and external event indicators (exogenous variables).

We experiment with:

* **SARIMAX:** Classical statistical model handling trend, seasonality, and exogenous features.
* **Facebook Prophet:** Additive model suitable for capturing seasonality and holiday effects.
* **PatchTST:** Transformer-based deep learning model for multivariate time series.

All experiments are tracked using **MLflow**, including dataset versions, model parameters, metrics, and serialized models for reproducibility.


## Dataset

| File                    | Description                                                                   |
| ----------------------- | ----------------------------------------------------------------------------- |
| `train_1.csv`           | Raw time series data of webpage views by date and language.                   |
| `prep_ts.csv`           | Preprocessed time series data with datetime index and target variable (`y`).  |
| `Exog_Campaign_eng.csv` | Exogenous variable file containing event or campaign dates (binary flag 1/0). |

---

## ⚙️ How to Run

### 1. Clone and open the notebook

Open the Jupyter Notebook / Colab notebook provided in the repository (e.g., `forecast_experiments.ipynb`).

### 2. Set up MLflow

Before running the notebook, ensure MLflow is installed and tracking server is configured.
If not, you can use the default local setup:

```bash
pip install mlflow
mlflow ui
```

Then open the MLflow tracking UI at `http://127.0.0.1:5000`.

### 3. Run the notebook cells sequentially

The notebook includes:

* **Data Preprocessing:** Reads and prepares the input files (`train_1.csv`, `prep_ts.csv`, and `Exog_Campaign_eng.csv`).
* **Exploratory Analysis:** Visualizes time series trends and seasonality across languages.
* **Model Training:** Fits SARIMAX, Prophet, and PatchTST models.
* **Experiment Tracking:** Logs data sources, parameters, and metrics (RMSE, MAPE) to MLflow.
* **Evaluation & Visualization:** Uses helper functions (`performance`, `plot_forecast`) to assess and visualize results.

### 4. Check MLflow UI

After running the models, open the MLflow dashboard to view and compare:

* Model parameters and hyperparameters
* Metrics (RMSE, MAPE)
* Model versions and serialized artifacts

Requirements: 
```bash
pip install pandas numpy matplotlib seaborn statsmodels prophet pystan mlflow tsai
```


Evaluation Metrics
