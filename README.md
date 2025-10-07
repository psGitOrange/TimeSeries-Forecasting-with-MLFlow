# Time Series Forecasting with Experiment Tracking

> *“All models are wrong, but some are useful.” – George Box*
>
> In this project, we explore multiple time series forecasting approaches - from classical SARIMAX and Prophet to deep learning-based PatchTST - and use **MLflow** for systematic experiment tracking and model comparison.

There isn’t a one-size-fits-all solution, different models like SARIMAX, Prophet, or deep learning approaches may perform differently depending on the data and the application. Finding the right model often takes many experiments. In this process, it becomes crucial to keep track of inputs, evaluation metrics, and model versions, with so many moving parts it’s difficult to compare to find best model. This is where tools like MLflow helps by providing experiment tracking, model versioning, etc to ensure reproducibility and make deployment reliable.

### Business Problem statement

Understand the per page view report for different wikipedia pages for 550 days, and forecasting the number of views so that you can predict and optimize the ad placement for your clients.

You are provided with the data of 145k wikipedia pages and daily view count.

**Concepts Used:**
- Data Preprocessing
- Time Series forecasting- SARIMAX, Prophet, PatchTST
- Experiment Tracking with MLFlow

We experiment with:

* **SARIMAX:** Classical statistical model handling trend, seasonality, and exogenous features.
* **Facebook Prophet:** Additive model suitable for capturing seasonality and holiday effects.
* **PatchTST:** Transformer-based deep learning model for multivariate time series.

All experiments are tracked using **MLflow**, including dataset versions, model parameters, metrics, and serialized models for reproducibility.

### Dataset
Datalink: https://drive.google.com/drive/folders/1mdgQscjqnCtdg7LGItomyK0abN6lcHBb

| File                    | Description                                                                   |
| ----------------------- | ----------------------------------------------------------------------------- |
| `train_1.csv`           | Raw time series data of webpage views by date and language.                   |
| `prep_ts.csv`           | Preprocessed time series data with datetime index and target variable (`y`).  |
| `Exog_Campaign_eng.csv` | Exogenous variable file containing event or campaign dates (binary flag 1/0). |

## How to Run
### 1. Clone and open the notebook
Open the Jupyter Notebook / Colab notebook provided in the repository (`TimeSeries_MLFlow.ipynb`).

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
* **Evaluation & Visualization:** Uses helper functions to assess and visualize results.

### 4. Check MLflow UI

After running the models, open the MLflow dashboard to view and compare:

* Model parameters and hyperparameters
* Metrics (RMSE, MAPE)
* Model versions and serialized artifacts

Requirements: 
```bash
pip install pandas numpy matplotlib seaborn statsmodels prophet pystan mlflow tsai
```

### Evaluation Metrics for Model Comparisons  
1. RMSE (Root Mean Squared Error): Measures the square root of the average squared differences between actual and predicted values. It penalizes larger errors more heavily and is sensitive to the scale of data.
2. MAPE (Mean Absolute Percentage Error): Calculates the absolute percentage difference between actual and predicted values. Easy to interpret since it expresses error as a percentage.
