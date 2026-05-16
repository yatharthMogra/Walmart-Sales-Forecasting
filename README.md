# Walmart Store Sales Forecasting

This project builds end‑to‑end models to forecast weekly sales for Walmart stores and departments using historical sales, holiday information, and promotional features from the Kaggle “Walmart Recruiting – Store Sales Forecasting” competition. The focus is on handling seasonality and holiday effects and comparing tree‑based regression with classical time‑series models on a realistic retail dataset.

---

## Data and Preprocessing

**Source data**

- `train.csv`, `features.csv`, and `stores.csv` from [Kaggle](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting/overview).
- Weekly sales for 45 stores and 82 departments across multiple regions.

**Cleaning and feature engineering** (in `STEP1_Cleaning_and_EDA.ipynb`):

- Merge `train`, `features`, and `stores` into a single dataframe.
- Drop duplicate holiday column (`IsHoliday_y`) and rename `IsHoliday_x` to `IsHoliday`.
- Remove rows with non‑positive `Weekly_Sales` (421,570 → 420,212 rows) to avoid degenerate targets.
- Create holiday indicator columns (`Super_Bowl`, `Labor_Day`, `Thanksgiving`, `Christmas`) using explicit date rules.
- Fill `MarkDown1`–`MarkDown5` nulls with `0`.
- Convert `Date` to `datetime` and derive `week`, `month`, and `year`.
- Save the processed dataset to `clean_data.csv` for downstream modeling.

The notebook also includes exploratory plots and descriptive statistics (monthly/yearly averages, top weeks, store/department comparisons) to understand seasonality and holiday effects.

---

## Random Forest Regression

Notebook: `STEP2_Random_Forest_Regressor.ipynb`

**Setup**

- Load `clean_data.csv`.
- Encode:
  - `Type` as numeric (A/B/C → 1/2/3).
  - Boolean fields (`IsHoliday`, `Super_Bowl`, `Labor_Day`, `Thanksgiving`, `Christmas`) as 0/1.
- Sort the data chronologically by `Date`.
- Perform a **time‑ordered holdout split** (≈70% train, 30% test) rather than a random split.
- Drop the `Date` column before fitting.

**Metric**

- Implement a custom weighted mean absolute error:

  - Errors on holiday weeks are given weight 5, others weight 1.
  - This mirrors the competition’s WMAE, where holiday periods are more important.

**Modeling**

- Use a `RandomForestRegressor` wrapped in a `make_pipeline(RobustScaler(), rf)` to handle scale differences across features.
- Run several feature variants:
  - Reduced feature set without individual holiday columns and without some macro features.
  - Full encoded dataset.
  - Full encoded dataset with feature selection driven by Random Forest feature importances.

**Results**

- WMAE on the holdout set for different configurations (approximate values from the notebook):

  - Reduced feature set (no split holiday columns): ~5850
  - Reduced set without `month`: ~5494
  - Full encoded data: ~2450
  - Full encoded data + feature selection: **~1801** (best Random Forest result)
  - Full encoded + feature selection without `month`: ~2093

- The notebook also reports `R²` scores for some runs (≈0.70–0.74), indicating that a substantial portion of variance is captured by the tree‑based model.

---

## Time‑Series Models (ARIMA and Exponential Smoothing)

Notebook: `STEP3_Modeling_ARIMA_and_ExponentialSmoothing.ipynb`

**Aggregation and stationarity**

- Load `clean_data.csv`, drop `Unnamed: 0`, and convert `Date` to datetime.
- Set `Date` as the index and resample to weekly frequencies:

  - `df_week = df.resample('W').mean()`

- Run stationarity checks (e.g., Augmented Dickey–Fuller) on `Weekly_Sales`.
- Create transformed versions of the series (differenced, lagged, logged) and choose the differenced series for modeling due to non‑stationarity in the raw data.

**Train/test setup**

- Chronological split on the weekly series:

  - Train: 100 weeks
  - Test: 43 weeks

**Models**

1. **ARIMA (auto_ARIMA)**
   - Use `auto_arima` on the differenced training series with a wide search space (`stepwise=False`, larger max p/q/P/Q).
   - Selected ARIMA order in the notebook is `(0, 0, 5)` on the transformed series.

2. **Exponential Smoothing**
   - Fit an `ExponentialSmoothing` model on the differenced training data with:
     - `seasonal='additive'`
     - `trend='additive'`
     - `damped=True`
     - `seasonal_periods=20`
   - Generate forecasts over the test horizon and invert the differencing where needed to interpret results.

3. **ARCH/GARCH (exploratory)**
   - Experiment with `arch_model` (TARCH/ZARCH variants).
   - These models did not perform as well, and the notebook notes that they were not pursued further.

**Evaluation**

- Re‑use the weighted MAE function:

  - Apply holiday weights (5x for holiday weeks, 1x otherwise) when computing errors on the weekly test period.

- Reported best WMAE:

  - **≈821.33** for the Exponential Smoothing configuration on the test set.

This significantly improves over the Random Forest benchmark WMAE and serves as the best model in the current notebook pipeline.

---

## Repository Structure

- `Raw Data/` – Original Kaggle CSV files.
- `clean_data.csv` – Preprocessed merged dataset.
- `STEP1_Cleaning_and_EDA.ipynb` – Data cleaning, feature engineering, and EDA.
- `STEP2_Random_Forest_Regressor.ipynb` – Feature encoding, Random Forest training, feature importance, and WMAE evaluation.
- `STEP3_Modeling_ARIMA_and_ExponentialSmoothing.ipynb` – Weekly resampling, stationarity checks, ARIMA/ExponentialSmoothing/ARCH modeling, and final WMAE evaluation.
- `Walmart Sales Forecast Presentation.pdf` – Slide deck summarizing setup, results, and business insights.

---

## Possible Extensions

If extending this project further, next steps could include:

- Adding lag and rolling‑window features explicitly to the time‑series models.
- Evaluating gradient boosting or deep time‑series models on the same train/test splits.
- Turning residuals from the forecasting models into anomaly scores for detecting unusual demand spikes/drops, analogous to an equipment‑failure early warning system.
