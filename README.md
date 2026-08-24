# TimeCast: Forecasting Future Sales Trends Using Time Series Analysis

An end-to-end time series forecasting project that analyzes historical retail sales data to identify trends and seasonality, and predicts future sales using machine learning models.

## 📌 Project Overview

Accurate sales forecasting helps businesses plan inventory, staffing, and marketing campaigns ahead of demand. This project analyzes daily sales data from a superstore, engineers time-based features, and builds regression models to forecast sales for the next 30 days.

## 📊 Dataset

- **Source:** [Superstore Sales Dataset — Kaggle](https://www.kaggle.com/datasets/rohitsahoo/sales-forecasting)
- **Data:** Daily order-level sales transactions aggregated into a continuous daily sales time series
- **Target variable:** `Sales` (daily total)

## 🛠️ Tools & Libraries

- **Language:** Python (Google Colab)
- **Data handling:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Time series decomposition:** Statsmodels
- **Machine Learning:** Scikit-learn

## 🔍 Project Workflow

### 1. Data Cleaning
- Converted `Order Date` to datetime format
- Removed duplicate records
- Sorted data chronologically (essential for time series)
- Aggregated transaction-level data into a continuous daily sales series (missing days filled with 0)

### 2. Exploratory Data Analysis
- Plotted raw daily sales to observe overall trend and volatility
- Aggregated to monthly sales to reveal a clearer upward/downward trend
- Analyzed seasonality using month-wise sales distribution (boxplots)
- Compared sales across years to identify recurring seasonal peaks
- Performed trend/seasonality/residual decomposition using `seasonal_decompose`

### 3. Feature Engineering
**Time-based features:**
- Year, Month, Day, DayOfWeek, DayOfYear, WeekOfYear, Quarter, IsWeekend

**Lag features:**
- Sales from 1, 7, 14, and 30 days prior

**Rolling-window features:**
- 7-day and 30-day rolling mean, 7-day rolling standard deviation

### 4. Chronological Train-Test Split
- Data split by time order (no shuffling) — last 15% of the timeline used as the test set, preserving the sequential nature of time series data

### 5. Model Building & Comparison
Trained and compared two regression models:
- Linear Regression (baseline)
- Random Forest Regressor

### 6. Forecasting
- Generated a 30-day future sales forecast using an iterative prediction approach, where each day's prediction feeds into the lag/rolling features for the next day

## 📈 Model Results

| Model | MAE | RMSE | MAPE |
|-------|------|------|------|
| Linear Regression | 1838.26 | 2508.66 | 676.65% |
| **Random Forest** | **1766.98** | **2509.79** | **525.06%** |

**Best Model:** Random Forest achieved a lower MAE and significantly lower MAPE than Linear Regression, indicating it captures the non-linear patterns in daily sales more effectively.

**Note on MAPE:** The high MAPE values are expected for this dataset — daily sales include many low-value and near-zero days, which inflate percentage-based error metrics. MAE and RMSE are more reliable indicators of forecast accuracy here, since MAPE becomes unstable when actual values are close to zero.

## 🔎 Key Findings

Top predictors of sales (from Random Forest feature importance):

1. **RollingMean_30** — the 30-day rolling average was the strongest predictor, showing that recent sustained sales level is the best indicator of near-term sales
2. **Sales_lag_7** — sales from exactly one week prior strongly predicts current sales, pointing to a weekly demand cycle
3. **DayOfYear** — captures seasonal position within the year, confirming seasonality plays a real role
4. **Sales_lag_14** and **RollingMean_7** — short and medium-term momentum both matter
5. Calendar features like **Year, Month, IsWeekend, Quarter** had comparatively low importance, suggesting daily sales are driven more by recent momentum than by broad calendar effects

**Observations on the forecast:** Historical daily sales show sharp, irregular spikes (large individual orders), while the 30-day forecast is much smoother — this is expected, since the model predicts the underlying trend/pattern rather than one-off spike orders, which are inherently hard to predict from historical patterns alone.

## 💡 Business Recommendations

1. Use the 30-day forecast as a **directional guide for inventory planning**, not an exact day-by-day prediction, given the smoothing effect on spikes
2. Since recent 30-day performance (RollingMean_30) is the top driver, **monitor rolling sales trends closely** as an early signal for demand shifts
3. Investigate the weekly pattern (Sales_lag_7 importance) to align staffing and restocking schedules with the weekly demand cycle
4. Treat large sales spikes as **separate large-order events** to plan for, rather than expecting the forecasting model to predict them
5. Retrain the model periodically as new sales data comes in, since recent momentum features drive most of the predictive power

## 📁 Repository Structure

```
timecast-sales-forecasting/
│
├── TimeCast_Project.ipynb   # Full Colab notebook (EDA → Feature Engineering → Forecasting)
└── README.md                # Project documentation
```

## 🚀 How to Run

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/rohitsahoo/sales-forecasting)
2. Open the notebook in [Google Colab](https://colab.research.google.com/)
3. Upload the dataset CSV to the Colab session
4. Run all cells in order

## 📦 Dependencies

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- statsmodels

## 🙋 About This Project

This project was built as part of the **QSkill Data Science Internship** (Slab 2 – Intermediate), covering time-series data cleaning, EDA, trend/seasonality analysis, feature engineering (time-based, lag, rolling-window), chronological train-test splitting, model comparison, and future sales forecasting.
