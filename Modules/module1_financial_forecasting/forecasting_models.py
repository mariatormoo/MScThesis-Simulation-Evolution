# MODEL UTILS
# Utilities for data loading and forecasting models.
# Robust to missing libraries; Prophet and XGBoost are optional.

from __future__ import annotations

# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Optional Imports
# Yahoo Finance
try:
    import yfinance as yf
except Exception:
    yf = None
# Prophet
try:
    from prophet import Prophet
except Exception:
    Prophet = None
# XGBoost
try:
    import xgboost as xgb
except Exception:
    xgb = None
#ARIMA
try:
    from statsmodels.tsa.arima.model import ARIMA
except Exception:
    ARIMA = None



# Load Financial Data from Yahoo Finance
def load_data_from_yahoo(ticker: str, period: str, interval: str) -> Optional[pd.DataFrame]:
    """ 
    Load historical stock data for a given ticker from Yahoo Finance if yfinance is available.
    Returns a DataFrame with historical stock data with columns ['ds','y'] suitable for Prophet-style models."""
    if yf is None:
        #print("yfinance library is not installed.")
        return None
    try:
        df = yf.download(ticker, period=period, interval=interval).reset_index(drop=False)
        if df.empty:
            #print(f"No data found for ticker: {ticker}")
            return None

        #Determine date column name depending on interval index
        #date_col = "Date" if "Date" in df.columns else ("Datetime" if "Datetime" in df.columns else df.columns[0])
        #df = df[[date_col, "Close"]].rename(columns={date_col: "ds", "Close": "y"})
        df = df[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
        df["ds"] = pd.to_datetime(df["ds"])
        df = df.dropna().sort_values("ds").reset_index(drop=True)
        return df
    except Exception:
        #print(f"Error loading data.")
        return None
    

# Load Financial Data from CSV File
def load_data_from_csv(file_path: str) -> Optional[pd.DataFrame]:
    """ 
    Allow user-provided CSV with historical stock data.
    The CSV should contain at least two columns: one for dates and one for values.
    Tries common headers.
    Returns a DataFrame with historical stock data with columns ['ds','y'] suitable for Prophet-style models.
    """
    # Load CSV Data 
    try:
        df = pd.read_csv(file_path)
    except Exception:
        #print(f"Error loading data from CSV.")
        return None
    
    # Candidate column names from Dataframe columns names
    candidate_cols = {col.lower(): col for col in df.columns}

    # Candidates for Date
    date_col = None
    for candidate in ["ds", "date", "datetime", "time", "period", "day"]:
        if candidate in candidate_cols:
            date_col = candidate_cols[candidate]
            break
    
    # Candidates for Value
    value_col = None
    for candidate in ["y", "value", "close", "price", "adj close", "adjclose", "amount"]:
        if candidate in candidate_cols:
            value_col = candidate_cols[candidate]
            break

    # If candidates for Date/Value not found
    if date_col is None or value_col is None:
        # By default, first 2 columns
        if len(df.columns) >= 2:
            date_col, value_col = df.columns[:2]
        # Less than 2 cols, data missing
        else:
            return None
    
    data = df[[date_col, value_col]].copy()
    # Change column names to Standard Names
    data.columns = ["ds", "y"]
    data["ds"] = pd.to_datetime(data["ds"], errors='coerce')
    #data["y"] = pd.to_numeric(data["y"], errors='coerce')
    data = data.dropna().sort_values("ds").reset_index(drop=True)   
    
    return data     


# Train/Test Split
def train_test_split(df: pd.DataFrame, test_size: int=30) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simple chronological split by last N rows.
    If test size is 0 or >= len(df) we return appropriate empty or full splits.
    """
    if df is None or df.empty:
        return df.copy(), df.iloc[0:0].copy()
    
    if test_size <= 0 or test_size >= len(df):
        return df.copy(), df.iloc[0:0].copy()
    
    return df.iloc[:-test_size].copy(), df.iloc[-test_size:].copy()


# Baseline Models

# Autoregressive Integrated Moving Average(ARIMA) Model
def fit_arima(train: pd.DataFrame, order=(1,1,1)):
    """
    Fit ARIMA model to training data (if statsmoel available).
    Returns fitted model or None if ARIMA not available.
    """
    if ARIMA is None:
        return None
    
    arima_model = ARIMA(train['y'], order=order)
    fitted_arima = arima_model.fit()
    return fitted_arima

def forecast_arima(fitted_model, steps: int) -> np.ndarray:
    """
    Forecast using fitted ARIMA model.
    Returns forecasted values or None if model is None.
    """
    if fitted_model is None:
        return None
    
    arima_forecast = fitted_model.forecast(steps=steps)
    return arima_forecast.values


# check options as input !!!
# Holt-Winters Exponential Smoothing Model
def fit_holtwinters(train: pd.DataFrame, seasonal: Optional[str] = None, seasonal_periods: Optional[int] = None):
    """
    Fit Holt-Winters Exponential Smoothing.
    seasonal: 'add' or 'mul' or None.
    seasonal_periods: integer or None.
    """
    holtwinters_model = ExponentialSmoothing(
        train["y"].values,
        trend="add",
        seasonal=seasonal,
        seasonal_periods=seasonal_periods
    )

    fitted_holtwinters = holtwinters_model.fit(optimized=True)
    return fitted_holtwinters

def forecast_holtwinters(fitted_model, steps:int) -> np.ndarray:
    """
    Forecast using fitted Holt-Winters model.
    Returns forecasted values or None if model is None.
    """
    if fitted_model is None:
        return None
    
    holtwinters_forecast = fitted_model.forecast(steps=steps)
    return np.asarray(holtwinters_forecast)


# Prophet Model
def fit_prophet(train: pd.DataFrame):
    """
    Fit Prophet model to training data.
    Returns model object or None if Prophet not available.
    """
    if Prophet is None:
        return None
    
    prophet_model = Prophet()
    prophet_model.fit(train)
    return prophet_model

def forecast_prophet(fitted_model, horizon_days: int, include_history: bool = True) -> pd.DataFrame:
    """
    Forecast using fitted Prophet model.
    """
    if fitted_model is None:
        return None
    
    future_data = fitted_model.make_future_dataframe(periods=horizon_days, freq='D', include_history=include_history)
    prophet_forecast = fitted_model.predict(future_data)
    return prophet_forecast


# XGBoost simple regressor over lag features
def fit_xgboost(train: pd.DataFrame, max_lag: int = 7):
    """
    Fit a small XGBoost model using lagged features.
    Returns (booster, max_lag) or (None, None) if xgboost not available.
    """
    if xgb is None:
        return None, None
    
    training_set = train["y"].values
    X, y = [], []
    for i in range(max_lag, len(training_set)):
        X.append(training_set[i-max_lag:i])
        y.append(training_set[i])

    if not X:
        return None, None
    
    X = np.array(X)
    y = np.array(y)

    # Rename variables pls!!
    dtrain = xgb.DMatrix(X, label=y)
    parameters = {'objective': 'reg:squarederror', 'max_depth': 3, 'eta': 0.1}
    xgbooster = xgb.train(parameters, dtrain, num_boost_round=200)

    return xgbooster, max_lag


def forecast_xgboost(model_tuple, steps: int, history: np.ndarray):
    """
    Rolling forecast using trained xgboost booster and lag window.
    """
    if xgb is None:
        return None
    
    booster, max_lag = model_tuple
    #history_list = history.copy().tolist()
    history_list = history.flatten().tolist() # Flatten 2D array
    predictions = []

    for i in range(steps):
        X = np.array(history_list[-max_lag:]).reshape(1, -1)
        dtest = xgb.DMatrix(X)
        y_pred = float(booster.predict(dtest))
        predictions.append(y_pred)
        history_list.append(y_pred)

    return np.array(predictions)



# Metrics

def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Describe what function does.    
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    eps = 1e-8
    return float(np.mean(np.abs((y_true - y_pred)/np.maximum(np.abs(y_true), eps))) * 100.0)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Describe what function does.    
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred)**2)))



# CFO Manual Forecast (baseline)
# Revise this part!!

def manual_cfo_naive_last_value(train: pd.DataFrame, steps: int) -> np.ndarray:
    """
    Naive: hold last value flat into the future.
    """
    last = float(train["y"].iloc[-1])
    return np.array([last] * steps)


def manual_cfo_growth_rate(train: pd.DataFrame, steps: int, monthly_growth_pct: float = 0.0) -> np.ndarray:
    """
    CFO-style top-down growth assumption.
    monthly_growth_pct is percentage (e.g., 0.5 for 0.5% per month).
    steps is in days.
    """
    last = float(train["y"].iloc[-1])
    growth = 1.0 + monthly_growth_pct / 100.0
    vals = [last * (growth ** (i/30)) for i in range(1, steps + 1)]
    return np.array(vals)






# Plot Forecast
#def plot_forecast(model, forecast):
    #return model.plot(forecast)
