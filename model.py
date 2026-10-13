from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


def _load_close_prices(symbol: str) -> pd.Series:
    filename = Path(__file__).resolve().parent / f"{symbol.upper()}.csv"
    data = pd.read_csv(filename)
    if "Date" in data.columns:
        data["Date"] = pd.to_datetime(data["Date"])
        data = data.sort_values("Date", kind="stable")
    return data["Close"]

def predict_price(symbol: str) -> float:
    """
    Fit a linear regression model to the historical closing prices and
    extrapolate the next day's closing price.
    """
    prices = _load_close_prices(symbol)
    if len(prices) < 2:
        raise ValueError("Not enough data (need at least 2 rows).")

    days = np.arange(len(prices)).reshape(-1, 1)
    model = LinearRegression()
    model.fit(days, prices.to_numpy())
    next_day = np.array([[len(prices)]])
    return float(model.predict(next_day)[0])


def moving_average_predict(symbol):
    prices = _load_close_prices(symbol)

    # Use last 5 days for moving average
    last_five = prices.tail(5)

    return float(last_five.mean())


def random_forest_predict(symbol: str) -> float:
    prices = _load_close_prices(symbol)
    if len(prices) < 2:
        raise ValueError("Not enough data (need at least 2 rows).")

    features = prices.iloc[:-1].to_numpy().reshape(-1, 1)
    targets = prices.iloc[1:].to_numpy()
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(features, targets)
    next_day = np.array([[prices.iloc[-1]]])
    return float(model.predict(next_day)[0])


def knn_predict(symbol: str) -> float:
    prices = _load_close_prices(symbol)
    if len(prices) < 2:
        raise ValueError("Not enough data (need at least 2 rows).")

    features = prices.iloc[:-1].to_numpy().reshape(-1, 1)
    targets = prices.iloc[1:].to_numpy()
    model = KNeighborsRegressor(n_neighbors=min(5, len(targets)))
    model.fit(features, targets)
    next_day = np.array([[prices.iloc[-1]]])
    return float(model.predict(next_day)[0])


def svr_predict(symbol: str) -> float:
    prices = _load_close_prices(symbol)
    if len(prices) < 2:
        raise ValueError("Not enough data (need at least 2 rows).")

    features = prices.iloc[:-1].to_numpy().reshape(-1, 1)
    targets = prices.iloc[1:].to_numpy()
    model = make_pipeline(StandardScaler(), SVR())
    model.fit(features, targets)
    next_day = np.array([[prices.iloc[-1]]])
    return float(model.predict(next_day)[0])
