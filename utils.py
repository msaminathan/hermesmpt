"""
Utility functions for Modern Portfolio Theory Streamlit App
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
import streamlit as st

# ─── Ticker Categories ───────────────────────────────────────────────
TICKER_CATEGORIES = {
    "US Tech Giants": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO", "ORCL", "CRM"],
    "US Healthcare": ["JNJ", "UNH", "LLY", "PFE", "ABBV", "MRK", "TMO", "ABT", "DHR", "BMY"],
    "US Financials": ["JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "SCHW", "BLK", "C"],
    "US Energy": ["XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY", "HAL"],
    "US Consumer": ["WMT", "PG", "KO", "PEP", "COST", "MCD", "NKE", "SBUX", "TGT", "HD"],
    "Global Indices": ["SPY", "QQQ", "IWM", "EFA", "EEM", "VGK", "VWO", "DIA", "VTI", "VEA"],
    "Bonds & REITs": ["TLT", "IEF", "LQD", "HYG", "VNQ", "IYR", "XLRE", "SCHH", "USRT", "BND"],
    "Crypto & Commodities": ["BTC-USD", "ETH-USD", "GLD", "SLV", "USO", "UNG", "DBA", "PDBC", "GDX", "XME"],
    "Indian Markets": ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
                        "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "LT.NS"],
    "Custom (edit below)": [],
}


@st.cache_data(ttl=3600, show_spinner="Downloading stock data...")
def fetch_data(tickers, period="1y"):
    """Download adjusted close prices for given tickers."""
    data = yf.download(tickers, period=period, auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data[["Close"]].rename(columns={"Close": tickers[0]})
    prices = prices.dropna(how="all").ffill()
    return prices


def compute_returns(prices):
    """Daily log returns."""
    return np.log(prices / prices.shift(1)).dropna()


def annualize_returns(daily_returns):
    return daily_returns.mean() * 252


def annualize_covariance(daily_returns):
    return daily_returns.cov() * 252


# ─── Portfolio Math ──────────────────────────────────────────────────

def portfolio_performance(weights, mean_returns, cov_matrix):
    """Return (annualized_return, annualized_volatility) for a portfolio."""
    ret = np.dot(weights, mean_returns)
    vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return ret, vol


def negative_sharpe(weights, mean_returns, cov_matrix, risk_free_rate=0.04):
    ret, vol = portfolio_performance(weights, mean_returns, cov_matrix)
    return -(ret - risk_free_rate) / vol


def portfolio_volatility(weights, mean_returns, cov_matrix):
    return portfolio_performance(weights, mean_returns, cov_matrix)[1]


def efficient_frontier(mean_returns, cov_matrix, num_points=100, risk_free_rate=0.04):
    """Compute the efficient frontier curve."""
    n = len(mean_returns)
    args = (mean_returns, cov_matrix)

    bounds = tuple((0, 1) for _ in range(n))
    constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}

    min_ret = mean_returns.min()
    max_ret = mean_returns.max()
    target_returns = np.linspace(min_ret, max_ret, num_points)

    frontier_vols = []
    frontier_rets = []
    frontier_weights = []

    for target in target_returns:
        cons = [
            {"type": "eq", "fun": lambda x: np.sum(x) - 1},
            {"type": "eq", "fun": lambda x, t=target: np.dot(x, mean_returns) - t},
        ]
        result = minimize(portfolio_volatility, n * [1.0 / n], args=args,
                          method="SLSQP", bounds=bounds, constraints=cons)
        if result.success:
            frontier_vols.append(result.fun)
            frontier_rets.append(target)
            frontier_weights.append(result.x)

    return frontier_vols, frontier_rets, frontier_weights


def max_sharpe_portfolio(mean_returns, cov_matrix, risk_free_rate=0.04):
    """Find the tangency (max Sharpe ratio) portfolio."""
    n = len(mean_returns)
    args = (mean_returns, cov_matrix)
    bounds = tuple((0, 1) for _ in range(n))
    constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
    result = minimize(negative_sharpe, n * [1.0 / n], args=args + (risk_free_rate,),
                      method="SLSQP", bounds=bounds, constraints=constraints)
    return result


def min_variance_portfolio(mean_returns, cov_matrix):
    """Find the minimum variance portfolio."""
    n = len(mean_returns)
    args = (mean_returns, cov_matrix)
    bounds = tuple((0, 1) for _ in range(n))
    constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
    result = minimize(portfolio_volatility, n * [1.0 / n], args=args,
                      method="SLSQP", bounds=bounds, constraints=constraints)
    return result


def random_portfolios(num_portfolios, mean_returns, cov_matrix, seed=42):
    """Generate random portfolio allocations for Monte Carlo visualization."""
    rng = np.random.default_rng(seed)
    n = len(mean_returns)
    results = np.zeros((3, num_portfolios))
    weights_record = []

    for i in range(num_portfolios):
        w = rng.random(n)
        w /= w.sum()
        weights_record.append(w)
        ret, vol = portfolio_performance(w, mean_returns, cov_matrix)
        results[0, i] = vol
        results[1, i] = ret
        results[2, i] = (ret - 0.04) / vol

    return results, weights_record
