"""
Page 4: Portfolio Simulator - Manual weight allocation and backtest
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils import (
    fetch_data, compute_returns, annualize_returns, annualize_covariance,
    portfolio_performance,
)

st.set_page_config(page_title="Portfolio Simulator", page_icon="💼", layout="wide")
st.title("Portfolio Simulator")

tickers = st.session_state.get("tickers", ["AAPL", "MSFT", "GOOGL"])
period = st.session_state.get("period", "1y")
risk_free = st.session_state.get("risk_free", 0.04)

if not tickers:
    st.warning("Please select tickers on the Home page.")
    st.stop()

with st.spinner("Loading data..."):
    prices = fetch_data(tickers, period)
    returns = compute_returns(prices)
    mean_rets = annualize_returns(returns)
    cov_mat = annualize_covariance(returns)

st.markdown("### Allocate Your Portfolio")
st.write("Adjust weights below. Weights are auto-normalized to sum to 100%.")

# ─── Weight Sliders ──────────────────────────────────────────────────
weights_raw = {}
cols = st.columns(min(5, len(tickers)))
for i, t in enumerate(tickers):
    with cols[i % len(cols)]:
        weights_raw[t] = st.slider(t, 0, 100, 100 // len(tickers), 5, key=f"w_{t}")

total_raw = sum(weights_raw.values())
if total_raw == 0:
    st.error("Total weight is zero. Please assign at least some weight.")
    st.stop()

weights = {t: w / total_raw for t, w in weights_raw.items()}
w_array = np.array([weights[t] for t in tickers])
st.session_state["opt_weights"] = w_array

# Show normalized weights
st.write("**Normalized weights:**", " | ".join(f"{t}: {w:.0%}" for t, w in weights.items()))

# ─── Compute Performance ────────────────────────────────────────────
port_ret, port_vol = portfolio_performance(w_array, mean_rets.values, cov_mat.values)
port_sharpe = (port_ret - risk_free) / port_vol if port_vol > 0 else 0

# Backtest: cumulative portfolio value
port_daily_returns = (returns * w_array).sum(axis=1)
cumulative = (1 + port_daily_returns).cumprod()

# Benchmark: equal weight
eq_w = np.ones(len(tickers)) / len(tickers)
eq_daily = (returns * eq_w).sum(axis=1)
eq_cumulative = (1 + eq_daily).cumprod()

# ─── Metrics ─────────────────────────────────────────────────────────
st.markdown("---")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Annual Return", f"{port_ret:.2%}")
m2.metric("Annual Volatility", f"{port_vol:.2%}")
m3.metric("Sharpe Ratio", f"{port_sharpe:.3f}")

# Max drawdown
cum_max = cumulative.cummax()
drawdown = (cumulative - cum_max) / cum_max
max_dd = drawdown.min()
m4.metric("Max Drawdown", f"{max_dd:.1%}")

# ─── Charts ──────────────────────────────────────────────────────────
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["Cumulative Return", "Drawdown", "Monthly Returns"])

with tab1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cumulative.index, y=cumulative.values,
        mode="lines", name="Your Portfolio",
        line=dict(color="#636EFA", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=eq_cumulative.index, y=eq_cumulative.values,
        mode="lines", name="Equal Weight",
        line=dict(color="#EF553B", width=2, dash="dash"),
    ))
    fig.update_layout(
        xaxis_title="Date", yaxis_title="Portfolio Value ($1 invested)",
        hovermode="x unified", height=450, template="plotly_dark",
        title="Backtested Cumulative Performance",
    )
    st.plotly_chart(fig, width='stretch')

with tab2:
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=drawdown.index, y=drawdown.values,
        mode="lines", fill="tozeroy",
        line=dict(color="#EF553B"),
        name="Drawdown",
    ))
    fig_dd.update_layout(
        xaxis_title="Date", yaxis_title="Drawdown",
        yaxis_tickformat=".0%",
        hovermode="x unified", height=400, template="plotly_dark",
        title="Portfolio Drawdown Over Time",
    )
    st.plotly_chart(fig_dd, width='stretch')

with tab3:
    monthly = port_daily_returns.resample("ME").sum()
    colors = ["#00CC96" if r >= 0 else "#EF553B" for r in monthly.values]
    fig_m = go.Figure(go.Bar(
        x=monthly.index, y=monthly.values * 100,
        marker_color=colors,
    ))
    fig_m.update_layout(
        xaxis_title="Month", yaxis_title="Return (%)",
        height=400, template="plotly_dark",
        title="Monthly Returns",
    )
    st.plotly_chart(fig_m, width='stretch')

# ─── Statistics Summary ─────────────────────────────────────────────
st.markdown("---")
st.subheader("Detailed Statistics")

col_a, col_b = st.columns(2)

with col_a:
    st.markdown("**Your Portfolio**")
    stats = {
        "Annual Return": f"{port_ret:.2%}",
        "Annual Volatility": f"{port_vol:.2%}",
        "Sharpe Ratio": f"{port_sharpe:.3f}",
        "Max Drawdown": f"{max_dd:.2%}",
        "Best Day": f"{port_daily_returns.max():.2%}",
        "Worst Day": f"{port_daily_returns.min():.2%}",
        "Skewness": f"{port_daily_returns.skew():.3f}",
        "Kurtosis": f"{port_daily_returns.kurtosis():.3f}",
    }
    for k, v in stats.items():
        st.write(f"- **{k}:** {v}")

with col_b:
    st.markdown("**Equal Weight Benchmark**")
    eq_ret, eq_vol = portfolio_performance(eq_w, mean_rets.values, cov_mat.values)
    eq_sharpe = (eq_ret - risk_free) / eq_vol if eq_vol > 0 else 0
    eq_dd = ((eq_cumulative - eq_cumulative.cummax()) / eq_cumulative.cummax()).min()
    stats_eq = {
        "Annual Return": f"{eq_ret:.2%}",
        "Annual Volatility": f"{eq_vol:.2%}",
        "Sharpe Ratio": f"{eq_sharpe:.3f}",
        "Max Drawdown": f"{eq_dd:.2%}",
        "Best Day": f"{eq_daily.max():.2%}",
        "Worst Day": f"{eq_daily.min():.2%}",
        "Skewness": f"{eq_daily.skew():.3f}",
        "Kurtosis": f"{eq_daily.kurtosis():.3f}",
    }
    for k, v in stats_eq.items():
        st.write(f"- **{k}:** {v}")

st.markdown("---")
st.caption("Past performance does not guarantee future results. This is for educational purposes only.")
