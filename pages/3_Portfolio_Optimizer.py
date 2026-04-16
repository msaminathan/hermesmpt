"""
Page 3: Portfolio Optimizer - Deep dive into optimization with constraint controls
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import minimize
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils import (
    fetch_data, compute_returns, annualize_returns, annualize_covariance,
    portfolio_performance, negative_sharpe, portfolio_volatility,
)

st.set_page_config(page_title="Portfolio Optimizer", page_icon="⚙️", layout="wide")
st.title("Portfolio Optimizer")

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

# ─── Optimization Mode ──────────────────────────────────────────────
st.sidebar.subheader("Optimizer Settings")
opt_mode = st.sidebar.radio(
    "Objective",
    ["Max Sharpe Ratio", "Min Variance", "Target Return", "Target Volatility"],
)

max_weight = st.sidebar.slider("Max weight per asset (%)", 10, 100, 50) / 100.0
n = len(tickers)
bounds = tuple((0, max_weight) for _ in range(n))
constraints_base = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]

# ─── Run Optimization ───────────────────────────────────────────────
result = None

if opt_mode == "Max Sharpe Ratio":
    result = minimize(
        negative_sharpe, n * [1.0 / n],
        args=(mean_rets.values, cov_mat.values, risk_free),
        method="SLSQP", bounds=bounds, constraints=constraints_base,
    )

elif opt_mode == "Min Variance":
    result = minimize(
        portfolio_volatility, n * [1.0 / n],
        args=(mean_rets.values, cov_mat.values),
        method="SLSQP", bounds=bounds, constraints=constraints_base,
    )

elif opt_mode == "Target Return":
    target_ret = st.sidebar.slider(
        "Target Annual Return (%)",
        float(mean_rets.min() * 100), float(mean_rets.max() * 100),
        float(mean_rets.mean() * 100), 0.5,
    ) / 100.0
    cons = constraints_base + [
        {"type": "eq", "fun": lambda x, t=target_ret: np.dot(x, mean_rets.values) - t}
    ]
    result = minimize(
        portfolio_volatility, n * [1.0 / n],
        args=(mean_rets.values, cov_mat.values),
        method="SLSQP", bounds=bounds, constraints=cons,
    )

elif opt_mode == "Target Volatility":
    target_vol = st.sidebar.slider(
        "Target Annual Volatility (%)",
        1.0, 80.0, 20.0, 0.5,
    ) / 100.0

    def vol_constraint(x):
        return portfolio_volatility(x, mean_rets.values, cov_mat.values) - target_vol

    def neg_return(x):
        return -np.dot(x, mean_rets.values)

    cons = constraints_base + [{"type": "eq", "fun": vol_constraint}]
    result = minimize(
        neg_return, n * [1.0 / n],
        method="SLSQP", bounds=bounds, constraints=cons,
    )

if result is None or not result.success:
    st.error("Optimization failed. Try relaxing constraints.")
    st.stop()

opt_weights = result.x
st.session_state["opt_weights"] = opt_weights
opt_ret, opt_vol = portfolio_performance(opt_weights, mean_rets.values, cov_mat.values)
opt_sharpe = (opt_ret - risk_free) / opt_vol if opt_vol > 0 else 0

# ─── Results ─────────────────────────────────────────────────────────
st.markdown("---")

col_metrics = st.columns(4)
col_metrics[0].metric("Objective", opt_mode)
col_metrics[1].metric("Expected Return", f"{opt_ret:.2%}")
col_metrics[2].metric("Volatility", f"{opt_vol:.2%}")
col_metrics[3].metric("Sharpe Ratio", f"{opt_sharpe:.3f}")

col_chart, col_table = st.columns([3, 2])

with col_chart:
    st.subheader("Optimal Allocation")
    # Pie chart
    nonzero = [(t, w) for t, w in zip(tickers, opt_weights) if w > 0.001]
    if nonzero:
        labels, values = zip(*sorted(nonzero, key=lambda x: -x[1]))
        pie_colors = [
            "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
            "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
            "#1F77B4", "#FF7F0E",
        ][:len(labels)]
        fig = go.Figure(go.Pie(
            labels=labels, values=[v * 100 for v in values],
            textinfo="label+percent",
            hole=0.35,
            marker=dict(colors=pie_colors),
        ))
        fig.update_layout(height=450, template="plotly_dark", showlegend=False)
        import plotly.express as px
        st.plotly_chart(fig, width='stretch')
    else:
        st.info("All weights are near zero -- try relaxing constraints.")

with col_table:
    st.subheader("Weights Detail")
    weight_df = pd.DataFrame({
        "Ticker": tickers,
        "Weight": [f"{w:.2%}" for w in opt_weights],
        "Ann. Return": [f"{mean_rets[t]:.1%}" for t in tickers],
        "Ann. Vol": [f"{(cov_mat.loc[t, t] ** 0.5):.1%}" for t in tickers],
    })
    weight_df = weight_df[opt_weights > 0.001].sort_values("Weight", ascending=False)
    st.dataframe(weight_df, width='stretch', hide_index=True)

# ─── Risk Contribution ──────────────────────────────────────────────
st.markdown("---")
st.subheader("Risk Contribution by Asset")

port_var = np.dot(opt_weights.T, np.dot(cov_mat.values, opt_weights))
marginal_risk = np.dot(cov_mat.values, opt_weights)
risk_contrib = opt_weights * marginal_risk / np.sqrt(port_var)
risk_pct = risk_contrib / risk_contrib.sum() * 100

fig_rc = go.Figure(go.Bar(
    x=tickers, y=risk_pct,
    marker_color=["#636EFA" if r > 0 else "#EF553B" for r in risk_pct],
    text=[f"{r:.1f}%" for r in risk_pct],
    textposition="outside",
))
fig_rc.update_layout(
    yaxis_title="% of Total Portfolio Risk",
    height=350, template="plotly_dark",
    title="Marginal Risk Contribution",
)
st.plotly_chart(fig_rc, width='stretch')

st.info("""
**Risk Contribution** shows how much each asset contributes to overall portfolio risk.
In a truly diversified portfolio, no single asset dominates risk. 
If one asset contributes >50% of risk, the portfolio may not be well-diversified.
""")
