"""
Page 2: Efficient Frontier - Monte Carlo simulation and efficient frontier curve
"""
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils import (
    fetch_data, compute_returns, annualize_returns, annualize_covariance,
    efficient_frontier, max_sharpe_portfolio, min_variance_portfolio,
    random_portfolios, portfolio_performance,
)

st.set_page_config(page_title="Efficient Frontier", page_icon="🎯", layout="wide")
st.title("Efficient Frontier")

tickers = st.session_state.get("tickers", ["AAPL", "MSFT", "GOOGL"])
period = st.session_state.get("period", "1y")
risk_free = st.session_state.get("risk_free", 0.04)

if not tickers:
    st.warning("Please select tickers on the Home page.")
    st.stop()

# Fetch and compute
with st.spinner("Computing efficient frontier..."):
    prices = fetch_data(tickers, period)
    returns = compute_returns(prices)
    mean_rets = annualize_returns(returns)
    cov_mat = annualize_covariance(returns)

n_assets = len(tickers)

# ─── Controls ────────────────────────────────────────────────────────
col_ctrl1, col_ctrl2 = st.columns(2)
with col_ctrl1:
    num_random = st.slider("Random portfolios (Monte Carlo)", 500, 10000, 3000, 500)
with col_ctrl2:
    num_frontier = st.slider("Frontier points", 20, 200, 100, 10)

# ─── Compute ─────────────────────────────────────────────────────────
mc_results, mc_weights = random_portfolios(num_random, mean_rets.values, cov_mat.values)
f_vols, f_rets, f_weights = efficient_frontier(mean_rets.values, cov_mat.values, num_frontier, risk_free)

# Max Sharpe
ms_result = max_sharpe_portfolio(mean_rets.values, cov_mat.values, risk_free)
ms_ret, ms_vol = portfolio_performance(ms_result.x, mean_rets.values, cov_mat.values)
ms_sharpe = (ms_ret - risk_free) / ms_vol

# Min Variance
mv_result = min_variance_portfolio(mean_rets.values, cov_mat.values)
mv_ret, mv_vol = portfolio_performance(mv_result.x, mean_rets.values, cov_mat.values)

# ─── Plot ────────────────────────────────────────────────────────────
fig = go.Figure()

# Random portfolios colored by Sharpe
fig.add_trace(go.Scatter(
    x=mc_results[0], y=mc_results[1],
    mode="markers",
    marker=dict(
        size=3, color=mc_results[2],
        colorscale="Viridis", showscale=True,
        colorbar=dict(title="Sharpe"),
    ),
    name="Random Portfolios",
    hovertemplate="Vol: %{x:.1%}<br>Ret: %{y:.1%}<extra></extra>",
))

# Efficient frontier
fig.add_trace(go.Scatter(
    x=f_vols, y=f_rets,
    mode="lines",
    line=dict(color="cyan", width=3),
    name="Efficient Frontier",
))

# Max Sharpe point
fig.add_trace(go.Scatter(
    x=[ms_vol], y=[ms_ret],
    mode="markers+text",
    marker=dict(size=18, color="gold", symbol="star", line=dict(width=2, color="white")),
    text=[f"Max Sharpe\n({ms_sharpe:.2f})"],
    textposition="top center",
    textfont=dict(color="gold", size=12),
    name="Tangency Portfolio",
))

# Min Variance point
fig.add_trace(go.Scatter(
    x=[mv_vol], y=[mv_ret],
    mode="markers+text",
    marker=dict(size=16, color="lime", symbol="diamond", line=dict(width=2, color="white")),
    text=[f"Min Var"],
    textposition="top center",
    textfont=dict(color="lime", size=12),
    name="Min Variance",
))

# Capital Market Line
cml_x = np.linspace(0, max(f_vols) * 1.1, 50)
cml_y = risk_free + ms_sharpe * cml_x
fig.add_trace(go.Scatter(
    x=cml_x, y=cml_y,
    mode="lines",
    line=dict(color="gold", width=2, dash="dash"),
    name="Capital Market Line",
))

fig.update_layout(
    xaxis_title="Annualized Volatility",
    yaxis_title="Annualized Return",
    xaxis_tickformat=".0%",
    yaxis_tickformat=".0%",
    height=600,
    template="plotly_dark",
    title="Efficient Frontier with Random Portfolios",
    legend=dict(x=0.01, y=0.99),
)
st.plotly_chart(fig, width='stretch')

# ─── Key Metrics ─────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Optimal Portfolios")

c1, c2 = st.columns(2)

with c1:
    st.markdown("### Maximum Sharpe Ratio (Tangency)")
    st.metric("Sharpe Ratio", f"{ms_sharpe:.3f}")
    st.metric("Expected Return", f"{ms_ret:.1%}")
    st.metric("Volatility", f"{ms_vol:.1%}")
    st.markdown("**Allocation:**")
    alloc_ms = {tickers[i]: f"{w:.1%}" for i, w in enumerate(ms_result.x) if w > 0.001}
    for t, w in sorted(alloc_ms.items(), key=lambda x: -float(x[1].strip('%'))):
        st.write(f"  {t}: {w}")

with c2:
    st.markdown("### Minimum Variance")
    sharpe_mv = (mv_ret - risk_free) / mv_vol if mv_vol > 0 else 0
    st.metric("Sharpe Ratio", f"{sharpe_mv:.3f}")
    st.metric("Expected Return", f"{mv_ret:.1%}")
    st.metric("Volatility", f"{mv_vol:.1%}")
    st.markdown("**Allocation:**")
    alloc_mv = {tickers[i]: f"{w:.1%}" for i, w in enumerate(mv_result.x) if w > 0.001}
    for t, w in sorted(alloc_mv.items(), key=lambda x: -float(x[1].strip('%'))):
        st.write(f"  {t}: {w}")

st.markdown("""
---
**Reading the chart:**
- Each dot is a possible portfolio with different asset weights
- Color indicates Sharpe ratio (brighter = better risk-adjusted return)
- The **cyan curve** is the efficient frontier -- no portfolio beats it for the same risk level
- The **gold star** is the tangency portfolio -- optimal risk-adjusted allocation
- The **dashed gold line** is the Capital Market Line -- combinations of risk-free asset and tangency portfolio
""")
