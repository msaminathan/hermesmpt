"""
Page 6: Risk Analysis - VaR/CVaR, Factor Model Decomposition, Monte Carlo Forward Simulation
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats as sp_stats
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils import fetch_data, compute_returns, annualize_returns, annualize_covariance

st.set_page_config(page_title="Risk Analysis", page_icon="Risk", layout="wide")
st.title("Advanced Risk Analysis")

tickers = st.session_state.get("tickers", ["AAPL", "MSFT", "GOOGL"])
period = st.session_state.get("period", "1y")
risk_free = st.session_state.get("risk_free", 0.04)

if not tickers:
    st.warning("Please select tickers on the Home page.")
    st.stop()

# ─── Get weights (from optimizer page or equal-weight default) ──────
if "opt_weights" in st.session_state:
    opt_weights = st.session_state["opt_weights"]
else:
    opt_weights = np.ones(len(tickers)) / len(tickers)

with st.spinner("Loading data..."):
    prices = fetch_data(tickers, period)
    returns = compute_returns(prices)
    mean_rets = annualize_returns(returns)
    cov_mat = annualize_covariance(returns)

port_returns = (returns * opt_weights).sum(axis=1)

# ─── Tabs ────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["VaR / CVaR", "Factor Model", "Monte Carlo Forward"])

# =====================================================================
# TAB 1: VaR / CVaR
# =====================================================================
with tab1:
    st.header("Value at Risk & Conditional Value at Risk")
    st.markdown("""
    **Value at Risk (VaR)** answers: *"What is the maximum loss I can expect with X% confidence over N days?"*
    
    **Conditional Value at Risk (CVaR)** (also called Expected Shortfall) answers: *"If losses exceed VaR, 
    what is the average expected loss?"* -- CVaR is always >= VaR and captures tail risk better.
    """)

    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
    with col_ctrl1:
        confidence = st.select_slider("Confidence Level", [90, 95, 99], value=95) / 100.0
    with col_ctrl2:
        holding_days = st.selectbox("Holding Period (days)", [1, 5, 10, 21], index=0)
    with col_ctrl3:
        initial_invest = st.number_input("Portfolio Value ($)", 1000, 10000000, 100000, 1000)

    alpha = 1 - confidence

    # --- Parametric (Variance-Covariance) VaR ---
    port_mean = port_returns.mean()
    port_std = port_returns.std()
    z_score = sp_stats.norm.ppf(alpha)
    param_var = -(port_mean + z_score * port_std) * np.sqrt(holding_days)
    param_cvar = -(port_mean - port_std * sp_stats.norm.pdf(z_score) / alpha) * np.sqrt(holding_days)

    # --- Historical VaR ---
    hist_var = -np.percentile(port_returns, alpha * 100) * np.sqrt(holding_days)
    hist_cvar = -port_returns[port_returns <= -hist_var / np.sqrt(holding_days)].mean() * np.sqrt(holding_days)

    # --- Monte Carlo VaR ---
    np.random.seed(42)
    n_sim = 50000
    mc_samples = np.random.normal(port_mean, port_std, (n_sim, holding_days))
    mc_portfolio = mc_samples.sum(axis=1)
    mc_var = -np.percentile(mc_portfolio, alpha * 100)
    mc_cvar = -mc_portfolio[mc_portfolio <= -mc_var].mean()

    # --- Display Metrics ---
    st.markdown("---")
    st.subheader(f"VaR & CVaR at {confidence:.0%} confidence, {holding_days}-day holding period")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Parametric (Normal)**")
        st.metric("VaR", f"${param_var * initial_invest:,.0f}", f"{param_var:.2%} of portfolio")
        st.metric("CVaR", f"${param_cvar * initial_invest:,.0f}", f"{param_cvar:.2%} of portfolio")
    with col2:
        st.markdown("**Historical**")
        st.metric("VaR", f"${hist_var * initial_invest:,.0f}", f"{hist_var:.2%} of portfolio")
        st.metric("CVaR", f"${hist_cvar * initial_invest:,.0f}", f"{hist_cvar:.2%} of portfolio")
    with col3:
        st.markdown("**Monte Carlo**")
        st.metric("VaR", f"${mc_var * initial_invest:,.0f}", f"{mc_var:.2%} of portfolio")
        st.metric("CVaR", f"${mc_cvar * initial_invest:,.0f}", f"{mc_cvar:.2%} of portfolio")

    st.info("""
    **Three methods compared:**
    - **Parametric**: Assumes returns are normally distributed. Fast but may underestimate tail risk.
    - **Historical**: Uses actual return percentiles. No distribution assumption but limited by sample size.
    - **Monte Carlo**: Simulates 50,000 return paths. Flexible and captures compound effects over holding period.
    """)

    # --- VaR Distribution Chart ---
    st.markdown("---")
    st.subheader("Return Distribution with VaR Thresholds")

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=port_returns * 100, nbinsx=100, name="Daily Returns",
        marker_color="#1565c0", opacity=0.7,
    ))
    # VaR lines
    fig.add_vline(x=-param_var / np.sqrt(holding_days) * 100, line_dash="dash",
                  line_color="red", annotation_text=f"Parametric VaR {confidence:.0%}")
    fig.add_vline(x=-hist_var / np.sqrt(holding_days) * 100, line_dash="dash",
                  line_color="orange", annotation_text=f"Historical VaR {confidence:.0%}")
    fig.add_vline(x=-mc_var / np.sqrt(holding_days) * 100, line_dash="dash",
                  line_color="yellow", annotation_text=f"MC VaR {confidence:.0%}")

    fig.update_layout(
        xaxis_title="Daily Return (%)", yaxis_title="Frequency",
        height=400, template="plotly_dark",
        title="Portfolio Daily Return Distribution with VaR Markers",
    )
    st.plotly_chart(fig, width='stretch')

    # --- VaR by confidence level ---
    st.subheader("VaR Across Confidence Levels")
    conf_levels = np.arange(0.85, 0.999, 0.01)
    var_param = []; var_hist = []
    for c in conf_levels:
        z = sp_stats.norm.ppf(1 - c)
        var_param.append(-(port_mean + z * port_std) * np.sqrt(holding_days) * initial_invest)
        var_hist.append(-np.percentile(port_returns, (1-c) * 100) * np.sqrt(holding_days) * initial_invest)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=conf_levels*100, y=var_param, mode="lines", name="Parametric", line=dict(color="#1565c0")))
    fig2.add_trace(go.Scatter(x=conf_levels*100, y=var_hist, mode="lines", name="Historical", line=dict(color="#EF553B")))
    fig2.update_layout(
        xaxis_title="Confidence Level (%)", yaxis_title="VaR ($)",
        height=350, template="plotly_dark",
        title=f"VaR by Confidence Level ({holding_days}-day, ${initial_invest:,} invested)",
    )
    st.plotly_chart(fig2, width='stretch')

# =====================================================================
# TAB 2: FACTOR MODEL
# =====================================================================
with tab2:
    st.header("Factor Model Decomposition")
    st.markdown("""
    **Factor models** decompose portfolio returns into contributions from systematic risk factors 
    (market, size, value, etc.) and idiosyncratic (asset-specific) risk.
    
    We use **OLS regression**: R_portfolio = alpha + beta1*F1 + beta2*F2 + ... + epsilon
    
    This reveals how much of your portfolio's return is explained by broad market movements 
    versus the unique stock-picking alpha.
    """)

    # Factor selection
    st.subheader("Select Risk Factors")
    factor_options = {
        "SPY (S&P 500 Market)": "SPY",
        "IWM (Small Cap / Size)": "IWM",
        "TLT (Long-Term Bonds)": "TLT",
        "GLD (Gold / Commodities)": "GLD",
        "VWO (Emerging Markets)": "VWO",
        "USO (Oil)": "USO",
    }

    selected_factors = st.multiselect(
        "Choose factors to include in the regression",
        list(factor_options.keys()),
        default=list(factor_options.keys())[:3],
    )

    if not selected_factors:
        st.warning("Please select at least one factor.")
        st.stop()

    factor_tickers = [factor_options[f] for f in selected_factors]

    with st.spinner("Downloading factor data..."):
        # Fetch factor data for the same period
        all_tickers = list(set(tickers + factor_tickers))
        all_prices = fetch_data(all_tickers, period)
        all_returns = compute_returns(all_prices)

    # Align dates
    common_idx = all_returns.index.intersection(port_returns.index)
    y = port_returns.loc[common_idx].values
    factor_data = all_returns[factor_tickers].loc[common_idx]

    # Run regression
    from numpy.linalg import lstsq
    X = factor_data.values
    X_with_const = np.column_stack([np.ones(len(X)), X])
    betas, residuals_sum, rank, sv = lstsq(X_with_const, y, rcond=None)

    alpha_factor = betas[0]
    factor_betas = betas[1:]
    y_pred = X_with_const @ betas
    residuals = y - y_pred

    # R-squared and statistics
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot
    adj_r_squared = 1 - (1 - r_squared) * (len(y) - 1) / (len(y) - len(betas) - 1)

    # Standard errors
    mse = ss_res / (len(y) - len(betas))
    se_betas = np.sqrt(mse * np.diag(np.linalg.inv(X_with_const.T @ X_with_const)))
    t_stats = betas / se_betas
    p_values = 2 * (1 - sp_stats.t.cdf(np.abs(t_stats), len(y) - len(betas)))

    # Annualized alpha
    ann_alpha = alpha_factor * 252

    # --- Display Results ---
    st.markdown("---")
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("R-squared", f"{r_squared:.3f}")
    col_m2.metric("Adj. R-squared", f"{adj_r_squared:.3f}")
    col_m3.metric("Annualized Alpha", f"{ann_alpha:.2%}")
    col_m4.metric("Idiosyncratic Risk", f"{np.std(residuals) * np.sqrt(252):.2%}")

    st.subheader("Regression Results")
    reg_data = [["Factor", "Beta", "Std Error", "t-stat", "p-value", "Significant?"]]
    reg_data.append(["Intercept (Alpha)", f"{alpha_factor:.6f}", f"{se_betas[0]:.6f}",
                     f"{t_stats[0]:.3f}", f"{p_values[0]:.4f}",
                     "***" if p_values[0] < 0.01 else "**" if p_values[0] < 0.05 else "*" if p_values[0] < 0.1 else ""])
    for i, f in enumerate(selected_factors):
        reg_data.append([f, f"{factor_betas[i]:.4f}", f"{se_betas[i+1]:.4f}",
                         f"{t_stats[i+1]:.3f}", f"{p_values[i+1]:.4f}",
                         "***" if p_values[i+1] < 0.01 else "**" if p_values[i+1] < 0.05 else "*" if p_values[i+1] < 0.1 else ""])

    tbl = go.Figure(go.Table(
        header=dict(values=reg_data[0], fill_color="#1a237e", font_color="white", align="center"),
        cells=dict(values=list(zip(*reg_data[1:])), fill_color=[["#f5f5f5", "white"] * 3], align="center"),
    ))
    tbl.update_layout(height=200, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(tbl, width='stretch')

    st.caption("Significance: *** p<0.01, ** p<0.05, * p<0.10")

    # --- Risk Decomposition ---
    st.markdown("---")
    st.subheader("Variance Decomposition")

    systematic_var = np.var(y_pred)
    idiosyncratic_var = np.var(residuals)
    total_var = np.var(y)

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        fig_pie = go.Figure(go.Pie(
            labels=["Systematic Risk", "Idiosyncratic Risk"],
            values=[systematic_var, idiosyncratic_var],
            hole=0.4, textinfo="label+percent",
            marker=dict(colors=["#1565c0", "#EF553B"]),
        ))
        fig_pie.update_layout(height=350, template="plotly_dark",
                              title="Portfolio Variance Decomposition")
        st.plotly_chart(fig_pie, width='stretch')

    with col_d2:
        st.markdown("**Variance Breakdown:**")
        st.write(f"- Total Variance: {total_var:.8f}")
        st.write(f"- Systematic (explained by factors): {systematic_var:.8f} ({systematic_var/total_var:.1%})")
        st.write(f"- Idiosyncratic (unique to portfolio): {idiosyncratic_var:.8f} ({idiosyncratic_var/total_var:.1%})")
        st.write(f"- R-squared: {r_squared:.4f}")
        st.markdown("---")
        st.markdown("**Interpretation:**")
        if r_squared > 0.8:
            st.write("High R-squared: Your portfolio is largely driven by systematic market factors.")
        elif r_squared > 0.5:
            st.write("Moderate R-squared: Mix of systematic and idiosyncratic risk.")
        else:
            st.write("Low R-squared: Your portfolio has significant unique risk not explained by these factors.")

    # --- Factor Contribution Chart ---
    st.subheader("Factor Exposure (Beta Coefficients)")
    fig_beta = go.Figure(go.Bar(
        x=[f.split("(")[0].strip() for f in selected_factors],
        y=factor_betas,
        marker_color=["#1565c0" if b > 0 else "#EF553B" for b in factor_betas],
        text=[f"{b:.3f}" for b in factor_betas],
        textposition="outside",
    ))
    fig_beta.update_layout(
        yaxis_title="Beta (sensitivity)", height=350, template="plotly_dark",
        title="Portfolio Beta to Each Factor",
    )
    st.plotly_chart(fig_beta, width='stretch')

    # --- Actual vs Predicted ---
    st.subheader("Actual vs Factor-Model Predicted Returns")
    cum_actual = (1 + pd.Series(y, index=common_idx)).cumprod()
    cum_predicted = (1 + pd.Series(y_pred, index=common_idx)).cumprod()

    fig_avp = go.Figure()
    fig_avp.add_trace(go.Scatter(x=cum_actual.index, y=cum_actual.values, mode="lines",
                                  name="Actual", line=dict(color="#1565c0", width=2)))
    fig_avp.add_trace(go.Scatter(x=cum_predicted.index, y=cum_predicted.values, mode="lines",
                                  name="Factor Model Predicted", line=dict(color="#EF553B", width=2, dash="dash")))
    fig_avp.update_layout(
        xaxis_title="Date", yaxis_title="Growth of $1",
        height=350, template="plotly_dark", hovermode="x unified",
        title="Cumulative Actual vs Predicted Returns",
    )
    st.plotly_chart(fig_avp, width='stretch')

# =====================================================================
# TAB 3: MONTE CARLO FORWARD SIMULATION
# =====================================================================
with tab3:
    st.header("Monte Carlo Forward Simulation")
    st.markdown("""
    **Monte Carlo simulation** projects future portfolio values by randomly sampling from the 
    historical return distribution thousands of times. Unlike a single-point forecast, it provides
    a **probability distribution** of outcomes, showing best-case, worst-case, and expected scenarios.
    
    Each simulation path compounds daily returns: P(t+1) = P(t) * (1 + r_t), where r_t is 
    randomly drawn from a normal distribution calibrated to your portfolio's historical mean and volatility.
    """)

    col_mc1, col_mc2, col_mc3 = st.columns(3)
    with col_mc1:
        mc_sims = st.slider("Number of Simulations", 100, 10000, 2000, 100)
    with col_mc2:
        mc_days = st.selectbox("Simulation Horizon", [30, 60, 90, 126, 252], index=4,
                                format_func=lambda x: f"{x} days ({x/21:.0f} months)")
    with col_mc3:
        mc_invest = st.number_input("Initial Investment ($)", 1000, 10000000, 100000, 1000, key="mc_inv")

    # Also allow GBM (Geometric Brownian Motion) or simple bootstrap
    sim_method = st.radio("Simulation Method", ["Parametric (Normal)", "Historical Bootstrap"], horizontal=True)

    port_mean_d = port_returns.mean()
    port_std_d = port_returns.std()

    np.random.seed(42)

    if sim_method == "Parametric (Normal)":
        # GBM-style: daily returns ~ N(mu, sigma)
        sim_returns = np.random.normal(port_mean_d, port_std_d, (mc_sims, mc_days))
    else:
        # Bootstrap: sample with replacement from historical returns
        sim_returns = np.random.choice(port_returns.values, size=(mc_sims, mc_days), replace=True)

    # Compound returns
    sim_paths = np.zeros((mc_sims, mc_days + 1))
    sim_paths[:, 0] = mc_invest
    for t in range(1, mc_days + 1):
        sim_paths[:, t] = sim_paths[:, t-1] * (1 + sim_returns[:, t-1])

    final_values = sim_paths[:, -1]
    final_returns = (final_values / mc_invest - 1)

    # --- Statistics ---
    st.markdown("---")
    st.subheader(f"Simulation Results ({mc_sims:,} paths, {mc_days} days)")

    col_s1, col_s2, col_s3, col_s4, col_s5 = st.columns(5)
    col_s1.metric("Median Outcome", f"${np.median(final_values):,.0f}")
    col_s2.metric("Mean Outcome", f"${np.mean(final_values):,.0f}")
    col_s3.metric("5th Percentile", f"${np.percentile(final_values, 5):,.0f}")
    col_s4.metric("95th Percentile", f"${np.percentile(final_values, 95):,.0f}")
    col_s5.metric("Prob of Loss", f"{(final_values < mc_invest).mean():.1%}")

    # Percentile table
    st.subheader("Outcome Distribution")
    pct_data = [
        ["Percentile", "Portfolio Value", "Return"],
        ["Worst Case (1st)", f"${np.percentile(final_values, 1):,.0f}", f"{np.percentile(final_returns, 1):.1%}"],
        ["Bearish (5th)", f"${np.percentile(final_values, 5):,.0f}", f"{np.percentile(final_returns, 5):.1%}"],
        ["Lower (25th)", f"${np.percentile(final_values, 25):,.0f}", f"{np.percentile(final_returns, 25):.1%}"],
        ["Median (50th)", f"${np.percentile(final_values, 50):,.0f}", f"{np.percentile(final_returns, 50):.1%}"],
        ["Upper (75th)", f"${np.percentile(final_values, 75):,.0f}", f"{np.percentile(final_returns, 75):.1%}"],
        ["Bullish (95th)", f"${np.percentile(final_values, 95):,.0f}", f"{np.percentile(final_returns, 95):.1%}"],
        ["Best Case (99th)", f"${np.percentile(final_values, 99):,.0f}", f"{np.percentile(final_returns, 99):.1%}"],
    ]
    pct_tbl = go.Figure(go.Table(
        header=dict(values=pct_data[0], fill_color="#1a237e", font_color="white", align="center"),
        cells=dict(values=list(zip(*pct_data[1:])), fill_color=[["#f5f5f5", "white"] * 4], align="center"),
    ))
    pct_tbl.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(pct_tbl, width='stretch')

    # --- Fan Chart (sample of paths) ---
    st.markdown("---")
    st.subheader("Simulation Paths")
    show_paths = min(200, mc_sims)
    fig_fan = go.Figure()
    for i in range(show_paths):
        fig_fan.add_trace(go.Scatter(
            x=list(range(mc_days + 1)), y=sim_paths[i],
            mode="lines", line=dict(width=0.3, color="rgba(100,150,255,0.1)"),
            showlegend=False, hoverinfo="skip",
        ))

    # Percentile bands
    days_axis = list(range(mc_days + 1))
    p5 = np.percentile(sim_paths, 5, axis=0)
    p25 = np.percentile(sim_paths, 25, axis=0)
    p50 = np.percentile(sim_paths, 50, axis=0)
    p75 = np.percentile(sim_paths, 75, axis=0)
    p95 = np.percentile(sim_paths, 95, axis=0)

    fig_fan.add_trace(go.Scatter(x=days_axis, y=p50, mode="lines", name="Median",
                                  line=dict(color="gold", width=3)))
    fig_fan.add_trace(go.Scatter(x=days_axis, y=p95, mode="lines", name="95th pct",
                                  line=dict(color="lime", width=1.5, dash="dash")))
    fig_fan.add_trace(go.Scatter(x=days_axis, y=p5, mode="lines", name="5th pct",
                                  line=dict(color="red", width=1.5, dash="dash")))
    fig_fan.add_hline(y=mc_invest, line_dash="dot", line_color="white",
                      annotation_text="Break-even")

    # Shaded area between 5th and 95th
    fig_fan.add_trace(go.Scatter(
        x=days_axis + days_axis[::-1], y=list(p95) + list(p5[::-1]),
        fill="toself", fillcolor="rgba(100,150,255,0.15)", line=dict(width=0),
        showlegend=False, hoverinfo="skip",
    ))

    fig_fan.update_layout(
        xaxis_title="Trading Days", yaxis_title="Portfolio Value ($)",
        height=500, template="plotly_dark",
        title=f"Monte Carlo Simulation: {mc_sims:,} paths over {mc_days} days",
    )
    st.plotly_chart(fig_fan, width='stretch')

    # --- Final Value Distribution ---
    st.subheader("Distribution of Final Portfolio Value")
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=final_values, nbinsx=80, name="Final Values",
        marker_color="#1565c0", opacity=0.7,
    ))
    fig_hist.add_vline(x=mc_invest, line_dash="dash", line_color="white",
                       annotation_text="Break-even")
    fig_hist.add_vline(x=np.median(final_values), line_dash="dash", line_color="gold",
                       annotation_text="Median")
    fig_hist.add_vline(x=np.percentile(final_values, 5), line_dash="dash", line_color="red",
                       annotation_text="5th pct")
    fig_hist.update_layout(
        xaxis_title="Portfolio Value ($)", yaxis_title="Frequency",
        height=350, template="plotly_dark",
        title="Histogram of Final Portfolio Values",
    )
    st.plotly_chart(fig_hist, width='stretch')

    st.info("""
    **How to read these results:**
    - The **fan chart** shows individual simulation paths fading into a band; the gold line is the median path.
    - The **dashed lines** show the 5th and 95th percentile -- there is a 90% chance the outcome falls between them.
    - The **histogram** shows the probability distribution of final values.
    - Use the **"Prob of Loss"** metric to understand downside risk over your chosen horizon.
    """)

st.markdown("---")
st.caption("VaR assumes i.i.d. returns. Factor model uses OLS. Monte Carlo assumes log-normal returns (parametric) or i.i.d. bootstrap. For educational purposes only.")
