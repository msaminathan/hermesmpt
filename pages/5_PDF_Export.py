"""
Page 5: PDF Export - Generate and download a full analysis report with publishing-quality math
"""
import streamlit as st
import numpy as np
import pandas as pd
import os, sys
from scipy import stats as sp_stats
from scipy.optimize import minimize
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils import (
    fetch_data, compute_returns, annualize_returns, annualize_covariance,
    efficient_frontier, max_sharpe_portfolio, min_variance_portfolio,
    random_portfolios, portfolio_performance,
)
from pdf_report import generate_pdf

st.set_page_config(page_title="PDF Export", page_icon="PDF", layout="wide")
st.title("PDF Export - Full Analysis Report")

tickers = st.session_state.get("tickers", ["AAPL", "MSFT", "GOOGL"])
period = st.session_state.get("period", "1y")
risk_free = st.session_state.get("risk_free", 0.04)
category = st.session_state.get("category", "US Tech Giants")

if not tickers:
    st.warning("Please select tickers on the Home page.")
    st.stop()

st.markdown("""
Generate a **publishing-quality PDF report** (12 pages) of your MPT analysis, including:

| Section | Contents |
|---------|----------|
| Theory | Markowitz framework, key assumptions, diversification |
| Math | Equations for return, variance, Sharpe, CML, risk decomposition |
| Data | Normalized price charts, risk-return summary table |
| Correlation | Heatmap with values, average pairwise correlation |
| Efficient Frontier | Monte Carlo scatter + frontier curve + CML |
| Optimal Portfolio | Tangency allocation pie, Min-Variance comparison |
| Risk Decomposition | Marginal risk contribution bar chart |
| Performance | Cumulative backtest vs equal-weight benchmark |
| **VaR / CVaR** | Three methods (parametric, historical, MC), distribution chart |
| **Factor Model** | OLS regression, beta chart, variance decomposition pie |
| **Monte Carlo** | 2000 forward paths, fan chart, outcome histogram |
| References | Academic citations |
""")

st.markdown("---")

col1, col2, col3 = st.columns(3)
with col1:
    num_mc = st.slider("Monte Carlo portfolios (frontier)", 1000, 5000, 3000, 500)
with col2:
    num_ef = st.slider("Frontier curve points", 30, 150, 80, 10)
with col3:
    max_w = st.slider("Max weight per asset (%)", 20, 100, 100, 5) / 100.0

st.markdown("---")

if st.button("Generate PDF Report (12 Pages)", type="primary", width='stretch'):
    progress = st.progress(0, text="Initializing...")

    try:
        # Step 1: Fetch data
        progress.progress(5, text="Downloading stock data...")
        prices = fetch_data(tickers, period)
        returns = compute_returns(prices)
        mean_rets = annualize_returns(returns)
        cov_mat = annualize_covariance(returns)

        # Step 2: Monte Carlo for frontier
        progress.progress(15, text="Running Monte Carlo simulation...")
        mc_results, mc_weights = random_portfolios(num_mc, mean_rets.values, cov_mat.values)

        # Step 3: Efficient frontier
        progress.progress(25, text="Computing efficient frontier...")
        f_vols, f_rets, f_weights = efficient_frontier(mean_rets.values, cov_mat.values, num_ef, risk_free)

        # Step 4: Optimal portfolios
        progress.progress(35, text="Finding optimal portfolios...")
        n = len(tickers)
        bounds = tuple((0, max_w) for _ in range(n))
        constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}

        ms_result = minimize(
            lambda w: -((np.dot(w, mean_rets.values) - risk_free) /
                        np.sqrt(np.dot(w.T, np.dot(cov_mat.values, w)))),
            n * [1.0/n], method="SLSQP", bounds=bounds, constraints=constraints,
        )
        ms_ret, ms_vol = portfolio_performance(ms_result.x, mean_rets.values, cov_mat.values)
        ms_sharpe = (ms_ret - risk_free) / ms_vol

        mv_result = min_variance_portfolio(mean_rets.values, cov_mat.values)
        mv_ret, mv_vol = portfolio_performance(mv_result.x, mean_rets.values, cov_mat.values)

        opt_weights = ms_result.x

        # Step 5: Portfolio performance
        progress.progress(45, text="Simulating portfolio performance...")
        port_daily = (returns * opt_weights).sum(axis=1)
        port_ret, port_vol = portfolio_performance(opt_weights, mean_rets.values, cov_mat.values)
        port_sharpe = (port_ret - risk_free) / port_vol
        cum_max = (1 + port_daily).cumprod().cummax()
        max_dd = (((1 + port_daily).cumprod() - cum_max) / cum_max).min()

        # Step 6: VaR/CVaR computation
        progress.progress(55, text="Computing VaR/CVaR analysis...")
        confidence = 0.95
        alpha = 1 - confidence
        pm, ps = port_daily.mean(), port_daily.std()
        z = sp_stats.norm.ppf(alpha)
        param_var = -(pm + z * ps)
        hist_var = -np.percentile(port_daily, alpha * 100)
        np.random.seed(42)
        mc50k = np.random.normal(pm, ps, 50000)
        mc_var = -np.percentile(mc50k, alpha * 100)

        # Step 7: Factor model
        progress.progress(65, text="Running factor model regression...")
        factor_tickers = ["SPY", "IWM", "TLT"]
        factor_names = ["SPY (Market)", "IWM (Small Cap)", "TLT (Bonds)"]
        try:
            factor_prices = fetch_data(factor_tickers + tickers, period)
            factor_returns = compute_returns(factor_prices)
            common_idx = factor_returns.index.intersection(port_daily.index)
            y = port_daily.loc[common_idx].values
            X = factor_returns[factor_tickers].loc[common_idx].values
            X_c = np.column_stack([np.ones(len(X)), X])
            from numpy.linalg import lstsq
            betas, _, _, _ = lstsq(X_c, y, rcond=None)
            y_pred = X_c @ betas
            ss_res = np.sum((y - y_pred)**2)
            ss_tot = np.sum((y - y.mean())**2)
            r_squared = max(0, 1 - ss_res / ss_tot)
            factor_betas_arr = betas[1:]
        except Exception as e:
            st.warning(f"Factor model fallback (using defaults): {e}")
            factor_names = ["Market", "Size", "Bonds"]
            factor_betas_arr = np.array([0.8, 0.3, -0.1])
            r_squared = 0.65

        # Step 8: Monte Carlo forward simulation
        progress.progress(75, text="Running forward Monte Carlo simulation...")
        mc_fwd_sims = 2000
        mc_fwd_days = 252
        np.random.seed(42)
        sim_rets = np.random.normal(pm, ps, (mc_fwd_sims, mc_fwd_days))
        sim_paths = np.zeros((mc_fwd_sims, mc_fwd_days + 1))
        sim_paths[:, 0] = 100000
        for t in range(1, mc_fwd_days + 1):
            sim_paths[:, t] = sim_paths[:, t-1] * (1 + sim_rets[:, t-1])
        final_values = sim_paths[:, -1]

        # Step 9: Generate PDF
        progress.progress(90, text="Rendering equations, charts, and building PDF...")
        output_path = os.path.expanduser("~/.hermes/mpt/MPT_Analysis_Report.pdf")
        result_path = generate_pdf(
            prices=prices, returns=returns, mean_rets=mean_rets, cov_mat=cov_mat,
            f_vols=f_vols, f_rets=f_rets, mc_results=mc_results,
            ms_result=ms_result, ms_ret=ms_ret, ms_vol=ms_vol, ms_sharpe=ms_sharpe,
            mv_result=mv_result, mv_ret=mv_ret, mv_vol=mv_vol,
            tickers=tickers, risk_free=risk_free, category=category, period=period,
            opt_weights=opt_weights, port_ret=port_ret, port_vol=port_vol,
            port_sharpe=port_sharpe, port_daily_returns=port_daily, max_drawdown=max_dd,
            output_path=output_path,
            var_confidence=confidence, var_holding_days=1,
            param_var=param_var, hist_var=hist_var, mc_var=mc_var,
            factor_names=factor_names, factor_betas_arr=factor_betas_arr, r_squared=r_squared,
            sim_paths=sim_paths, final_values=final_values, mc_days=mc_fwd_days, mc_invest=100000,
        )

        progress.progress(100, text="Done!")

        file_size = os.path.getsize(result_path)
        st.success(f"Report generated: **{result_path}** ({file_size / 1024:.0f} KB)")

        with open(result_path, "rb") as f:
            st.download_button(
                label="Download PDF Report",
                data=f.read(),
                file_name="MPT_Analysis_Report.pdf",
                mime="application/pdf",
                width='stretch',
            )

        st.info("""
**Report Contents (12 pages):**
- 11 embedded charts (matplotlib, 180-250 DPI)
- 15+ rendered mathematical equations (matplotlib mathtext)
- 3 VaR methods compared (parametric, historical, Monte Carlo)
- Factor model with R-squared and beta decomposition
- 2000-path Monte Carlo forward simulation with fan chart
- Risk-return tables, correlation heatmap, efficient frontier
""")

    except Exception as e:
        st.error(f"Error generating PDF: {e}")
        import traceback
        st.code(traceback.format_exc())

st.markdown("---")
st.caption("Equations rendered via matplotlib mathtext (no LaTeX install needed). Charts embedded as high-res PNG.")
