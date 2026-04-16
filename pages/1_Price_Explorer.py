"""
Page 1: Price Explorer - Historical prices, returns, and correlation analysis
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils import fetch_data, compute_returns, annualize_returns, annualize_covariance

st.set_page_config(page_title="Price Explorer", page_icon="📈", layout="wide")
st.title("Price Explorer")

# Get settings from session
tickers = st.session_state.get("tickers", ["AAPL", "MSFT", "GOOGL"])
period = st.session_state.get("period", "1y")

if not tickers:
    st.warning("Please select at least one ticker on the Home page.")
    st.stop()

# Fetch data
with st.spinner("Fetching data..."):
    prices = fetch_data(tickers, period)

if prices.empty:
    st.error("No data returned. Check your tickers.")
    st.stop()

returns = compute_returns(prices)
ann_ret = annualize_returns(returns)
ann_vol = returns.std() * (252 ** 0.5)

st.success(f"Loaded {len(prices)} trading days for {len(prices.columns)} assets")

# ─── Tabs ────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["Price History", "Normalized", "Returns", "Correlation"])

with tab1:
    st.subheader("Historical Prices")
    fig = go.Figure()
    for col in prices.columns:
        fig.add_trace(go.Scatter(x=prices.index, y=prices[col], mode="lines", name=col))
    fig.update_layout(
        xaxis_title="Date", yaxis_title="Price ($)",
        hovermode="x unified", height=500,
        template="plotly_dark",
    )
    st.plotly_chart(fig, width='stretch')

with tab2:
    st.subheader("Normalized Prices (Base = 100)")
    normalized = prices / prices.iloc[0] * 100
    fig = go.Figure()
    for col in normalized.columns:
        fig.add_trace(go.Scatter(x=normalized.index, y=normalized[col], mode="lines", name=col))
    fig.update_layout(
        xaxis_title="Date", yaxis_title="Indexed Value (Start = 100)",
        hovermode="x unified", height=500,
        template="plotly_dark",
    )
    st.plotly_chart(fig, width='stretch')
    st.info("Normalized view makes it easy to compare performance across assets with different price levels.")

with tab3:
    st.subheader("Daily Log Returns Distribution")
    col_sel = st.multiselect("Select assets to compare", tickers, default=tickers[:3])
    if col_sel:
        fig = make_subplots(rows=len(col_sel), cols=1, shared_xaxes=True,
                            subplot_titles=col_sel, vertical_spacing=0.05)
        for i, t in enumerate(col_sel, 1):
            if t in returns.columns:
                fig.add_trace(
                    go.Scatter(x=returns.index, y=returns[t], mode="lines",
                              name=t, line=dict(width=0.8)),
                    row=i, col=1
                )
        fig.update_layout(height=150 * len(col_sel), showlegend=False,
                          template="plotly_dark", title_text="Daily Returns")
        st.plotly_chart(fig, width='stretch')

    st.subheader("Annualized Risk-Return Summary")
    summary = {
        "Ticker": tickers,
        "Ann. Return": [f"{ann_ret[t]:.1%}" for t in tickers],
        "Ann. Volatility": [f"{ann_vol[t]:.1%}" for t in tickers],
        "Sharpe (rf=4%)": [f"{(ann_ret[t] - 0.04) / ann_vol[t]:.2f}" if ann_vol[t] > 0 else "N/A" for t in tickers],
    }
    st.dataframe(summary, width='stretch', hide_index=True)

with tab4:
    st.subheader("Correlation Matrix")
    corr = returns.corr()
    fig = px.imshow(
        corr, text_auto=".2f", color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1, aspect="auto",
        title="Correlation of Daily Returns",
    )
    fig.update_layout(height=max(400, len(tickers) * 45), template="plotly_dark")
    st.plotly_chart(fig, width='stretch')

    st.markdown("""
    **Reading the correlation matrix:**
    - **+1.0** = Perfect positive correlation (assets move together)
    - **0.0** = No correlation (independent movement)
    - **-1.0** = Perfect negative correlation (assets move opposite)
    - Lower correlations = better diversification benefits
    """)
