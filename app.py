"""
Modern Portfolio Theory - Interactive Streamlit App
Main entry point (Home page)
"""
import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from utils import TICKER_CATEGORIES

st.set_page_config(
    page_title="Modern Portfolio Theory",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Sidebar: Ticker Selection ───────────────────────────────────────
st.sidebar.title("Portfolio Settings")

category = st.sidebar.selectbox(
    "Select Asset Category",
    list(TICKER_CATEGORIES.keys()),
    index=0,
    help="Choose a predefined set of assets or use Custom to enter your own tickers.",
)

if category == "Custom (edit below)":
    custom_input = st.sidebar.text_area(
        "Enter tickers (comma-separated)",
        value="AAPL, MSFT, GOOGL, AMZN, NVDA",
        height=100,
    )
    tickers = [t.strip().upper() for t in custom_input.split(",") if t.strip()]
else:
    tickers = TICKER_CATEGORIES[category]
    st.sidebar.write(f"**{len(tickers)} assets selected**")

# Show selected tickers
if tickers:
    st.sidebar.markdown("**Tickers:** " + ", ".join(tickers))

period = st.sidebar.selectbox(
    "Historical Period",
    ["6mo", "1y", "2y", "5y", "10y"],
    index=1,
)

risk_free = st.sidebar.slider(
    "Risk-Free Rate (%)", 0.0, 10.0, 4.0, 0.5,
    help="Annual risk-free rate (e.g., US Treasury yield)"
) / 100.0

# Store in session state for other pages
st.session_state["tickers"] = tickers
st.session_state["period"] = period
st.session_state["risk_free"] = risk_free
st.session_state["category"] = category

# ─── Home Page Content ───────────────────────────────────────────────
st.title("Modern Portfolio Theory Explorer")
st.markdown("---")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    ## Welcome!

    This interactive app helps you understand **Modern Portfolio Theory (MPT)**,
    developed by Harry Markowitz in 1952.

    ### What You Can Do

    Navigate through the pages in the sidebar:

    1. **Price Explorer** -- View historical prices, returns, and correlations
    2. **Efficient Frontier** -- Visualize the risk-return tradeoff and find optimal portfolios
    3. **Portfolio Optimizer** -- Find the Maximum Sharpe Ratio and Minimum Variance portfolios
    4. **Portfolio Simulator** -- Allocate custom weights and see projected performance

    ### Key Concepts

    - **Expected Return**: The mean return an investor anticipates
    - **Risk (Volatility)**: Standard deviation of returns -- higher means more uncertainty
    - **Diversification**: Combining assets with imperfect correlations reduces overall risk
    - **Efficient Frontier**: The set of portfolios offering maximum return for each level of risk
    - **Sharpe Ratio**: Risk-adjusted return: (Return - Risk-Free Rate) / Volatility
    - **Tangency Portfolio**: The portfolio on the efficient frontier with the highest Sharpe Ratio

    ### Quick Start

    1. Pick an **asset category** in the sidebar (or enter custom tickers)
    2. Adjust the **time period** and **risk-free rate**
    3. Navigate to the pages to explore!
    """)

with col2:
    st.info(f"""
    **Current Selection**
    - Category: {category}
    - Assets: {len(tickers)}
    - Period: {period}
    - Risk-Free Rate: {risk_free:.1%}
    """)

    st.markdown("""
    ### The Formula

    **Portfolio Return:**
    ```
    E(Rp) = sum(wi * E(Ri))
    ```

    **Portfolio Variance:**
    ```
    sigma^2 = sum(sum(wi * wj * sigma_ij))
    ```

    **Sharpe Ratio:**
    ```
    S = (E(Rp) - Rf) / sigma_p
    ```
    """)

st.markdown("---")
st.caption("Built with Streamlit | Data from Yahoo Finance | Based on Markowitz (1952) MPT")
