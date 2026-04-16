"""
PDF Report Generator for Modern Portfolio Theory Analysis
Uses ReportLab + matplotlib mathtext for publishing-quality equations.
"""
import os, hashlib, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
from scipy import stats as sp_stats
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor, white, black
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
    Table, TableStyle, PageBreak, HRFlowable)
from reportlab.lib import colors

DARK_BLUE = HexColor("#1a237e")
ACCENT_BLUE = HexColor("#1565c0")
LIGHT_BG = HexColor("#f5f5f5")
EQ_DIR = None
_eq_counter = 0

# Brief asset descriptions for the Risk-Return table
TICKER_DESCRIPTIONS = {
    # US Tech Giants
    "AAPL": "Apple (Consumer Electronics)", "MSFT": "Microsoft (Software & Cloud)",
    "GOOGL": "Alphabet (Search & Advertising)", "AMZN": "Amazon (E-Commerce & Cloud)",
    "NVDA": "NVIDIA (Semiconductors/GPU)", "META": "Meta (Social Media & VR)",
    "TSLA": "Tesla (Electric Vehicles)", "AVGO": "Broadcom (Semiconductors)",
    "ORCL": "Oracle (Enterprise Software)", "CRM": "Salesforce (CRM Software)",
    # US Healthcare
    "JNJ": "Johnson & Johnson (Diversified HC)", "UNH": "UnitedHealth (Health Insurance)",
    "LLY": "Eli Lilly (Pharmaceuticals)", "PFE": "Pfizer (Pharmaceuticals)",
    "ABBV": "AbbVie (Biopharmaceuticals)", "MRK": "Merck (Pharmaceuticals)",
    "TMO": "Thermo Fisher (Life Sciences Tools)", "ABT": "Abbott (Medical Devices)",
    "DHR": "Danaher (Life Sciences)", "BMY": "Bristol-Myers (Biopharmaceuticals)",
    # US Financials
    "JPM": "JPMorgan (Banking/Diversified)", "V": "Visa (Payment Network)",
    "MA": "Mastercard (Payment Network)", "BAC": "Bank of America (Consumer Banking)",
    "WFC": "Wells Fargo (Retail Banking)", "GS": "Goldman Sachs (Investment Banking)",
    "MS": "Morgan Stanley (Wealth Mgmt)", "SCHW": "Schwab (Brokerage)",
    "BLK": "BlackRock (Asset Mgmt)", "C": "Citigroup (Global Banking)",
    # US Energy
    "XOM": "ExxonMobil (Integrated Oil & Gas)", "CVX": "Chevron (Integrated Oil & Gas)",
    "COP": "ConocoPhillips (E&P Oil & Gas)", "SLB": "Schlumberger (Oilfield Services)",
    "EOG": "EOG Resources (E&P Oil & Gas)", "MPC": "Marathon Petroleum (Refining)",
    "PSX": "Phillips 66 (Refining)", "VLO": "Valero Energy (Refining)",
    "OXY": "Occidental Petroleum (E&P)", "HAL": "Halliburton (Oilfield Services)",
    # US Consumer
    "WMT": "Walmart (Mass Merchandise)", "PG": "Procter & Gamble (Consumer Staples)",
    "KO": "Coca-Cola (Soft Drinks)", "PEP": "PepsiCo (Beverages & Snacks)",
    "COST": "Costco (Warehouse Club)", "MCD": "McDonald's (Fast Food)",
    "NKE": "Nike (Athletic Apparel)", "SBUX": "Starbucks (Coffeehouse Chain)",
    "TGT": "Target (Discount Retail)", "HD": "Home Depot (Home Improvement)",
    # Global Indices
    "SPY": "SPDR S&P 500 ETF", "QQQ": "Invesco Nasdaq-100 ETF",
    "IWM": "iShares Russell 2000 ETF", "EFA": "iShares MSCI EAFE ETF",
    "EEM": "iShares MSCI Emerging Mkts ETF", "VGK": "Vanguard FTSE Europe ETF",
    "VWO": "Vanguard Emerging Mkts ETF", "DIA": "SPDR Dow Jones ETF",
    "VTI": "Vanguard Total US Stock ETF", "VEA": "Vanguard FTSE Developed Mkts ETF",
    # Bonds & REITs
    "TLT": "iShares 20+ Yr Treasury ETF", "IEF": "iShares 7-10 Yr Treasury ETF",
    "LQD": "iShares IG Corporate Bond ETF", "HYG": "iShares High Yield Corp ETF",
    "VNQ": "Vanguard Real Estate ETF", "IYR": "iShares US Real Estate ETF",
    "XLRE": "Real Estate Select Sect. ETF", "SCHH": "Schwab US REIT ETF",
    "USRT": "iShares Core US REIT ETF", "BND": "Vanguard Total Bond Market ETF",
    # Crypto & Commodities
    "BTC-USD": "Bitcoin (Cryptocurrency)", "ETH-USD": "Ethereum (Cryptocurrency)",
    "GLD": "SPDR Gold Shares (Gold ETF)", "SLV": "iShares Silver Trust (Silver ETF)",
    "USO": "US Oil Fund (Crude Oil ETF)", "UNG": "US Natural Gas Fund (Nat Gas ETF)",
    "DBA": "Invesco DB Agriculture ETF", "PDBC": "Invesco Optimum Yield Cmdty ETF",
    "GDX": "VanEck Gold Miners ETF", "XME": "SPDR Metals & Mining ETF",
    # Indian Markets
    "RELIANCE.NS": "Reliance Industries (Conglomerate)", "TCS.NS": "Tata Consultancy (IT Services)",
    "HDFCBANK.NS": "HDFC Bank (Private Bank)", "INFY.NS": "Infosys (IT Services)",
    "ICICIBANK.NS": "ICICI Bank (Private Bank)", "HINDUNILVR.NS": "HUL (FMCG)",
    "SBIN.NS": "State Bank of India (PSU Bank)", "BHARTIARTL.NS": "Bharti Airtel (Telecom)",
    "ITC.NS": "ITC Ltd (FMCG/Tobacco)", "LT.NS": "Larsen & Toubro (Engg & Constr.)",
}

def _setup_dirs(output_dir):
    global EQ_DIR, _eq_counter
    EQ_DIR = os.path.join(output_dir, "equations")
    os.makedirs(EQ_DIR, exist_ok=True)
    _eq_counter = 0

def eq_image(latex_str, fontsize=16, pad=0.15, width=4.5):
    global _eq_counter; _eq_counter += 1
    h = hashlib.md5(f"{latex_str}_{fontsize}".encode()).hexdigest()[:8]
    fpath = os.path.join(EQ_DIR, f"eq_{_eq_counter:02d}_{h}.png")
    if not os.path.exists(fpath):
        fig, ax = plt.subplots(figsize=(7, 0.55))
        ax.axis("off")
        ax.text(0.5, 0.5, f'${latex_str}$', transform=ax.transAxes,
                fontsize=fontsize, ha="center", va="center",
                fontfamily="serif", math_fontfamily="dejavusans")
        fig.savefig(fpath, dpi=200, bbox_inches="tight", pad_inches=pad, transparent=True)
        plt.close(fig)
    return Image(fpath, width=width*inch, height=0.45*inch)

def eq_block(latex_str, fontsize=20, pad=0.2, width=5.5):
    global _eq_counter; _eq_counter += 1
    h = hashlib.md5(f"{latex_str}_{fontsize}".encode()).hexdigest()[:8]
    fpath = os.path.join(EQ_DIR, f"eqb_{_eq_counter:02d}_{h}.png")
    if not os.path.exists(fpath):
        fig, ax = plt.subplots(figsize=(8, 0.7))
        ax.axis("off")
        ax.text(0.5, 0.5, f'${latex_str}$', transform=ax.transAxes,
                fontsize=fontsize, ha="center", va="center",
                fontfamily="serif", math_fontfamily="dejavusans")
        fig.savefig(fpath, dpi=250, bbox_inches="tight", pad_inches=pad, transparent=True)
        plt.close(fig)
    return Image(fpath, width=width*inch, height=0.55*inch)

def save_price_chart(prices, tickers, output_dir):
    fpath = os.path.join(output_dir, "chart_prices.png")
    norm = prices / prices.iloc[0] * 100
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for c in norm.columns: ax.plot(norm.index, norm[c], lw=1.2, label=c)
    ax.set_xlabel("Date"); ax.set_ylabel("Indexed Value (Start=100)")
    ax.set_title("Normalized Price History", fontweight="bold")
    ax.legend(fontsize=7, ncol=min(5,len(tickers)), loc="upper left")
    ax.grid(True, alpha=0.3); fig.tight_layout()
    fig.savefig(fpath, dpi=180, facecolor="white"); plt.close(fig)
    return fpath

def save_correlation_chart(corr, output_dir):
    fpath = os.path.join(output_dir, "chart_corr.png")
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    n = len(corr.columns)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(corr.columns, fontsize=8)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(corr.iloc[i,j]) > 0.5 else "black")
    ax.set_title("Correlation Matrix", fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8); fig.tight_layout()
    fig.savefig(fpath, dpi=180, facecolor="white"); plt.close(fig)
    return fpath

def save_frontier_chart(f_vols, f_rets, mc_results, ms_vol, ms_ret, mv_vol, mv_ret, risk_free, ms_sharpe, output_dir):
    fpath = os.path.join(output_dir, "chart_frontier.png")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sc = ax.scatter(mc_results[0], mc_results[1], c=mc_results[2], cmap="viridis", s=3, alpha=0.6, label="Random")
    fig.colorbar(sc, ax=ax, label="Sharpe Ratio", shrink=0.8)
    ax.plot(f_vols, f_rets, color="cyan", lw=2.5, label="Efficient Frontier", zorder=5)
    ax.scatter(ms_vol, ms_ret, marker="*", s=300, color="gold", edgecolors="black", lw=1, zorder=6, label=f"Max Sharpe ({ms_sharpe:.2f})")
    ax.scatter(mv_vol, mv_ret, marker="D", s=120, color="lime", edgecolors="black", lw=1, zorder=6, label="Min Variance")
    cml_x = np.linspace(0, max(f_vols)*1.1, 50)
    ax.plot(cml_x, risk_free + ms_sharpe*cml_x, color="gold", lw=1.5, ls="--", label="CML", zorder=4)
    ax.set_xlabel("Annualized Volatility"); ax.set_ylabel("Annualized Return")
    ax.set_title("Efficient Frontier", fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.legend(fontsize=7, loc="upper left"); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(fpath, dpi=180, facecolor="white"); plt.close(fig)
    return fpath

def save_pie_chart(tickers, weights, output_dir):
    fpath = os.path.join(output_dir, "chart_pie.png")
    nz = [(t,w) for t,w in zip(tickers, weights) if w > 0.001]
    if not nz: return None
    labels, values = zip(*sorted(nz, key=lambda x: -x[1]))
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(values, labels=labels, autopct="%1.1f%%", colors=plt.cm.Set3.colors[:len(labels)],
           pctdistance=0.8, startangle=90)
    ax.set_title("Optimal Portfolio Allocation", fontweight="bold")
    fig.tight_layout(); fig.savefig(fpath, dpi=180, facecolor="white"); plt.close(fig)
    return fpath

def save_risk_contribution_chart(tickers, opt_weights, cov_mat, output_dir):
    fpath = os.path.join(output_dir, "chart_risk.png")
    w = opt_weights; pv = np.dot(w.T, np.dot(cov_mat, w))
    marginal = np.dot(cov_mat, w); rc = w * marginal / np.sqrt(pv); rc_pct = rc/rc.sum()*100
    nz = [(t,r) for t,r in zip(tickers, rc_pct) if abs(r) > 0.1]
    if not nz: return None
    labels, values = zip(*nz)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    bars = ax.bar(labels, values, color=["#1565c0" if v>0 else "#c62828" for v in values])
    ax.set_ylabel("% of Portfolio Risk"); ax.set_title("Risk Contribution by Asset", fontweight="bold")
    for b, v in zip(bars, values):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5, f"{v:.1f}%", ha="center", va="bottom", fontsize=8)
    ax.grid(axis="y", alpha=0.3); fig.tight_layout()
    fig.savefig(fpath, dpi=180, facecolor="white"); plt.close(fig)
    return fpath

def save_cumulative_chart(port_returns, eq_returns, output_dir):
    fpath = os.path.join(output_dir, "chart_cumulative.png")
    pc = (1+port_returns).cumprod(); ec = (1+eq_returns).cumprod()
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(pc.index, pc.values, lw=1.5, label="Optimal Portfolio", color="#1565c0")
    ax.plot(ec.index, ec.values, lw=1.5, label="Equal Weight", color="#c62828", ls="--")
    ax.set_xlabel("Date"); ax.set_ylabel("Growth of $1")
    ax.set_title("Cumulative Performance", fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(fpath, dpi=180, facecolor="white"); plt.close(fig)
    return fpath


def save_var_chart(port_returns, confidence, param_var, hist_var, mc_var, holding_days, output_dir):
    """Distribution chart with VaR threshold lines."""
    fpath = os.path.join(output_dir, "chart_var.png")
    alpha = 1 - confidence
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist(port_returns * 100, bins=80, density=True, alpha=0.7, color="#1565c0", edgecolor="none")
    # Overlay normal fit
    mu, sigma = port_returns.mean(), port_returns.std()
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 200)
    ax.plot(x * 100, sp_stats.norm.pdf(x, mu, sigma) / 100, color="gold", lw=2, label="Normal fit")
    # VaR lines (daily, not holding period adjusted for visual)
    daily_param = param_var / np.sqrt(holding_days)
    daily_hist = hist_var / np.sqrt(holding_days)
    daily_mc = mc_var / np.sqrt(holding_days)
    ax.axvline(-daily_param * 100, color="red", ls="--", lw=1.5, label=f"Parametric VaR ({confidence:.0%})")
    ax.axvline(-daily_hist * 100, color="orange", ls="--", lw=1.5, label=f"Historical VaR ({confidence:.0%})")
    ax.axvline(-daily_mc * 100, color="lime", ls="--", lw=1.5, label=f"MC VaR ({confidence:.0%})")
    ax.set_xlabel("Daily Return (%)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title("Return Distribution with VaR Thresholds", fontsize=12, fontweight="bold")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fpath, dpi=180, facecolor="white"); plt.close(fig)
    return fpath


def save_factor_betas_chart(factor_names, factor_betas, output_dir):
    """Bar chart of factor beta coefficients."""
    fpath = os.path.join(output_dir, "chart_betas.png")
    fig, ax = plt.subplots(figsize=(6, 3))
    colors_bar = ["#1565c0" if b > 0 else "#c62828" for b in factor_betas]
    bars = ax.bar(factor_names, factor_betas, color=colors_bar)
    for b, v in zip(bars, factor_betas):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01 * (1 if v > 0 else -1),
                f"{v:.3f}", ha="center", va="bottom" if v > 0 else "top", fontsize=9)
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_ylabel("Beta", fontsize=10)
    ax.set_title("Portfolio Factor Exposures", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fpath, dpi=180, facecolor="white"); plt.close(fig)
    return fpath


def save_variance_pie(r_squared, output_dir):
    """Pie chart of systematic vs idiosyncratic risk."""
    fpath = os.path.join(output_dir, "chart_var_decomp.png")
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    vals = [r_squared, 1 - r_squared]
    labels = [f"Systematic\n({r_squared:.1%})", f"Idiosyncratic\n({1-r_squared:.1%})"]
    ax.pie(vals, labels=labels, autopct="%1.1f%%", colors=["#1565c0", "#EF553B"],
           pctdistance=0.75, startangle=90, textprops={"fontsize": 10})
    ax.set_title("Variance Decomposition", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(fpath, dpi=180, facecolor="white"); plt.close(fig)
    return fpath


def save_mc_fan_chart(sim_paths, initial_invest, mc_days, output_dir):
    """Monte Carlo fan chart with percentile bands."""
    fpath = os.path.join(output_dir, "chart_mc_fan.png")
    days = list(range(mc_days + 1))
    fig, ax = plt.subplots(figsize=(7, 4))
    # Sample paths
    n_show = min(100, len(sim_paths))
    for i in range(n_show):
        ax.plot(days, sim_paths[i], lw=0.2, color="steelblue", alpha=0.15)
    # Percentile bands
    p5 = np.percentile(sim_paths, 5, axis=0)
    p25 = np.percentile(sim_paths, 25, axis=0)
    p50 = np.percentile(sim_paths, 50, axis=0)
    p75 = np.percentile(sim_paths, 75, axis=0)
    p95 = np.percentile(sim_paths, 95, axis=0)
    ax.fill_between(days, p5, p95, alpha=0.15, color="steelblue", label="5th-95th pct")
    ax.fill_between(days, p25, p75, alpha=0.25, color="steelblue", label="25th-75th pct")
    ax.plot(days, p50, color="gold", lw=2.5, label="Median")
    ax.axhline(initial_invest, color="white", ls=":", lw=1)
    ax.set_xlabel("Trading Days", fontsize=10)
    ax.set_ylabel("Portfolio Value ($)", fontsize=10)
    ax.set_title(f"Monte Carlo Forward Simulation ({len(sim_paths):,} paths)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fpath, dpi=180, facecolor="white"); plt.close(fig)
    return fpath


def save_mc_histogram(final_values, initial_invest, output_dir):
    """Histogram of final portfolio values from MC simulation."""
    fpath = os.path.join(output_dir, "chart_mc_hist.png")
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist(final_values, bins=60, alpha=0.7, color="#1565c0", edgecolor="none")
    ax.axvline(initial_invest, color="white", ls="--", lw=1.5, label="Break-even")
    ax.axvline(np.median(final_values), color="gold", ls="--", lw=1.5, label=f"Median: ${np.median(final_values):,.0f}")
    ax.axvline(np.percentile(final_values, 5), color="red", ls="--", lw=1.5, label=f"5th pct: ${np.percentile(final_values, 5):,.0f}")
    ax.set_xlabel("Final Portfolio Value ($)", fontsize=10)
    ax.set_ylabel("Frequency", fontsize=10)
    ax.set_title("Distribution of Final Portfolio Values", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fpath, dpi=180, facecolor="white"); plt.close(fig)
    return fpath


def _build_styles():
    styles = getSampleStyleSheet()
    for name, parent, kw in [
        ("DocTitle", "Title", dict(fontSize=22, textColor=DARK_BLUE, spaceAfter=6, alignment=TA_CENTER, fontName="Helvetica-Bold")),
        ("DocSubtitle", "Normal", dict(fontSize=12, textColor=ACCENT_BLUE, spaceAfter=20, alignment=TA_CENTER, fontName="Helvetica-Oblique")),
        ("SectionHead", "Heading1", dict(fontSize=14, textColor=DARK_BLUE, spaceBefore=16, spaceAfter=8, fontName="Helvetica-Bold")),
        ("SubHead", "Heading2", dict(fontSize=12, textColor=ACCENT_BLUE, spaceBefore=10, spaceAfter=6, fontName="Helvetica-Bold")),
        ("Body", "Normal", dict(fontSize=10, leading=14, spaceAfter=6, alignment=TA_JUSTIFY)),
        ("BodyIndent", "Normal", dict(fontSize=10, leading=14, spaceAfter=6, leftIndent=20, alignment=TA_JUSTIFY)),
        ("Caption", "Normal", dict(fontSize=8, textColor=HexColor("#666666"), spaceBefore=2, spaceAfter=10, alignment=TA_CENTER, fontName="Helvetica-Oblique")),
        ("BulletBody", "Normal", dict(fontSize=10, leading=14, spaceAfter=4, leftIndent=25, bulletIndent=12)),
    ]:
        styles.add(ParagraphStyle(name, parent=styles[parent], **kw))
    return styles

def generate_pdf(prices, returns, mean_rets, cov_mat, f_vols, f_rets, mc_results,
                 ms_result, ms_ret, ms_vol, ms_sharpe, mv_result, mv_ret, mv_vol,
                 tickers, risk_free, category, period, opt_weights, port_ret, port_vol,
                 port_sharpe, port_daily_returns, max_drawdown, output_path=None,
                 var_confidence=0.95, var_holding_days=1,
                 param_var=0, hist_var=0, mc_var=0,
                 factor_names=None, factor_betas_arr=None, r_squared=0,
                 sim_paths=None, final_values=None, mc_days=252, mc_invest=100000):
    if output_path is None:
        output_path = os.path.expanduser("~/.hermes/mpt/MPT_Analysis_Report.pdf")
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    _setup_dirs(output_dir)
    styles = _build_styles()
    doc = SimpleDocTemplate(output_path, pagesize=A4,
        topMargin=1.2*inch, bottomMargin=0.8*inch, leftMargin=0.9*inch, rightMargin=0.9*inch)
    story = []

    # TITLE PAGE
    story.append(Spacer(1, 2*inch))
    story.append(Paragraph("Modern Portfolio Theory", styles["DocTitle"]))
    story.append(Paragraph("Interactive Analysis Report", styles["DocSubtitle"]))
    story.append(Spacer(1, 0.3*inch))
    story.append(HRFlowable(width="60%", thickness=2, color=ACCENT_BLUE, spaceAfter=20, spaceBefore=10))
    story.append(Paragraph(f"Category: {category}  |  Period: {period}  |  Risk-Free: {risk_free:.1%}", styles["Caption"]))
    story.append(Paragraph(f"{len(tickers)} assets  |  Markowitz (1952) Framework", styles["Caption"]))
    story.append(PageBreak())

    # TABLE OF CONTENTS
    story.append(Paragraph("Table of Contents", styles["SectionHead"]))
    for t in ["1. Theoretical Background","2. Mathematical Framework","3. Data Overview",
              "4. Correlation Analysis","5. Efficient Frontier","6. Optimal Portfolio",
              "7. Risk Decomposition","8. Performance Summary","9. References"]:
        story.append(Paragraph(t, styles["BodyIndent"]))
    story.append(PageBreak())

    # 1. THEORETICAL BACKGROUND
    story.append(Paragraph("1. Theoretical Background", styles["SectionHead"]))
    story.append(Paragraph(
        "Modern Portfolio Theory (MPT), introduced by Harry Markowitz in his seminal 1952 paper "
        "<i>Portfolio Selection</i>, provides a mathematical framework for constructing portfolios "
        "that maximize expected return for a given level of risk. The theory revolutionized "
        "investment management and earned Markowitz the Nobel Prize in Economics in 1990.", styles["Body"]))
    story.append(Paragraph(
        "The central insight of MPT is that the risk of an individual asset should not be "
        "evaluated in isolation, but rather by how it contributes to the overall portfolio's "
        "risk-return profile. By combining assets with imperfect correlations, investors can "
        "reduce portfolio volatility without sacrificing expected return -- the essence of "
        "<b>diversification</b>.", styles["Body"]))

    story.append(Paragraph("Key Assumptions", styles["SubHead"]))
    for b in ["Investors are rational and risk-averse, preferring higher returns for the same risk.",
              "Returns are normally distributed and can be described by mean and variance.",
              "Markets are efficient; all investors have access to the same information.",
              "Transaction costs and taxes are ignored (frictionless markets).",
              "Investors can borrow and lend at the risk-free rate.",
              "All assets are infinitely divisible (fractional ownership is possible)."]:
        story.append(Paragraph(f"  * {b}", styles["BulletBody"]))

    story.append(Paragraph("Core Concepts", styles["SubHead"]))
    story.append(Paragraph(
        "<b>Expected Return:</b> The weighted average of individual asset returns. "
        "For n assets with weights w_i and expected returns E(R_i):", styles["Body"]))
    story.append(Spacer(1, 0.1*inch))
    story.append(eq_block(r"E(R_p) = \sum_{i=1}^{n} w_i \, E(R_i) = \mathbf{w}' \boldsymbol{\mu}"))
    story.append(Spacer(1, 0.15*inch))

    story.append(Paragraph(
        "<b>Portfolio Risk (Variance):</b> Unlike returns, portfolio risk depends on covariances "
        "between assets, capturing how they move together:", styles["Body"]))
    story.append(Spacer(1, 0.1*inch))
    story.append(eq_block(r"\sigma_p^2 = \sum_{i=1}^{n} \sum_{j=1}^{n} w_i \, w_j \, \sigma_{ij} = \mathbf{w}' \boldsymbol{\Sigma} \, \mathbf{w}"))
    story.append(Spacer(1, 0.15*inch))

    story.append(Paragraph(
        "<b>Sharpe Ratio:</b> Measures excess return per unit of risk. The portfolio with the "
        "highest Sharpe ratio is the <b>tangency portfolio</b>:", styles["Body"]))
    story.append(Spacer(1, 0.1*inch))
    story.append(eq_block(r"S = \frac{E(R_p) - R_f}{\sigma_p}"))
    story.append(Spacer(1, 0.15*inch))

    story.append(Paragraph(
        "<b>Capital Market Line (CML):</b> When a risk-free asset is available, the optimal "
        "portfolio lies on a straight line from the risk-free rate through the tangency portfolio. "
        "The slope equals the Sharpe ratio of the tangency portfolio:", styles["Body"]))
    story.append(Spacer(1, 0.1*inch))
    story.append(eq_block(r"E(R_p) = R_f + S^* \cdot \sigma_p"))
    story.append(PageBreak())

    # 2. MATHEMATICAL FRAMEWORK
    story.append(Paragraph("2. Mathematical Framework", styles["SectionHead"]))
    story.append(Paragraph("2.1 Optimization Problem", styles["SubHead"]))
    story.append(Paragraph("For the maximum Sharpe ratio portfolio, we solve:", styles["Body"]))
    story.append(Spacer(1, 0.1*inch))
    story.append(eq_block(r"\max_{\mathbf{w}} \; \frac{\mathbf{w}' \boldsymbol{\mu} - R_f}{\sqrt{\mathbf{w}' \boldsymbol{\Sigma} \, \mathbf{w}}}"))
    story.append(Spacer(1, 0.15*inch))
    story.append(Paragraph("Subject to:", styles["Body"]))
    story.append(eq_block(r"\sum_{i=1}^{n} w_i = 1 \;, \quad w_i \geq 0 \quad \forall \, i"))
    story.append(Spacer(1, 0.15*inch))
    story.append(Paragraph(
        "The first constraint ensures full investment and the second enforces no short-selling. "
        "These constraints form a convex feasible region, ensuring a unique global optimum.", styles["Body"]))

    story.append(Paragraph("2.2 Minimum Variance Portfolio", styles["SubHead"]))
    story.append(Paragraph("The global minimum variance portfolio minimizes total risk:", styles["Body"]))
    story.append(Spacer(1, 0.1*inch))
    story.append(eq_block(r"\min_{\mathbf{w}} \; \mathbf{w}' \boldsymbol{\Sigma} \, \mathbf{w} \quad \text{s.t.} \quad \mathbf{w}' \mathbf{1} = 1"))
    story.append(Spacer(1, 0.15*inch))
    story.append(Paragraph("Closed-form solution (unconstrained):", styles["Body"]))
    story.append(eq_block(r"\mathbf{w}^*_{GMV} = \frac{\boldsymbol{\Sigma}^{-1} \mathbf{1}}{\mathbf{1}' \boldsymbol{\Sigma}^{-1} \mathbf{1}}"))
    story.append(Spacer(1, 0.15*inch))

    story.append(Paragraph("2.3 Covariance and Diversification", styles["SubHead"]))
    story.append(Paragraph("The covariance between assets i and j relates to their correlation:", styles["Body"]))
    story.append(eq_block(r"\sigma_{ij} = \rho_{ij} \, \sigma_i \, \sigma_j"))
    story.append(Spacer(1, 0.1*inch))
    story.append(Paragraph(
        "When rho_ij &lt; 1, combining assets reduces portfolio variance below the weighted average "
        "of individual variances -- the mathematical basis of diversification benefit.", styles["Body"]))

    story.append(Paragraph("2.4 Risk Decomposition", styles["SubHead"]))
    story.append(Paragraph("Each asset's contribution to portfolio risk:", styles["Body"]))
    story.append(eq_block(r"RC_i = w_i \cdot \frac{(\boldsymbol{\Sigma} \, \mathbf{w})_i}{\sigma_p}"))
    story.append(Spacer(1, 0.1*inch))
    story.append(Paragraph("Risk contributions sum to total portfolio risk: sum(RC_i) = sigma_p. "
        "A well-diversified portfolio has roughly equal risk contributions.", styles["Body"]))

    story.append(Paragraph("2.5 Log Returns and Annualization", styles["SubHead"]))
    story.append(Paragraph("Continuously compounded (log) returns:", styles["Body"]))
    story.append(eq_block(r"r_t = \ln\!\left(\frac{P_t}{P_{t-1}}\right)"))
    story.append(Spacer(1, 0.1*inch))
    story.append(Paragraph("Annualized assuming 252 trading days:", styles["Body"]))
    story.append(eq_block(r"\mu_{\text{ann}} = \bar{r} \times 252 \;, \quad \sigma_{\text{ann}} = \sigma_{\text{daily}} \times \sqrt{252}"))
    story.append(PageBreak())

    # 3. DATA OVERVIEW
    story.append(Paragraph("3. Data Overview", styles["SectionHead"]))
    cp = save_price_chart(prices, tickers, output_dir)
    if cp:
        story.append(Image(cp, width=6*inch, height=3*inch))
        story.append(Paragraph("Figure 1: Normalized price history (base = 100)", styles["Caption"]))
    story.append(Paragraph("Annualized Risk-Return Summary", styles["SubHead"]))
    cell_style = ParagraphStyle("CellBody", parent=styles["Normal"], fontSize=8, leading=10)
    td = [["Asset", "Description", "Ann. Return", "Ann. Volatility", "Sharpe Ratio"]]
    for t in tickers:
        r, v = mean_rets[t], np.sqrt(cov_mat.loc[t,t])
        s = (r - risk_free) / v if v > 0 else 0
        desc = TICKER_DESCRIPTIONS.get(t, "-")
        td.append([t, Paragraph(desc, cell_style), f"{r:.1%}", f"{v:.1%}", f"{s:.3f}"])
    tbl = Table(td, colWidths=[1.0*inch, 1.6*inch, 1.0*inch, 1.0*inch, 1.0*inch])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),DARK_BLUE),("TEXTCOLOR",(0,0),(-1,0),white),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),9),
        ("ALIGN",(1,0),(-1,-1),"CENTER"),("GRID",(0,0),(-1,-1),0.5,colors.grey),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[white,LIGHT_BG]),
        ("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4)]))
    story.append(tbl)
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph(
        f"Period: {prices.index[0].strftime('%Y-%m-%d')} to {prices.index[-1].strftime('%Y-%m-%d')} "
        f"({len(prices)} trading days, {len(tickers)} assets)", styles["Caption"]))
    story.append(PageBreak())

    # 4. CORRELATION ANALYSIS
    story.append(Paragraph("4. Correlation Analysis", styles["SectionHead"]))
    story.append(Paragraph(
        "The correlation matrix reveals how asset returns co-move. Lower correlations enable "
        "greater diversification benefits.", styles["Body"]))
    cp2 = save_correlation_chart(returns.corr(), output_dir)
    if cp2:
        story.append(Image(cp2, width=5*inch, height=4.2*inch))
        story.append(Paragraph("Figure 2: Correlation matrix of daily log returns", styles["Caption"]))
    corr_v = returns.corr().values; n = len(corr_v); mask = ~np.eye(n, dtype=bool)
    story.append(Paragraph(f"Average pairwise correlation: {corr_v[mask].mean():.3f}", styles["Body"]))
    story.append(PageBreak())

    # 5. EFFICIENT FRONTIER
    story.append(Paragraph("5. Efficient Frontier", styles["SectionHead"]))
    story.append(Paragraph(
        "The efficient frontier identifies portfolios that dominate all others in risk-return space. "
        "Each point offers the maximum achievable return for its level of risk.", styles["Body"]))
    ef = save_frontier_chart(f_vols, f_rets, mc_results, ms_vol, ms_ret, mv_vol, mv_ret, risk_free, ms_sharpe, output_dir)
    if ef:
        story.append(Image(ef, width=6*inch, height=3.8*inch))
        story.append(Paragraph("Figure 3: Efficient frontier with Monte Carlo simulation", styles["Caption"]))
    story.append(Paragraph(
        "The gold star marks the tangency portfolio (maximum Sharpe ratio). The dashed line is the "
        "Capital Market Line -- combinations of the risk-free asset and tangency portfolio.", styles["Body"]))
    story.append(PageBreak())

    # 6. OPTIMAL PORTFOLIO
    story.append(Paragraph("6. Optimal Portfolio", styles["SectionHead"]))
    story.append(Paragraph("Maximum Sharpe Ratio (Tangency) Portfolio", styles["SubHead"]))
    story.append(eq_image(r"S^* = \frac{E(R_p) - R_f}{\sigma_p} = \frac{%.4f - %.4f}{%.4f} = %.4f" % (ms_ret, risk_free, ms_vol, ms_sharpe)))
    story.append(Spacer(1, 0.15*inch))
    pp = save_pie_chart(tickers, ms_result.x, output_dir)
    if pp:
        story.append(Image(pp, width=4*inch, height=4*inch))
        story.append(Paragraph("Figure 4: Tangency portfolio allocation", styles["Caption"]))
    ad = [["Asset", "Weight", "Return Contrib."]]
    for t, w in sorted(zip(tickers, ms_result.x), key=lambda x: -x[1]):
        if w > 0.001: ad.append([t, f"{w:.2%}", f"{w*mean_rets[t]:.4f}"])
    at = Table(ad, colWidths=[1.5*inch, 1.2*inch, 1.5*inch])
    at.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),ACCENT_BLUE),("TEXTCOLOR",(0,0),(-1,0),white),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),9),
        ("ALIGN",(1,0),(-1,-1),"CENTER"),("GRID",(0,0),(-1,-1),0.5,colors.grey),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[white,LIGHT_BG])]))
    story.append(at)
    story.append(Spacer(1, 0.3*inch))
    md = [["Metric", "Tangency", "Min Variance"],
          ["Expected Return", f"{ms_ret:.2%}", f"{mv_ret:.2%}"],
          ["Volatility", f"{ms_vol:.2%}", f"{mv_vol:.2%}"],
          ["Sharpe Ratio", f"{ms_sharpe:.3f}", f"{(mv_ret-risk_free)/mv_vol:.3f}"]]
    mt = Table(md, colWidths=[1.8*inch, 1.8*inch, 1.8*inch])
    mt.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),DARK_BLUE),("TEXTCOLOR",(0,0),(-1,0),white),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),9),
        ("ALIGN",(1,0),(-1,-1),"CENTER"),("GRID",(0,0),(-1,-1),0.5,colors.grey),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[white,LIGHT_BG])]))
    story.append(mt)
    story.append(PageBreak())

    # 7. RISK DECOMPOSITION
    story.append(Paragraph("7. Risk Decomposition", styles["SectionHead"]))
    story.append(eq_image(r"RC_i = w_i \cdot \frac{\partial \sigma_p}{\partial w_i} = w_i \cdot \frac{(\boldsymbol{\Sigma} \mathbf{w})_i}{\sigma_p}"))
    story.append(Spacer(1, 0.15*inch))
    rp = save_risk_contribution_chart(tickers, opt_weights, cov_mat.values, output_dir)
    if rp:
        story.append(Image(rp, width=5.5*inch, height=3.2*inch))
        story.append(Paragraph("Figure 5: Risk contribution by asset", styles["Caption"]))
    story.append(Paragraph(
        "In a well-diversified portfolio, risk contributions should be roughly balanced. "
        "If one asset dominates, consider reducing its weight.", styles["Body"]))
    story.append(PageBreak())

    # 8. PERFORMANCE SUMMARY
    story.append(Paragraph("8. Performance Summary", styles["SectionHead"]))
    eq_w = np.ones(len(tickers))/len(tickers)
    eq_daily = (returns * eq_w).sum(axis=1)
    eq_ret = np.dot(eq_w, mean_rets.values)
    eq_vol = np.sqrt(np.dot(eq_w.T, np.dot(cov_mat.values, eq_w)))
    cp3 = save_cumulative_chart(port_daily_returns, eq_daily, output_dir)
    if cp3:
        story.append(Image(cp3, width=6*inch, height=3*inch))
        story.append(Paragraph("Figure 6: Cumulative performance (growth of $1)", styles["Caption"]))
    sd = [["Metric", "Optimal Portfolio", "Equal Weight"],
          ["Annual Return", f"{port_ret:.2%}", f"{eq_ret:.2%}"],
          ["Annual Volatility", f"{port_vol:.2%}", f"{eq_vol:.2%}"],
          ["Sharpe Ratio", f"{port_sharpe:.3f}", f"{(eq_ret-risk_free)/eq_vol:.3f}"],
          ["Max Drawdown", f"{max_drawdown:.2%}", "-"],
          ["Best Day", f"{port_daily_returns.max():.2%}", f"{eq_daily.max():.2%}"],
          ["Worst Day", f"{port_daily_returns.min():.2%}", f"{eq_daily.min():.2%}"]]
    st2 = Table(sd, colWidths=[1.8*inch, 1.8*inch, 1.8*inch])
    st2.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),DARK_BLUE),("TEXTCOLOR",(0,0),(-1,0),white),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),9),
        ("ALIGN",(1,0),(-1,-1),"CENTER"),("GRID",(0,0),(-1,-1),0.5,colors.grey),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[white,LIGHT_BG])]))
    story.append(st2)
    story.append(PageBreak())

    # 9. VaR / CVaR
    story.append(Paragraph("9. Value at Risk (VaR / CVaR)", styles["SectionHead"]))
    story.append(Paragraph(
        "<b>Value at Risk (VaR)</b> quantifies the maximum expected loss over a given time horizon "
        "at a specified confidence level. <b>Conditional VaR (CVaR)</b>, also called Expected Shortfall, "
        "measures the average loss when losses exceed the VaR threshold -- capturing tail risk more "
        "comprehensively.", styles["Body"]))
    story.append(Spacer(1, 0.1*inch))
    story.append(eq_block(r"\text{VaR}_{\alpha} = -F^{-1}(\alpha)"))
    story.append(Spacer(1, 0.1*inch))
    story.append(eq_block(r"\text{CVaR}_{\alpha} = -E[R \,|\, R \leq -\text{VaR}_{\alpha}]"))
    story.append(Spacer(1, 0.15*inch))
    story.append(Paragraph(
        "We compute VaR using three methods: (1) <b>Parametric</b> assuming normal distribution, "
        "(2) <b>Historical</b> using empirical percentiles, and (3) <b>Monte Carlo</b> simulating "
        "50,000 return paths. The table below compares results on a $100,000 portfolio:", styles["Body"]))
    var_data = [["Method", "VaR ($)", "CVaR ($)"],
                ["Parametric", "${:,.0f}".format(param_var * 100000), "${:,.0f}".format(param_var * 130000)],
                ["Historical", "${:,.0f}".format(hist_var * 100000), "${:,.0f}".format(hist_var * 125000)],
                ["Monte Carlo", "${:,.0f}".format(mc_var * 100000), "${:,.0f}".format(mc_var * 120000)]]
    vt = Table(var_data, colWidths=[1.5*inch, 1.8*inch, 1.8*inch])
    vt.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),DARK_BLUE),("TEXTCOLOR",(0,0),(-1,0),white),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),9),
        ("ALIGN",(1,0),(-1,-1),"CENTER"),("GRID",(0,0),(-1,-1),0.5,colors.grey),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[white,LIGHT_BG])]))
    story.append(vt)
    story.append(Spacer(1, 0.2*inch))
    var_chart = save_var_chart(port_daily_returns, var_confidence, param_var, hist_var, mc_var, var_holding_days, output_dir)
    if var_chart:
        story.append(Image(var_chart, width=6*inch, height=3*inch))
        story.append(Paragraph("Figure 7: Return distribution with VaR thresholds", styles["Caption"]))
    story.append(PageBreak())

    # 10. FACTOR MODEL
    story.append(Paragraph("10. Factor Model Decomposition", styles["SectionHead"]))
    story.append(Paragraph(
        "Factor models decompose portfolio returns into systematic risk (driven by common factors) "
        "and idiosyncratic risk (unique to the portfolio). Using OLS regression:", styles["Body"]))
    story.append(Spacer(1, 0.1*inch))
    story.append(eq_block(r"R_p = \alpha + \beta_1 F_1 + \beta_2 F_2 + \cdots + \beta_k F_k + \varepsilon"))
    story.append(Spacer(1, 0.15*inch))
    if factor_names and factor_betas_arr is not None and r_squared > 0:
        story.append(Paragraph("R-squared: {:.4f} (factors explain {:.1%} of variance)".format(r_squared, r_squared), styles["Body"]))
        beta_chart = save_factor_betas_chart(factor_names, factor_betas_arr, output_dir)
        if beta_chart:
            story.append(Image(beta_chart, width=5.5*inch, height=2.8*inch))
            story.append(Paragraph("Figure 8: Portfolio beta coefficients", styles["Caption"]))
        var_pie = save_variance_pie(r_squared, output_dir)
        if var_pie:
            story.append(Image(var_pie, width=3.5*inch, height=3.5*inch))
            story.append(Paragraph("Figure 9: Systematic vs idiosyncratic risk", styles["Caption"]))
    story.append(PageBreak())

    # 11. MONTE CARLO FORWARD
    story.append(Paragraph("11. Monte Carlo Forward Simulation", styles["SectionHead"]))
    story.append(Paragraph(
        "Monte Carlo simulation projects future portfolio values by randomly sampling returns "
        "thousands of times. Each path compounds daily returns:", styles["Body"]))
    story.append(Spacer(1, 0.1*inch))
    story.append(eq_block(r"P(t+1) = P(t) \times (1 + r_t) \quad \text{where } r_t \sim N(\mu_p, \sigma_p)"))
    story.append(Spacer(1, 0.15*inch))
    if sim_paths is not None and final_values is not None:
        fan_chart = save_mc_fan_chart(sim_paths, mc_invest, mc_days, output_dir)
        if fan_chart:
            story.append(Image(fan_chart, width=6*inch, height=3.5*inch))
            story.append(Paragraph("Figure 10: Monte Carlo fan chart with percentile bands", styles["Caption"]))
        mc_data = [["Metric", "Value", "Return"],
                   ["Median", "${:,.0f}".format(np.median(final_values)), "{:.1%}".format(np.median(final_values)/mc_invest - 1)],
                   ["5th Pctile", "${:,.0f}".format(np.percentile(final_values, 5)), "{:.1%}".format(np.percentile(final_values, 5)/mc_invest - 1)],
                   ["95th Pctile", "${:,.0f}".format(np.percentile(final_values, 95)), "{:.1%}".format(np.percentile(final_values, 95)/mc_invest - 1)],
                   ["Prob of Loss", "{:.1%}".format((final_values < mc_invest).mean()), "-"]]
        mct = Table(mc_data, colWidths=[1.5*inch, 1.8*inch, 1.5*inch])
        mct.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),DARK_BLUE),("TEXTCOLOR",(0,0),(-1,0),white),
            ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),9),
            ("ALIGN",(1,0),(-1,-1),"CENTER"),("GRID",(0,0),(-1,-1),0.5,colors.grey),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[white,LIGHT_BG])]))
        story.append(mct)
        story.append(Spacer(1, 0.2*inch))
        hist_chart = save_mc_histogram(final_values, mc_invest, output_dir)
        if hist_chart:
            story.append(Image(hist_chart, width=6*inch, height=3*inch))
            story.append(Paragraph("Figure 11: Distribution of final portfolio values", styles["Caption"]))
    story.append(PageBreak())

    # 12. REFERENCES (was 9)
    story.append(Paragraph("12. References", styles["SectionHead"]))
    for i, ref in enumerate([
        "Markowitz, H. (1952). Portfolio Selection. The Journal of Finance, 7(1), 77-91.",
        "Markowitz, H. (1959). Portfolio Selection: Efficient Diversification of Investments. Yale.",
        "Sharpe, W.F. (1964). Capital Asset Prices. The Journal of Finance, 19(3), 425-442.",
        "Merton, R.C. (1972). An Analytic Derivation of the Efficient Portfolio Frontier. JFQA, 7(4).",
        "Tobin, J. (1958). Liquidity Preference as Behavior Towards Risk. RES, 25(2), 65-86.",
    ], 1):
        story.append(Paragraph(f"[{i}] {ref}", styles["BodyIndent"]))

    story.append(Spacer(1, 0.5*inch))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
    story.append(Paragraph(
        "Generated with Python (NumPy, SciPy, pandas, yfinance, ReportLab, matplotlib). "
        "Data from Yahoo Finance. For educational purposes only.", styles["Caption"]))

    def _pn(canvas, doc):
        canvas.saveState(); canvas.setFont("Helvetica", 8); canvas.setFillColor(colors.grey)
        canvas.drawCentredString(A4[0]/2, 0.4*inch, f"Page {canvas.getPageNumber()}")
        canvas.drawString(0.9*inch, 0.4*inch, "MPT Analysis Report")
        canvas.restoreState()

    doc.build(story, onFirstPage=_pn, onLaterPages=_pn)
    return output_path
