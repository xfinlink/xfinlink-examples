# Full write-up: https://xfinlink.com/blog/ai-growth-valuation-screen-python

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xfinlink as xfl

xfl.set_api_key(os.environ.get("XFINLINK_API_KEY", "YOUR_API_KEY"))  # free at https://xfinlink.com/signup

SLUG = "ai-growth-valuation-screen-python"
CHART_PATH = Path("worker/src/site/blog-images") / f"{SLUG}.png"
TICKERS = ["NVDA", "AVGO", "AMD", "PLTR", "MSFT", "META", "GOOG", "AMZN", "ORCL", "ADBE"]
FIELDS = ["market_cap", "revenue_growth", "pe_ratio", "price_to_fcf", "fcf_margin"]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def fmt_pct(value: float) -> str:
    return f"{value:+6.1%}"


def fmt_pct_plain(value: float) -> str:
    return f"{value:5.1%}"


def fmt_money(value: float) -> str:
    return f"${value / 1000:,.1f}B"


def apply_dark_theme() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "#0a0a0a",
            "axes.facecolor": "#0a0a0a",
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#e0e0e0",
            "xtick.color": "#e0e0e0",
            "ytick.color": "#e0e0e0",
            "text.color": "#e0e0e0",
            "font.size": 10,
        }
    )


df = xfl.metrics(TICKERS, period_type="ttm", fields=FIELDS)
require(not df.empty, "metrics returned no rows")
latest = df.sort_values("period_end").groupby("ticker", group_keys=False).tail(1).set_index("ticker")
require(set(TICKERS).issubset(set(latest.index)), "missing one or more requested tickers")
require(latest[FIELDS].notna().all().all(), "latest metrics contain missing values")
require((latest["market_cap"] > 1_000).all(), "market cap sanity check failed")
require(((latest["revenue_growth"] > 0) & (latest["pe_ratio"] > 0)).all(), "growth-adjusted P/E requires positive growth and positive P/E")

screen = latest.copy()
screen["growth_adjusted_pe"] = screen["pe_ratio"] / (screen["revenue_growth"] * 100)
screen["fcf_yield"] = 1 / screen["price_to_fcf"]
screen = screen.sort_values("growth_adjusted_pe")

apply_dark_theme()
fig, ax = plt.subplots(figsize=(10, 5))
colors = ["#3b82f6" if value >= 0 else "#ef4444" for value in screen["fcf_yield"]]
sizes = (screen["market_cap"] / screen["market_cap"].max()).clip(lower=0.02) * 650
ax.scatter(screen["revenue_growth"] * 100, screen["pe_ratio"], s=sizes, color=colors, alpha=0.88)
for ticker, row in screen.iterrows():
    ax.annotate(ticker, (row["revenue_growth"] * 100, row["pe_ratio"]), xytext=(5, 5), textcoords="offset points", fontsize=9)
ax.set_title("AI Growth-Adjusted Valuation")
ax.set_xlabel("TTM revenue growth (%)")
ax.set_ylabel("P/E ratio")
ax.grid(axis="y", color="#2a2a2a", linewidth=0.6, alpha=0.55)
plt.tight_layout()
CHART_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(CHART_PATH, dpi=150, facecolor=fig.get_facecolor())
plt.close(fig)

print("=== AI Growth-Adjusted Valuation Screen ===")
print("Universe: 10 AI platform, software, and semiconductor stocks")
print(f"Latest TTM periods: {screen['period_end'].min().date()} to {screen['period_end'].max().date()}")
print(f"Lowest P/E per revenue-growth point: {screen.index[0]} ({screen.iloc[0]['growth_adjusted_pe']:.2f})")
print(f"Highest free-cash-flow yield: {screen['fcf_yield'].idxmax()} ({fmt_pct(screen['fcf_yield'].max())})")
print()
print("Valuation ranking:")
for ticker, row in screen.iterrows():
    print(
        f"{ticker:5s} market_cap={fmt_money(row['market_cap'])}  "
        f"rev_growth={fmt_pct(row['revenue_growth'])}  "
        f"PE={row['pe_ratio']:6.1f}  "
        f"PE/growth_pt={row['growth_adjusted_pe']:5.2f}  "
        f"P/FCF={row['price_to_fcf']:7.1f}  "
        f"FCF_yield={fmt_pct(row['fcf_yield'])}  "
        f"FCF_margin={fmt_pct_plain(row['fcf_margin'])}"
    )
