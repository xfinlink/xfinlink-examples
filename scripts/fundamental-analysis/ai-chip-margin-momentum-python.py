# Full write-up: https://xfinlink.com/blog/ai-chip-margin-momentum-python

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import xfinlink as xfl

xfl.set_api_key(os.environ.get("XFINLINK_API_KEY", "YOUR_API_KEY"))  # free at https://xfinlink.com/signup

SLUG = "ai-chip-margin-momentum-python"
CHART_PATH = Path("worker/src/site/blog-images") / f"{SLUG}.png"
TICKERS = ["NVDA", "AVGO", "AMD", "MU", "QCOM", "INTC", "AMAT", "LRCX"]
FIELDS = ["revenue", "gross_profit", "operating_income", "free_cash_flow"]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def fmt_pct(value: float) -> str:
    return f"{value:+6.1%}"


def fmt_pct_plain(value: float) -> str:
    return f"{value:5.1%}"


def fmt_pp(value: float) -> str:
    return f"{value * 100:+5.1f}pp"


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


df = xfl.fundamentals(TICKERS, period_type="annual", period="5y", fields=FIELDS)
require(not df.empty, "fundamentals returned no rows")
df = df.sort_values(["ticker", "period_end"])

rows = []
for ticker, group in df.groupby("ticker"):
    group = group.dropna(subset=FIELDS).sort_values("period_end")
    require(len(group) >= 2, f"{ticker} needs at least two annual observations")
    latest = group.iloc[-1]
    prior = group.iloc[-2]
    require(latest["revenue"] > 0 and prior["revenue"] > 0, f"{ticker} revenue must be positive")
    latest_gross_margin = latest["gross_profit"] / latest["revenue"]
    prior_gross_margin = prior["gross_profit"] / prior["revenue"]
    rows.append(
        {
            "ticker": ticker,
            "period_end": latest["period_end"],
            "revenue": latest["revenue"],
            "revenue_growth": latest["revenue"] / prior["revenue"] - 1,
            "gross_margin": latest_gross_margin,
            "gross_margin_change": latest_gross_margin - prior_gross_margin,
            "operating_margin": latest["operating_income"] / latest["revenue"],
            "fcf_margin": latest["free_cash_flow"] / latest["revenue"],
        }
    )

screen = pd.DataFrame(rows).sort_values("gross_margin_change", ascending=False)
require(set(TICKERS).issubset(set(screen["ticker"])), "missing one or more chip tickers")
require(screen[["revenue_growth", "gross_margin", "gross_margin_change", "operating_margin", "fcf_margin"]].notna().all().all(), "screen contains missing values")

apply_dark_theme()
fig, ax = plt.subplots(figsize=(10, 5))
colors = ["#3b82f6" if value >= 0 else "#ef4444" for value in screen["gross_margin_change"]]
ax.bar(screen["ticker"], screen["gross_margin_change"] * 100, color=colors)
ax.axhline(0, color="#e0e0e0", linewidth=0.8)
ax.set_title("AI Chip Margin Momentum")
ax.set_xlabel("Company")
ax.set_ylabel("Gross margin change, percentage points")
ax.grid(axis="y", color="#2a2a2a", linewidth=0.6, alpha=0.55)
plt.tight_layout()
CHART_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(CHART_PATH, dpi=150, facecolor=fig.get_facecolor())
plt.close(fig)

print("=== AI Chip Margin Momentum Screen ===")
print("Universe: 8 semiconductor and semiconductor-equipment stocks")
print(f"Latest annual periods: {screen['period_end'].min().date()} to {screen['period_end'].max().date()}")
print(f"Best gross-margin expansion: {screen.iloc[0]['ticker']} {fmt_pp(screen.iloc[0]['gross_margin_change'])}")
print(f"Median revenue growth: {fmt_pct(screen['revenue_growth'].median())}")
print()
print("Margin ranking:")
for _, row in screen.iterrows():
    print(
        f"{row['ticker']:5s} revenue={fmt_money(row['revenue'])}  "
        f"rev_growth={fmt_pct(row['revenue_growth'])}  "
        f"gross_margin={fmt_pct_plain(row['gross_margin'])}  "
        f"gross_margin_change={fmt_pp(row['gross_margin_change'])}  "
        f"op_margin={fmt_pct_plain(row['operating_margin'])}  "
        f"FCF_margin={fmt_pct_plain(row['fcf_margin'])}"
    )
