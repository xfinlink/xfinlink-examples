# Full write-up: https://xfinlink.com/blog/ai-balance-sheet-net-cash-python

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xfinlink as xfl

xfl.set_api_key(os.environ.get("XFINLINK_API_KEY", "YOUR_API_KEY"))  # free at https://xfinlink.com/signup

SLUG = "ai-balance-sheet-net-cash-python"
CHART_PATH = Path("worker/src/site/blog-images") / f"{SLUG}.png"
TICKERS = ["NVDA", "AVGO", "AMD", "PLTR", "MSFT", "META", "GOOG", "AMZN", "ORCL", "ADBE", "INTC", "CRM"]
FIELDS = [
    "revenue",
    "cash_and_equivalents",
    "cash_and_short_term_investments",
    "total_debt",
    "total_assets",
    "free_cash_flow",
    "operating_income",
    "interest_expense",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def fmt_pct(value: float) -> str:
    return f"{value:+6.1%}"


def fmt_pct_plain(value: float) -> str:
    return f"{value:5.1%}"


def fmt_money(value: float) -> str:
    return f"${value / 1000:,.1f}B"


def fmt_interest_coverage(value: float) -> str:
    if pd.isna(value):
        return "No interest expense"
    return f"{value:5.1f}x"


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


df = xfl.fundamentals(TICKERS, period_type="annual", period="3y", fields=FIELDS)
require(not df.empty, "fundamentals returned no rows")
latest = df.sort_values(["ticker", "period_end"]).groupby("ticker", group_keys=False).tail(1).set_index("ticker")
require(set(TICKERS).issubset(set(latest.index)), "missing one or more requested tickers")

screen = latest.copy()
screen["cash_buffer"] = screen["cash_and_short_term_investments"].combine_first(screen["cash_and_equivalents"])
screen["total_debt"] = screen["total_debt"].fillna(0)
required_cols = ["revenue", "cash_buffer", "total_debt", "total_assets", "free_cash_flow", "operating_income"]
require(screen[required_cols].notna().all().all(), "latest balance-sheet data contains missing values")
require((screen["revenue"] > 0).all(), "revenue should be positive for this universe")

screen["net_cash"] = screen["cash_buffer"] - screen["total_debt"]
screen["net_cash_to_revenue"] = screen["net_cash"] / screen["revenue"]
screen["debt_to_assets"] = screen["total_debt"] / screen["total_assets"]
screen["fcf_margin"] = screen["free_cash_flow"] / screen["revenue"]
interest = screen["interest_expense"].abs().replace(0, np.nan)
screen["interest_coverage"] = screen["operating_income"] / interest
screen = screen.sort_values("net_cash_to_revenue", ascending=False)

apply_dark_theme()
fig, ax = plt.subplots(figsize=(10, 5))
colors = ["#3b82f6" if value >= 0 else "#ef4444" for value in screen["net_cash_to_revenue"]]
ax.bar(screen.index, screen["net_cash_to_revenue"] * 100, color=colors)
ax.axhline(0, color="#e0e0e0", linewidth=0.8)
ax.set_title("AI Balance-Sheet Net Cash")
ax.set_xlabel("Company")
ax.set_ylabel("Net cash as share of revenue (%)")
ax.grid(axis="y", color="#2a2a2a", linewidth=0.6, alpha=0.55)
plt.tight_layout()
CHART_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(CHART_PATH, dpi=150, facecolor=fig.get_facecolor())
plt.close(fig)

print("=== AI Balance-Sheet Net Cash Screen ===")
print("Universe: 12 AI platform, software, and semiconductor stocks")
print(f"Latest annual periods: {screen['period_end'].min().date()} to {screen['period_end'].max().date()}")
print(f"Highest net cash / revenue: {screen.index[0]} ({fmt_pct(screen.iloc[0]['net_cash_to_revenue'])})")
print(f"Lowest net cash / revenue: {screen.index[-1]} ({fmt_pct(screen.iloc[-1]['net_cash_to_revenue'])})")
print()
print("Balance-sheet ranking:")
for ticker, row in screen.iterrows():
    print(
        f"{ticker:5s} cash={fmt_money(row['cash_buffer'])}  "
        f"debt={fmt_money(row['total_debt'])}  "
        f"net_cash={fmt_money(row['net_cash'])}  "
        f"net_cash/revenue={fmt_pct(row['net_cash_to_revenue'])}  "
        f"debt/assets={fmt_pct_plain(row['debt_to_assets'])}  "
        f"FCF_margin={fmt_pct_plain(row['fcf_margin'])}  "
        f"interest_cover={fmt_interest_coverage(row['interest_coverage'])}"
    )
