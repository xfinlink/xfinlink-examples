# Full write-up: https://xfinlink.com/blog/dividend-cash-flow-stress-python

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import xfinlink as xfl

xfl.set_api_key(os.environ.get("XFINLINK_API_KEY", "YOUR_API_KEY"))  # free at https://xfinlink.com/signup

SLUG = "dividend-cash-flow-stress-python"
TICKERS = ["T", "VZ", "MO", "PM", "KO", "PEP", "PG", "WMT", "DUK", "XOM", "CVX"]
FIELDS = ["dividend_yield", "dividend_payout_ratio", "debt_to_ebitda", "fcf_margin", "pe_ratio"]


def zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if std == 0:
        return series * 0
    return (series - series.mean()) / std


def fmt_pct(value: float) -> str:
    return f"{value * 100:6.1f}%"


def make_chart(screen: pd.DataFrame) -> None:
    plt.rcParams.update({
        "figure.facecolor": "#0a0a0a",
        "axes.facecolor": "#0a0a0a",
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#e0e0e0",
        "xtick.color": "#e0e0e0",
        "ytick.color": "#e0e0e0",
        "text.color": "#e0e0e0",
        "axes.titleweight": "bold",
        "font.size": 10,
    })

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#3b82f6" if score >= 0 else "#6b7280" for score in screen["resilience_score"]]
    ax.scatter(
        screen["dividend_yield"] * 100,
        screen["dividend_payout_ratio"] * 100,
        s=(screen["fcf_margin"].clip(lower=0, upper=0.35) + 0.05) * 420,
        color=colors,
        alpha=0.88,
    )
    for ticker, row in screen.iterrows():
        ax.annotate(ticker, (row["dividend_yield"] * 100, row["dividend_payout_ratio"] * 100),
                    xytext=(5, 5), textcoords="offset points", fontsize=9)
    ax.axhline(100, color="#e0e0e0", linewidth=1, alpha=0.45)
    ax.set_title("Dividend yield versus payout stress")
    ax.set_xlabel("Dividend yield")
    ax.set_ylabel("Dividend payout ratio")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    plt.tight_layout()
    out = Path("worker/src/site/blog-images") / f"{SLUG}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, facecolor="#0a0a0a")
    plt.close(fig)


metrics = xfl.metrics(TICKERS, period_type="ttm", fields=FIELDS, max_rows=5000)
if metrics.empty:
    raise ValueError("Metrics DataFrame is empty")

required = {"ticker", "period_end", *FIELDS}
missing_cols = required - set(metrics.columns)
if missing_cols:
    raise ValueError(f"Missing metrics columns: {sorted(missing_cols)}")

latest = metrics.sort_values("period_end").groupby("ticker").tail(1).set_index("ticker")
screen = latest[FIELDS + ["period_end"]].dropna().copy()
if len(screen) != len(TICKERS):
    missing = sorted(set(TICKERS) - set(screen.index))
    raise ValueError(f"Complete dividend metrics missing for: {missing}")
if (screen["dividend_yield"] <= 0).any():
    raise ValueError("Dividend yield should be positive for the dividend universe")
if (screen["pe_ratio"] <= 0).any():
    raise ValueError("P/E ratio should be positive for the dividend universe")

screen["resilience_score"] = (
    zscore(screen["dividend_yield"])
    - zscore(screen["dividend_payout_ratio"])
    - zscore(screen["debt_to_ebitda"])
    + zscore(screen["fcf_margin"])
)

make_chart(screen)

ranked = screen.sort_values("resilience_score", ascending=False)
best = ranked.index[0]
weakest = ranked.index[-1]
highest_yield = screen["dividend_yield"].idxmax()

print("=== Dividend Cash-Flow Stress Screen ===")
print(f"Latest TTM periods: {screen['period_end'].min().date()} to {screen['period_end'].max().date()}")
print(f"Complete-data universe: {len(screen)} dividend stocks")
print(f"Highest yield: {highest_yield} ({fmt_pct(screen.loc[highest_yield, 'dividend_yield'])})")
print(f"Best resilience score: {best} ({ranked.loc[best, 'resilience_score']:5.2f})")
print(f"Weakest resilience score: {weakest} ({ranked.loc[weakest, 'resilience_score']:5.2f})")
print()
print("Dividend stress ranking:")
for ticker, row in ranked.iterrows():
    print(
        f"{ticker:<5} score={row['resilience_score']:6.2f}  "
        f"yield={fmt_pct(row['dividend_yield'])}  "
        f"payout={fmt_pct(row['dividend_payout_ratio'])}  "
        f"debt/EBITDA={row['debt_to_ebitda']:5.2f}  "
        f"FCF_margin={fmt_pct(row['fcf_margin'])}  "
        f"PE={row['pe_ratio']:5.1f}"
    )
