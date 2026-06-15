# Full write-up: https://xfinlink.com/blog/self-funding-growth-quality-python

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import xfinlink as xfl

xfl.set_api_key(os.environ.get("XFINLINK_API_KEY", "YOUR_API_KEY"))  # free at https://xfinlink.com/signup

SLUG = "self-funding-growth-quality-python"
TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOG", "AVGO", "CRM", "ORCL", "ADBE", "AMD", "QCOM"]
FIELDS = [
    "market_cap",
    "revenue_growth",
    "gross_margin",
    "fcf_margin",
    "cash_to_debt",
    "pe_ratio",
]


def fmt_pct(value: float) -> str:
    return f"{value * 100:6.1f}%"


def fmt_money_m(value: float) -> str:
    return f"${value / 1_000:,.0f}B"


def zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if std == 0 or pd.isna(std):
        return series * 0
    return (series - series.mean()) / std


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
    sizes = (screen["market_cap"] / screen["market_cap"].max()).clip(0.08, 1.0) * 900
    scatter = ax.scatter(
        screen["revenue_growth"] * 100,
        screen["fcf_margin"] * 100,
        c=screen["quality_score"],
        s=sizes,
        cmap="Blues",
        edgecolor="#e0e0e0",
        linewidth=0.5,
        alpha=0.9,
    )
    for _, row in screen.nlargest(5, "quality_score").iterrows():
        ax.annotate(row["ticker"], (row["revenue_growth"] * 100, row["fcf_margin"] * 100),
                    textcoords="offset points", xytext=(6, 5), fontsize=9)
    ax.axhline(0, color="#e0e0e0", linewidth=1, alpha=0.35)
    ax.axvline(0, color="#e0e0e0", linewidth=1, alpha=0.35)
    ax.set_title("Self-funding growth quality screen")
    ax.set_xlabel("TTM revenue growth")
    ax.set_ylabel("Free-cash-flow margin")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Quality score")
    cbar.ax.yaxis.set_tick_params(color="#e0e0e0")
    plt.setp(cbar.ax.get_yticklabels(), color="#e0e0e0")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    plt.tight_layout()
    out = Path("worker/src/site/blog-images") / f"{SLUG}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, facecolor="#0a0a0a")
    plt.close(fig)


metrics = xfl.metrics(TICKERS, period_type="ttm", fields=FIELDS, max_rows=1000)
if metrics.empty:
    raise ValueError("Metrics DataFrame is empty")

required_metrics = {"ticker", "period_end", *FIELDS}
missing_metrics = required_metrics - set(metrics.columns)
if missing_metrics:
    raise ValueError(f"Missing metrics columns: {sorted(missing_metrics)}")

latest = metrics.sort_values("period_end").groupby("ticker").tail(1).copy()
core_fields = ["market_cap", "revenue_growth", "fcf_margin", "cash_to_debt", "pe_ratio"]
latest = latest.dropna(subset=core_fields)
if latest.empty:
    raise ValueError("No complete rows after dropping missing metrics")
if len(latest) < 8:
    raise ValueError(f"Complete-data universe is too small: {len(latest)}")
if (latest["market_cap"] <= 10_000).any():
    raise ValueError("Market cap should be above $10B for this large-cap universe")
if not latest["revenue_growth"].between(-1.0, 2.0).all():
    raise ValueError("Revenue growth contains implausible values")
if not latest["fcf_margin"].between(-1.0, 1.2).all():
    raise ValueError("Free-cash-flow margin contains implausible values")
if (latest["pe_ratio"] <= 0).any():
    raise ValueError("P/E ratios should be positive for this screen")

latest["gross_margin"] = latest["gross_margin"].fillna(latest["gross_margin"].median())
latest["cash_to_debt_capped"] = latest["cash_to_debt"].clip(lower=0, upper=10)
latest["pe_ratio_capped"] = latest["pe_ratio"].clip(lower=1, upper=100)
latest["quality_score"] = (
    zscore(latest["revenue_growth"])
    + zscore(latest["fcf_margin"])
    + 0.5 * zscore(latest["gross_margin"])
    + 0.5 * zscore(latest["cash_to_debt_capped"])
    - 0.25 * zscore(latest["pe_ratio_capped"])
)

if latest["quality_score"].isna().any():
    raise ValueError("Quality score contains NaN values")

ranking = latest.sort_values("quality_score", ascending=False)
make_chart(ranking)

print("=== Self-Funding Growth Quality Screen ===")
print(f"Universe: {len(ranking)} large-cap technology and platform stocks")
print(f"Latest TTM period range: {ranking['period_end'].min().date()} to {ranking['period_end'].max().date()}")
print(f"Top score: {ranking.iloc[0]['ticker']} ({ranking.iloc[0]['quality_score']:5.2f})")
print(f"Weakest score: {ranking.iloc[-1]['ticker']} ({ranking.iloc[-1]['quality_score']:5.2f})")
print()
print("Self-funding growth ranking:")
for _, row in ranking.iterrows():
    print(
        f"{row['ticker']:<5} score={row['quality_score']:6.2f}  "
        f"rev_growth={fmt_pct(row['revenue_growth'])}  "
        f"FCF_margin={fmt_pct(row['fcf_margin'])}  "
        f"cash/debt={row['cash_to_debt']:5.2f}  "
        f"PE={row['pe_ratio']:5.1f}  "
        f"market_cap={fmt_money_m(row['market_cap'])}"
    )
