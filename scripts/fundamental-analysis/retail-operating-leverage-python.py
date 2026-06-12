# Full write-up: https://xfinlink.com/blog/retail-operating-leverage-python

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import xfinlink as xfl

xfl.set_api_key(os.environ.get("XFINLINK_API_KEY", "YOUR_API_KEY"))  # free at https://xfinlink.com/signup

SLUG = "retail-operating-leverage-python"
TICKERS = ["WMT", "COST", "TGT", "HD", "LOW", "LULU", "ROST", "BJ", "DLTR", "DG", "KR"]
FIELDS = ["revenue", "operating_income", "gross_profit", "free_cash_flow"]


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
    colors = ["#3b82f6" if value > 0 else "#6b7280" for value in screen["operating_margin_delta"]]
    ax.scatter(
        screen["revenue_growth"] * 100,
        screen["operating_margin_delta"] * 100,
        s=(screen["free_cash_flow_margin"].clip(lower=0, upper=0.15) + 0.04) * 560,
        color=colors,
        alpha=0.88,
    )
    for ticker, row in screen.iterrows():
        ax.annotate(ticker, (row["revenue_growth"] * 100, row["operating_margin_delta"] * 100),
                    xytext=(5, 5), textcoords="offset points", fontsize=9)
    ax.axhline(0, color="#e0e0e0", linewidth=1, alpha=0.45)
    ax.axvline(0, color="#e0e0e0", linewidth=1, alpha=0.45)
    ax.set_title("Retail operating leverage")
    ax.set_xlabel("Latest annual revenue growth")
    ax.set_ylabel("Operating margin change")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    plt.tight_layout()
    out = Path("worker/src/site/blog-images") / f"{SLUG}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, facecolor="#0a0a0a")
    plt.close(fig)


fund = xfl.fundamentals(TICKERS, period_type="annual", period="6y", fields=FIELDS, max_rows=5000)
if fund.empty:
    raise ValueError("Fundamentals DataFrame is empty")

required = {"ticker", "period_end", *FIELDS}
missing_cols = required - set(fund.columns)
if missing_cols:
    raise ValueError(f"Missing fundamentals columns: {sorted(missing_cols)}")

fund = fund.sort_values(["ticker", "period_end"])
recent = fund.groupby("ticker").tail(2)
latest = recent.groupby("ticker").tail(1).set_index("ticker")
prior = recent.groupby("ticker").head(1).set_index("ticker")
missing_tickers = sorted(set(TICKERS) - set(latest.index))
if missing_tickers:
    raise ValueError(f"Missing latest annual observations: {missing_tickers}")

required_latest = latest[FIELDS].join(prior[FIELDS], lsuffix="_latest", rsuffix="_prior")
if required_latest.isna().any().any():
    raise ValueError("Latest or prior retail fundamentals contain NaN values")
if (latest["revenue"] <= 0).any() or (prior["revenue"] <= 0).any():
    raise ValueError("Revenue should be positive for the retail universe")

screen = pd.DataFrame(index=latest.index)
screen["latest_period"] = latest["period_end"]
screen["revenue_growth"] = latest["revenue"] / prior["revenue"] - 1
screen["operating_income_growth"] = latest["operating_income"] / prior["operating_income"] - 1
screen["operating_margin"] = latest["operating_income"] / latest["revenue"]
screen["prior_operating_margin"] = prior["operating_income"] / prior["revenue"]
screen["operating_margin_delta"] = screen["operating_margin"] - screen["prior_operating_margin"]
screen["gross_margin_delta"] = latest["gross_profit"] / latest["revenue"] - prior["gross_profit"] / prior["revenue"]
screen["free_cash_flow_margin"] = latest["free_cash_flow"] / latest["revenue"]
screen["quality_score"] = (
    zscore(screen["revenue_growth"])
    + zscore(screen["operating_margin_delta"])
    + zscore(screen["free_cash_flow_margin"])
)

if screen.isna().any().any():
    raise ValueError("Retail operating leverage screen contains NaN values")

make_chart(screen)

ranked = screen.sort_values("quality_score", ascending=False)
best_margin = screen["operating_margin_delta"].idxmax()
worst_margin = screen["operating_margin_delta"].idxmin()

print("=== Retail Operating Leverage Screen ===")
print(f"Latest annual periods: {screen['latest_period'].min().date()} to {screen['latest_period'].max().date()}")
print(f"Complete-data universe: {len(screen)} retailers")
print(f"Best operating margin expansion: {best_margin} ({fmt_pct(screen.loc[best_margin, 'operating_margin_delta'])})")
print(f"Worst operating margin change: {worst_margin} ({fmt_pct(screen.loc[worst_margin, 'operating_margin_delta'])})")
print(f"Median revenue growth: {fmt_pct(screen['revenue_growth'].median())}")
print(f"Median operating margin change: {fmt_pct(screen['operating_margin_delta'].median())}")
print()
print("Retail operating leverage ranking:")
for ticker, row in ranked.iterrows():
    print(
        f"{ticker:<5} score={row['quality_score']:6.2f}  "
        f"revenue_growth={fmt_pct(row['revenue_growth'])}  "
        f"op_income_growth={fmt_pct(row['operating_income_growth'])}  "
        f"op_margin_change={fmt_pct(row['operating_margin_delta'])}  "
        f"FCF_margin={fmt_pct(row['free_cash_flow_margin'])}"
    )
