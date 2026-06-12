# Full write-up: https://xfinlink.com/blog/russell2000-quality-balance-sheet-python

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import xfinlink as xfl

xfl.set_api_key(os.environ.get("XFINLINK_API_KEY", "YOUR_API_KEY"))  # free at https://xfinlink.com/signup

SLUG = "russell2000-quality-balance-sheet-python"
INDEX_AS_OF = "2025-12-31"
CONSTITUENT_LIMIT = 300
FIELDS = ["roe", "fcf_margin", "debt_to_assets", "interest_coverage", "market_cap"]


def zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if std == 0:
        return series * 0
    return (series - series.mean()) / std


def fmt_pct(value: float) -> str:
    return f"{value * 100:6.1f}%"


def winsorize(series: pd.Series) -> pd.Series:
    low = series.quantile(0.05)
    high = series.quantile(0.95)
    return series.clip(low, high)


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

    colors = screen["quality_quintile"].map({5: "#3b82f6", 1: "#ef4444"}).fillna("#6b7280")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(
        screen["debt_to_assets"] * 100,
        screen["fcf_margin"] * 100,
        s=(screen["market_cap"].clip(upper=6000) / 6000 * 120) + 24,
        color=colors,
        alpha=0.78,
    )
    ax.axhline(0, color="#e0e0e0", linewidth=1, alpha=0.45)
    ax.set_title("Russell 2000 quality screen")
    ax.set_xlabel("Debt as share of assets")
    ax.set_ylabel("Free-cash-flow margin")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    plt.tight_layout()
    out = Path("worker/src/site/blog-images") / f"{SLUG}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, facecolor="#0a0a0a")
    plt.close(fig)


constituents = xfl.index("russell2000", as_of=INDEX_AS_OF, limit=CONSTITUENT_LIMIT)
if constituents.empty:
    raise ValueError("Russell 2000 constituent DataFrame is empty")

tickers = [
    ticker
    for ticker in constituents["ticker"].dropna().astype(str).tolist()
    if "." not in ticker and "-" not in ticker
]
if len(tickers) < 100:
    raise ValueError("Too few usable Russell 2000 tickers after ticker-format filtering")

metrics = xfl.metrics(tickers, period_type="annual", period="3y", fields=FIELDS, max_rows=10000)
if metrics.empty:
    raise ValueError("Metrics DataFrame is empty")

required = {"ticker", "period_end", *FIELDS}
missing_cols = required - set(metrics.columns)
if missing_cols:
    raise ValueError(f"Missing metrics columns: {sorted(missing_cols)}")

latest = metrics.sort_values("period_end").groupby("ticker").tail(1).set_index("ticker")
complete = latest[FIELDS + ["period_end"]].dropna().copy()
screen = complete[
    (complete["market_cap"] > 100)
    & (complete["market_cap"] < 20_000)
    & (complete["roe"].between(-1, 1))
    & (complete["fcf_margin"].between(-1, 1))
    & (complete["debt_to_assets"].between(0, 1.5))
].copy()
if len(screen) < 100:
    raise ValueError("Too few complete Russell 2000 records after outlier controls")

screen["interest_coverage_clipped"] = screen["interest_coverage"].clip(-10, 50)
screen["quality_score"] = (
    zscore(winsorize(screen["roe"]))
    + zscore(winsorize(screen["fcf_margin"]))
    - zscore(winsorize(screen["debt_to_assets"]))
    + zscore(winsorize(screen["interest_coverage_clipped"]))
)
screen["quality_quintile"] = pd.qcut(screen["quality_score"], 5, labels=[1, 2, 3, 4, 5])
screen["quality_quintile"] = screen["quality_quintile"].astype(int)
screen = screen.sort_values("quality_score", ascending=False)

make_chart(screen)

quintiles = screen.groupby("quality_quintile")[["roe", "fcf_margin", "debt_to_assets", "interest_coverage", "market_cap"]].median()
top = screen.head(8)
bottom = screen.tail(8).sort_values("quality_score")

print("=== Russell 2000 Quality Balance-Sheet Screen ===")
print(f"Membership date: {INDEX_AS_OF}")
print(f"Constituents requested: {len(constituents)}")
print(f"Complete annual metric records: {len(complete)}")
print(f"Records after outlier controls: {len(screen)}")
print(f"Top quality quintile median ROE: {fmt_pct(quintiles.loc[5, 'roe'])}")
print(f"Bottom quality quintile median ROE: {fmt_pct(quintiles.loc[1, 'roe'])}")
print(f"Top quality quintile median debt/assets: {fmt_pct(quintiles.loc[5, 'debt_to_assets'])}")
print(f"Bottom quality quintile median debt/assets: {fmt_pct(quintiles.loc[1, 'debt_to_assets'])}")
print()
print("Top quality names:")
for ticker, row in top.iterrows():
    print(
        f"{ticker:<6} score={row['quality_score']:6.2f}  "
        f"ROE={fmt_pct(row['roe'])}  "
        f"FCF_margin={fmt_pct(row['fcf_margin'])}  "
        f"debt/assets={fmt_pct(row['debt_to_assets'])}  "
        f"interest_cover={row['interest_coverage']:7.1f}  "
        f"market_cap=${row['market_cap']:,.0f}M"
    )
print()
print("Lowest quality names:")
for ticker, row in bottom.iterrows():
    print(
        f"{ticker:<6} score={row['quality_score']:6.2f}  "
        f"ROE={fmt_pct(row['roe'])}  "
        f"FCF_margin={fmt_pct(row['fcf_margin'])}  "
        f"debt/assets={fmt_pct(row['debt_to_assets'])}  "
        f"interest_cover={row['interest_coverage']:7.1f}  "
        f"market_cap=${row['market_cap']:,.0f}M"
    )
