# Full write-up: https://xfinlink.com/blog/micron-memory-cycle-forecast-python

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import xfinlink as xfl

xfl.set_api_key(os.environ.get("XFINLINK_API_KEY", "YOUR_API_KEY"))  # free at https://xfinlink.com/signup

SLUG = "micron-memory-cycle-forecast-python"
TICKERS = ["MU", "NVDA", "AMD", "INTC", "AVGO", "QCOM"]
FIELDS = [
    "revenue",
    "gross_profit",
    "operating_income",
    "inventory",
    "free_cash_flow",
    "capital_expenditures",
]


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
    colors = ["#3b82f6" if ticker == "MU" else "#6b7280" for ticker in screen.index]
    ax.scatter(
        screen["inventory_to_revenue"] * 100,
        screen["gross_margin"] * 100,
        s=(screen["revenue_growth"].clip(lower=-0.2, upper=0.8) + 0.25) * 180,
        color=colors,
        alpha=0.9,
    )
    for ticker, row in screen.iterrows():
        ax.annotate(ticker, (row["inventory_to_revenue"] * 100, row["gross_margin"] * 100),
                    xytext=(5, 5), textcoords="offset points", fontsize=9)
    ax.set_title("Semiconductor cycle position")
    ax.set_xlabel("Inventory as share of revenue")
    ax.set_ylabel("Gross margin")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    plt.tight_layout()
    out = Path("worker/src/site/blog-images") / f"{SLUG}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, facecolor="#0a0a0a")
    plt.close(fig)


df = xfl.fundamentals(TICKERS, period_type="annual", period="6y", fields=FIELDS)
if df.empty:
    raise ValueError("Fundamentals DataFrame is empty")

required = {"ticker", "period_end", *FIELDS}
missing_cols = required - set(df.columns)
if missing_cols:
    raise ValueError(f"Missing columns: {sorted(missing_cols)}")

df = df.sort_values(["ticker", "period_end"])
recent = df.groupby("ticker").tail(2)
if recent["ticker"].nunique() != len(TICKERS):
    raise ValueError("Not every ticker has two annual observations")

latest = recent.groupby("ticker").tail(1).set_index("ticker")
prior = recent.groupby("ticker").head(1).set_index("ticker")

if (latest["revenue"] <= 0).any():
    raise ValueError("Revenue should be positive for this semiconductor universe")
if latest[FIELDS].isna().any().any() or prior[FIELDS].isna().any().any():
    raise ValueError("Fundamentals contain NaN values in required fields")

screen = pd.DataFrame(index=latest.index)
screen["period_end"] = latest["period_end"]
screen["revenue_growth"] = latest["revenue"] / prior["revenue"] - 1
screen["gross_margin"] = latest["gross_profit"] / latest["revenue"]
screen["operating_margin"] = latest["operating_income"] / latest["revenue"]
screen["inventory_to_revenue"] = latest["inventory"] / latest["revenue"]
screen["inventory_growth"] = latest["inventory"] / prior["inventory"] - 1
screen["free_cash_flow_margin"] = latest["free_cash_flow"] / latest["revenue"]
screen["capex_intensity"] = latest["capital_expenditures"].abs() / latest["revenue"]

screen["cycle_score"] = (
    zscore(screen["revenue_growth"])
    + zscore(screen["gross_margin"])
    + zscore(screen["free_cash_flow_margin"])
    - zscore(screen["inventory_to_revenue"])
)

make_chart(screen)

ranked = screen.sort_values("cycle_score", ascending=False)
mu = screen.loc["MU"]

print("=== Micron Memory-Cycle Forecast Screen ===")
print(f"Latest annual periods: {screen['period_end'].min().date()} to {screen['period_end'].max().date()}")
print(f"MU revenue growth:              {fmt_pct(mu['revenue_growth'])}")
print(f"MU gross margin:                {fmt_pct(mu['gross_margin'])}")
print(f"MU operating margin:            {fmt_pct(mu['operating_margin'])}")
print(f"MU inventory / revenue:         {fmt_pct(mu['inventory_to_revenue'])}")
print(f"MU inventory growth:            {fmt_pct(mu['inventory_growth'])}")
print(f"MU free-cash-flow margin:       {fmt_pct(mu['free_cash_flow_margin'])}")
print(f"MU cycle score rank:            {ranked.index.get_loc('MU') + 1} of {len(ranked)}")
print()
print("Semiconductor cycle ranking:")
for ticker, row in ranked.iterrows():
    print(
        f"{ticker:<5} score={row['cycle_score']:6.2f}  "
        f"revenue_growth={fmt_pct(row['revenue_growth'])}  "
        f"gross_margin={fmt_pct(row['gross_margin'])}  "
        f"inventory/revenue={fmt_pct(row['inventory_to_revenue'])}  "
        f"FCF_margin={fmt_pct(row['free_cash_flow_margin'])}"
    )
