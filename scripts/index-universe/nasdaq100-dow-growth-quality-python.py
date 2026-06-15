# Full write-up: https://xfinlink.com/blog/nasdaq100-dow-growth-quality-python

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import xfinlink as xfl

xfl.set_api_key(os.environ.get("XFINLINK_API_KEY", "YOUR_API_KEY"))  # free at https://xfinlink.com/signup

SLUG = "nasdaq100-dow-growth-quality-python"
FIELDS = ["market_cap", "revenue_growth", "gross_margin", "fcf_margin", "pe_ratio"]


def fmt_pct(value: float) -> str:
    return f"{value * 100:6.1f}%"


def make_chart(metrics: pd.DataFrame) -> None:
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
    colors = {
        "Nasdaq 100 only": "#3b82f6",
        "Dow only": "#6b7280",
        "Both indexes": "#93c5fd",
    }
    for cohort, group in metrics.groupby("cohort"):
        ax.scatter(
            group["revenue_growth"] * 100,
            group["fcf_margin"] * 100,
            s=70,
            color=colors[cohort],
            alpha=0.82,
            edgecolor="#e0e0e0",
            linewidth=0.4,
            label=cohort,
        )
    for _, row in metrics.nlargest(8, "market_cap").iterrows():
        ax.annotate(row["ticker"], (row["revenue_growth"] * 100, row["fcf_margin"] * 100),
                    textcoords="offset points", xytext=(6, 5), fontsize=8)
    ax.axhline(0, color="#e0e0e0", linewidth=1, alpha=0.35)
    ax.axvline(0, color="#e0e0e0", linewidth=1, alpha=0.35)
    ax.set_title("Nasdaq 100 versus Dow: growth and cash-flow quality")
    ax.set_xlabel("TTM revenue growth")
    ax.set_ylabel("Free-cash-flow margin")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    plt.tight_layout()
    out = Path("worker/src/site/blog-images") / f"{SLUG}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, facecolor="#0a0a0a")
    plt.close(fig)


ndx = xfl.index("ndx100", limit=150)
dow = xfl.index("djia", limit=50)
if ndx.empty or dow.empty:
    raise ValueError("Index constituent DataFrame is empty")

required_index = {"entity_id", "ticker", "removed_date"}
for name, frame in [("Nasdaq 100", ndx), ("Dow", dow)]:
    missing = required_index - set(frame.columns)
    if missing:
        raise ValueError(f"Missing {name} index columns: {sorted(missing)}")

ndx_current = ndx[ndx["removed_date"].isna()].dropna(subset=["entity_id", "ticker"]).copy()
dow_current = dow[dow["removed_date"].isna()].dropna(subset=["entity_id", "ticker"]).copy()
ndx_current = ndx_current[~ndx_current["ticker"].str.contains(r"\\.", regex=True)]
dow_current = dow_current[~dow_current["ticker"].str.contains(r"\\.", regex=True)]

ndx_entities = set(ndx_current["entity_id"])
dow_entities = set(dow_current["entity_id"])
entity_to_cohort = {}
for entity_id in ndx_entities | dow_entities:
    if entity_id in ndx_entities and entity_id in dow_entities:
        entity_to_cohort[entity_id] = "Both indexes"
    elif entity_id in ndx_entities:
        entity_to_cohort[entity_id] = "Nasdaq 100 only"
    else:
        entity_to_cohort[entity_id] = "Dow only"

tickers = sorted(set(ndx_current["ticker"]) | set(dow_current["ticker"]))
if len(tickers) < 100:
    raise ValueError(f"Unexpectedly small combined index universe: {len(tickers)}")

metrics = xfl.metrics(tickers, period_type="ttm", fields=FIELDS, max_rows=2000)
if metrics.empty:
    raise ValueError("Metrics DataFrame is empty")

required_metrics = {"entity_id", "ticker", "period_end", *FIELDS}
missing_metrics = required_metrics - set(metrics.columns)
if missing_metrics:
    raise ValueError(f"Missing metrics columns: {sorted(missing_metrics)}")

latest = metrics.sort_values("period_end").groupby("entity_id").tail(1).copy()
latest["cohort"] = latest["entity_id"].map(entity_to_cohort)
latest = latest.dropna(subset=["cohort", *FIELDS])
latest = latest[
    (latest["market_cap"] > 1_000)
    & latest["revenue_growth"].between(-1.0, 2.0)
    & latest["gross_margin"].between(-1.0, 1.5)
    & latest["fcf_margin"].between(-1.0, 1.5)
    & (latest["pe_ratio"] > 0)
]
if latest.empty:
    raise ValueError("No complete metrics after sanity filters")

counts = latest.groupby("cohort")["ticker"].count()
if counts.get("Nasdaq 100 only", 0) < 50 or counts.get("Dow only", 0) < 10:
    raise ValueError(f"Complete-data cohort sizes are too small: {counts.to_dict()}")
if latest["market_cap"].max() > 10_000_000:
    raise ValueError("Market cap appears implausibly large")

summary = latest.groupby("cohort").agg(
    count=("ticker", "count"),
    median_revenue_growth=("revenue_growth", "median"),
    median_gross_margin=("gross_margin", "median"),
    median_fcf_margin=("fcf_margin", "median"),
    median_pe=("pe_ratio", "median"),
    positive_fcf_share=("fcf_margin", lambda s: (s > 0).mean()),
).reset_index()
if summary.isna().any().any():
    raise ValueError("Index summary contains NaN values")

make_chart(latest)
top_growth = latest.sort_values("revenue_growth", ascending=False).head(10)

print("=== Nasdaq 100 vs Dow Growth Quality ===")
print(f"Current Nasdaq 100 constituents after cleaning: {len(ndx_current)}")
print(f"Current Dow constituents after cleaning: {len(dow_current)}")
print(f"Combined complete-data universe: {len(latest)} companies")
print(f"Metric period range: {latest['period_end'].min().date()} to {latest['period_end'].max().date()}")
print()
print("Cohort medians:")
for _, row in summary.sort_values("cohort").iterrows():
    print(
        f"{row['cohort']:<16} n={int(row['count']):3d}  "
        f"rev_growth={fmt_pct(row['median_revenue_growth'])}  "
        f"gross_margin={fmt_pct(row['median_gross_margin'])}  "
        f"FCF_margin={fmt_pct(row['median_fcf_margin'])}  "
        f"positive_FCF={fmt_pct(row['positive_fcf_share'])}  "
        f"PE={row['median_pe']:5.1f}"
    )
print()
print("Fastest revenue growers in the combined index set:")
for _, row in top_growth.iterrows():
    print(
        f"{row['ticker']:<5} {row['cohort']:<16} "
        f"rev_growth={fmt_pct(row['revenue_growth'])}  "
        f"FCF_margin={fmt_pct(row['fcf_margin'])}  "
        f"PE={row['pe_ratio']:5.1f}"
    )
