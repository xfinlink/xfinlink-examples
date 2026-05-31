# Full write-up: https://xfinlink.com/blog/ai-customer-vendor-capex-loop-python

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xfinlink as xfl

xfl.set_api_key(os.environ.get("XFINLINK_API_KEY", "YOUR_API_KEY"))  # free at https://xfinlink.com/signup

SLUG = "ai-customer-vendor-capex-loop-python"
CHART_PATH = f"/Users/lyonghee97/Desktop/xfinlink/worker/src/site/blog-images/{SLUG}.png"

BUYERS = ["MSFT", "AMZN", "META", "GOOG", "ORCL"]
SUPPLIERS = ["NVDA", "AVGO"]
TICKERS = BUYERS + SUPPLIERS

FIELDS = [
    "revenue",
    "capital_expenditures",
    "free_cash_flow",
    "operating_cash_flow",
    "gross_profit",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


df = xfl.fundamentals(
    TICKERS,
    period_type="annual",
    period="8y",
    fields=FIELDS,
)
df = df[df["ticker"].isin(TICKERS)].copy()

require(not df.empty, "fundamentals returned no rows")
require(set(TICKERS).issubset(set(df["ticker"])), "missing one or more requested tickers")

df = df.sort_values(["ticker", "period_end"])
recent = df.groupby("ticker", group_keys=False).tail(5).copy()
require((recent.groupby("ticker").size() >= 5).all(), "each ticker needs five annual observations")

recent["observation"] = recent.groupby("ticker").cumcount() - 4
recent["capex_abs"] = recent["capital_expenditures"].abs()

buyer_recent = recent[recent["ticker"].isin(BUYERS)].copy()
supplier_recent = recent[recent["ticker"].isin(SUPPLIERS)].copy()

require(buyer_recent[["revenue", "capital_expenditures", "free_cash_flow", "operating_cash_flow"]].notna().all().all(), "buyer data contains missing values")
require(supplier_recent[["revenue", "gross_profit", "free_cash_flow"]].notna().all().all(), "supplier data contains missing values")

trend = pd.DataFrame(
    {
        "buyer_capex": buyer_recent.groupby("observation")["capex_abs"].sum(),
        "buyer_fcf": buyer_recent.groupby("observation")["free_cash_flow"].sum(),
        "supplier_revenue": supplier_recent.groupby("observation")["revenue"].sum(),
        "supplier_gross_profit": supplier_recent.groupby("observation")["gross_profit"].sum(),
    }
)

latest_buyer = buyer_recent[buyer_recent["observation"] == 0].set_index("ticker")
latest_supplier = supplier_recent[supplier_recent["observation"] == 0].set_index("ticker")
prior_buyer = buyer_recent[buyer_recent["observation"] == -1].set_index("ticker")
prior_supplier = supplier_recent[supplier_recent["observation"] == -1].set_index("ticker")

latest_buyer["capex_intensity"] = latest_buyer["capex_abs"] / latest_buyer["revenue"]
latest_buyer["capex_to_fcf"] = np.where(
    latest_buyer["free_cash_flow"] > 0,
    latest_buyer["capex_abs"] / latest_buyer["free_cash_flow"],
    np.inf,
)
latest_buyer["capex_yoy"] = latest_buyer["capex_abs"] / prior_buyer["capex_abs"] - 1

latest_supplier["revenue_growth"] = latest_supplier["revenue"] / prior_supplier["revenue"] - 1
latest_supplier["gross_margin"] = latest_supplier["gross_profit"] / latest_supplier["revenue"]
latest_supplier["fcf_margin"] = latest_supplier["free_cash_flow"] / latest_supplier["revenue"]

buyer_capex_growth = trend.loc[0, "buyer_capex"] / trend.loc[-4, "buyer_capex"] - 1
supplier_revenue_growth = trend.loc[0, "supplier_revenue"] / trend.loc[-4, "supplier_revenue"] - 1
supplier_to_buyer_ratio = trend.loc[0, "supplier_revenue"] / trend.loc[0, "buyer_capex"]

print("=== AI Customer-Vendor Capex Loop ===")
print("Five annual observations ending at each company's latest fiscal year")
print()
print(f"Buyer capex: ${trend.loc[0, 'buyer_capex'] / 1000:.1f}B ({buyer_capex_growth:+.0%} vs t-4)")
print(f"Buyer free cash flow: ${trend.loc[0, 'buyer_fcf'] / 1000:.1f}B")
print(f"Supplier revenue: ${trend.loc[0, 'supplier_revenue'] / 1000:.1f}B ({supplier_revenue_growth:+.0%} vs t-4)")
print(f"Supplier gross profit: ${trend.loc[0, 'supplier_gross_profit'] / 1000:.1f}B")
print(f"Supplier revenue / buyer capex: {supplier_to_buyer_ratio:.2f}x")
print()

print("Buyer capex pressure:")
for ticker in BUYERS:
    row = latest_buyer.loc[ticker]
    capex_to_fcf = "FCF<0" if np.isinf(row["capex_to_fcf"]) else f"{row['capex_to_fcf']:5.1f}x"
    print(
        f"{ticker:5s} capex=${row['capex_abs'] / 1000:6.1f}B  "
        f"capex/revenue={row['capex_intensity']:5.1%}  "
        f"capex/FCF={capex_to_fcf:>5s}  "
        f"YoY={row['capex_yoy']:+6.0%}"
    )

print()
print("Supplier monetization:")
for ticker in SUPPLIERS:
    row = latest_supplier.loc[ticker]
    print(
        f"{ticker:5s} revenue=${row['revenue'] / 1000:6.1f}B  "
        f"YoY revenue={row['revenue_growth']:+6.0%}  "
        f"gross margin={row['gross_margin']:5.1%}  "
        f"FCF margin={row['fcf_margin']:5.1%}"
    )

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

fig, ax = plt.subplots(figsize=(10, 5))
x = trend.index
ax.plot(x, trend["buyer_capex"] / 1000, marker="o", linewidth=2.5, color="#3b82f6", label="Buyer capex")
ax.plot(x, trend["supplier_revenue"] / 1000, marker="o", linewidth=2.5, color="#22c55e", label="Supplier revenue")
ax.plot(x, trend["buyer_fcf"] / 1000, marker="o", linewidth=2.0, color="#f59e0b", label="Buyer free cash flow")
ax.set_title("AI Customer-Vendor Capex Loop")
ax.set_xlabel("Annual observation relative to latest year")
ax.set_ylabel("Dollars, billions")
ax.set_xticks(list(x))
ax.legend(frameon=False)
ax.grid(axis="y", color="#2a2a2a", linewidth=0.6, alpha=0.55)
plt.tight_layout()
plt.savefig(CHART_PATH, dpi=150, facecolor=fig.get_facecolor())
