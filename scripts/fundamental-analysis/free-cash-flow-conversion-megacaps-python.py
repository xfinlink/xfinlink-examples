# Full write-up: https://xfinlink.com/blog/free-cash-flow-conversion-megacaps-python
import os
import pandas as pd
import matplotlib.pyplot as plt
import xfinlink as xfl

xfl.set_api_key(os.environ.get("XFINLINK_API_KEY", "YOUR_API_KEY"))  # free at https://xfinlink.com/signup

# Large non-financial companies across sectors. Free cash flow is not a
# meaningful concept for banks and insurers, so those are excluded.
TICKERS = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "JNJ", "XOM", "PG",
           "KO", "HD", "WMT", "UNH", "COST", "MRK", "ORCL", "CSCO"]

f = xfl.fundamentals(TICKERS, period_type="annual", period="3y",
                     fields=["revenue", "net_income", "free_cash_flow",
                             "operating_cash_flow", "capital_expenditures"])

# Keep the most recent completed fiscal year per company.
latest = (f.sort_values("period_end")
            .groupby("ticker", as_index=False)
            .last())

latest["fcf_margin"] = latest["free_cash_flow"] / latest["revenue"] * 100
latest["cash_conversion"] = latest["free_cash_flow"] / latest["net_income"] * 100
latest = latest.sort_values("cash_conversion", ascending=False)

out = latest[["ticker", "period_end", "revenue", "net_income",
              "free_cash_flow", "fcf_margin", "cash_conversion"]].copy()

# ---- formatted console output ----
disp = out.copy()
disp["period_end"] = disp["period_end"].dt.date
# Statement values are reported in millions of dollars.
for c in ["revenue", "net_income", "free_cash_flow"]:
    disp[c] = (disp[c] / 1e3).round(1)
disp["fcf_margin"] = disp["fcf_margin"].round(1)
disp["cash_conversion"] = disp["cash_conversion"].round(0).astype(int)
disp.columns = ["ticker", "fy_end", "rev_$bn", "ni_$bn", "fcf_$bn",
                "fcf_margin_%", "cash_conv_%"]
print("Free cash flow conversion, latest completed fiscal year")
print(disp.to_string(index=False))
print()
print(f"median cash conversion: {out['cash_conversion'].median():.0f}%")
print(f"median FCF margin:       {out['fcf_margin'].median():.1f}%")

# ---- chart ----
plt.rcParams.update({
    "figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
    "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
    "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
    "axes.edgecolor": "#333333",
})
fig, ax = plt.subplots(figsize=(10, 7))
colors = ["#3b82f6" if v >= 100 else "#ef4444" for v in out["cash_conversion"]]
ax.barh(out["ticker"], out["cash_conversion"], color=colors)
ax.axvline(100, color="#e0e0e0", linewidth=1, linestyle="--")
ax.set_xlabel("Free cash flow as a percent of net income")
ax.set_title("How much reported profit becomes cash")
ax.invert_yaxis()
ax.text(101, len(out) - 0.5, "profit fully backed by cash →",
        color="#9ca3af", fontsize=9, va="center")
plt.tight_layout()
plt.savefig("free-cash-flow-conversion-megacaps-python.png", dpi=150,
            facecolor="#0a0a0a")
print("\nchart saved")
