# Full write-up: https://xfinlink.com/blog/reverse-dcf-implied-growth-python
import pandas as pd
import matplotlib.pyplot as plt
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

COST_OF_EQUITY = 0.09

# --- Universe: current S&P 500, keyed on entity identifier -----------------
idx = xfl.index("sp500", as_of="2026-06-30", limit=1000)
ids = [int(i) for i in idx["entity_id"].dropna().unique()]

fun = xfl.fundamentals(entity_id=ids, period_type="annual",
                       start="2024-06-01", end="2026-06-30",
                       fields=["free_cash_flow", "revenue", "gics_sector"])
met = xfl.metrics(entity_id=ids, period_type="annual",
                  start="2024-06-01", end="2026-06-30",
                  fields=["market_cap"])

latest = lambda d: d.sort_values("period_end").groupby("entity_id").tail(1)
df = (latest(fun).set_index("entity_id")
      .join(latest(met).set_index("entity_id")[["market_cap"]], how="inner"))

print(f"Current members with a recent annual record: {len(df)}")

# A perpetuity on cash flow needs a positive cash flow to invert. Banks and
# property companies are excluded because their cash flow statements are not
# comparable to an operating business on this measure.
df = df[~df["gics_sector"].isin(["Financials", "Real Estate"])]
print(f"  excluding Financials and Real Estate: {len(df)}")
df = df[(df["free_cash_flow"] > 0) & (df["market_cap"] > 0)]
print(f"  with positive free cash flow:        {len(df)}")

# --- Invert the growing perpetuity -----------------------------------------
# P = FCF * (1 + g) / (r - g)   =>   g = (P*r - FCF) / (P + FCF)
def implied_growth(price, fcf, r):
    return (price * r - fcf) / (price + fcf)


df["implied_g"] = implied_growth(df["market_cap"], df["free_cash_flow"],
                                 COST_OF_EQUITY)

q = df["implied_g"].quantile([0.1, 0.25, 0.5, 0.75, 0.9]) * 100
print(f"\nImplied perpetual free cash flow growth at a "
      f"{COST_OF_EQUITY:.0%} cost of equity")
print(f"  10th {q[0.1]:6.2f}%   25th {q[0.25]:6.2f}%   median {q[0.5]:6.2f}%"
      f"   75th {q[0.75]:6.2f}%   90th {q[0.9]:6.2f}%")

for hurdle, label in [(0.02, "inflation alone"),
                      (0.045, "long-run nominal GDP growth")]:
    share = (df["implied_g"] > hurdle).mean() * 100
    print(f"  above {hurdle:5.1%} ({label}): {share:5.1f}% of companies")

# --- By sector -------------------------------------------------------------
print("\nMedian implied growth by sector (10 or more companies)")
sec = df.groupby("gics_sector")["implied_g"].agg(["median", "count"])
sec = sec[sec["count"] >= 10].sort_values("median", ascending=False)
for name, row in sec.iterrows():
    print(f"  {name[:24]:24s} {row['median'] * 100:6.2f}%   n={int(row['count']):3d}")

# --- The assumption that matters -------------------------------------------
print("\nSensitivity: the same prices under a different cost of equity")
for r in (0.08, 0.09, 0.10, 0.11):
    g = implied_growth(df["market_cap"], df["free_cash_flow"], r)
    print(f"  r = {r:.0%}   median {g.median() * 100:6.2f}%   "
          f"share above 4.5%: {(g > 0.045).mean() * 100:5.1f}%")

# --- Chart -----------------------------------------------------------------
plt.rcParams.update({"figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
                     "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
                     "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
                     "axes.edgecolor": "#333333"})
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(df["implied_g"].clip(-0.02, 0.09) * 100, bins=40, color="#3b82f6")
ax.axvline(4.5, color="#e0e0e0", linestyle="--", linewidth=1.5,
           label="Long-run nominal GDP growth (4.5%)")
ax.set_xlabel("Implied perpetual free cash flow growth (%)")
ax.set_ylabel("Number of companies")
ax.set_title("Growth already priced into the S&P 500")
ax.legend(facecolor="#0a0a0a", edgecolor="#333333", labelcolor="#e0e0e0")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
plt.tight_layout()
plt.savefig("reverse-dcf-implied-growth-python.png", dpi=150)
