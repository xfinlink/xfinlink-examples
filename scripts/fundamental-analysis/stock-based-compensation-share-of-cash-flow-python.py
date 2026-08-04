# Full write-up: https://xfinlink.com/blog/stock-based-compensation-share-of-cash-flow-python
"""How much of S&P 500 operating cash flow is share-based compensation?

Share-based compensation is a real expense on the income statement and a
non-cash add-back on the cash flow statement, so it lifts reported operating
cash flow (and free cash flow) without any cash leaving the company. This
script measures how large that add-back is across the S&P 500, how the
distribution has shifted since 2015, and how much of it buybacks absorb.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

FIELDS = ["operating_cash_flow", "stock_based_compensation_cf",
          "share_repurchases", "gics_sector"]


def cross_section(as_of, start, end):
    """Latest annual filing per member of the S&P 500 roster at `as_of`."""
    roster = xfl.index("sp500", as_of=as_of)
    ids = sorted({int(e) for e in roster["entity_id"].dropna()})
    frames = [xfl.fundamentals(entity_id=ids[i:i + 100], period_type="annual",
                               start=start, end=end, fields=FIELDS)
              for i in range(0, len(ids), 100)]
    df = pd.concat(frames, ignore_index=True)
    latest = df.sort_values(["entity_id", "period_end"]).groupby("entity_id").tail(1)
    latest = latest[(latest["operating_cash_flow"] > 0) &
                    (latest["stock_based_compensation_cf"] > 0)].copy()
    latest["sbc_share"] = (latest["stock_based_compensation_cf"] /
                           latest["operating_cash_flow"])
    return latest


panels = {
    "FY2015": cross_section("2015-12-31", "2015-01-01", "2016-06-30"),
    "FY2020": cross_section("2020-12-31", "2020-01-01", "2021-06-30"),
    "latest": cross_section(None, "2025-01-01", "2026-06-30"),
}

print("=" * 74)
print("SHARE-BASED COMPENSATION AS A SHARE OF OPERATING CASH FLOW")
print("S&P 500 members on the roster at each vintage")
print("=" * 74)
print(f"{'vintage':<10}{'firms':>7}{'median':>10}{'75th pct':>11}"
      f"{'90th pct':>11}{'above 10%':>12}")
for name, p in panels.items():
    s = p["sbc_share"]
    print(f"{name:<10}{len(p):>7}{s.median():>9.1%}{s.quantile(0.75):>11.1%}"
          f"{s.quantile(0.90):>11.1%}{(s > 0.10).mean():>12.1%}")

cur, old = panels["latest"], panels["FY2015"]

# Same-company comparison: firms on both the FY2015 and the current roster.
both = set(old["entity_id"]) & set(cur["entity_id"])
o = old[old["entity_id"].isin(both)]["sbc_share"]
c = cur[cur["entity_id"].isin(both)]["sbc_share"]
print()
print(f"SAME COMPANIES ON BOTH ROSTERS ({len(both)} firms)")
print(f"{'':<10}{'median':>10}{'90th pct':>11}{'above 10%':>12}")
for label, s in [("FY2015", o), ("latest", c)]:
    print(f"{label:<10}{s.median():>9.1%}{s.quantile(0.90):>11.1%}"
          f"{(s > 0.10).mean():>12.1%}")

sec = (cur.groupby("gics_sector")["sbc_share"].agg(["size", "median"])
          .join(old.groupby("gics_sector")["sbc_share"].median().rename("median_2015"))
          .sort_values("median", ascending=False))
print()
print("BY SECTOR, LATEST ANNUAL FILINGS")
print(f"{'sector':<26}{'firms':>7}{'median':>10}{'FY2015':>10}")
for s, r in sec.iterrows():
    print(f"{s:<26}{int(r['size']):>7}{r['median']:>9.1%}{r['median_2015']:>10.1%}")

buy = cur[cur["share_repurchases"] > 0].copy()
buy["offset"] = buy["stock_based_compensation_cf"] / buy["share_repurchases"]
print()
print("BUYBACKS VERSUS SHARE-BASED COMPENSATION, LATEST FILINGS")
print(f"repurchasers                        {len(buy):>7}")
print(f"median SBC / buyback spend          {buy['offset'].median():>7.1%}")
print(f"SBC above half of buyback spend     {(buy['offset'] > 0.5).sum():>7}")
print(f"SBC above all buyback spend         {(buy['offset'] > 1.0).sum():>7}")

# ── chart ────────────────────────────────────────────────────────────
plt.rcParams.update({"figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
                     "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
                     "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
                     "axes.edgecolor": "#333333", "font.size": 10})
plot = sec.iloc[::-1]
y = range(len(plot))
fig, ax = plt.subplots(figsize=(10, 5))
ax.barh([i + 0.20 for i in y], plot["median"] * 100, height=0.40,
        color="#3b82f6", label="Latest annual filing")
ax.barh([i - 0.20 for i in y], plot["median_2015"] * 100, height=0.40,
        color="#6b7280", label="FY2015")
ax.set_yticks(list(y))
ax.set_yticklabels(plot.index)
ax.set_xlabel("Median share-based compensation as a share of operating cash flow (%)")
ax.set_title("Stock compensation as a share of cash flow, S&P 500 by sector")
ax.legend(frameon=False, loc="lower right")
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
plt.tight_layout()
plt.savefig("stock-based-compensation-share-of-cash-flow-python.png",
            dpi=150, facecolor="#0a0a0a")
