"""When did each fundamental number become public?

Measures the gap between a fiscal period end and the filing that reported it,
then counts how many companies a period_end-based join would contaminate at
several rebalance dates.

Sample: the first 100 S&P 500 members alphabetically, quarterly statements
2023-01-01 to 2026-06-30. Fiscal Q4 is excluded because those figures arrive
with the annual report rather than a quarterly one.
"""

import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

members = xfl.index("sp500")
tickers = members["ticker"].dropna().tolist()[:100]

f = xfl.fundamentals(
    tickers,
    period_type="quarterly",
    start="2023-01-01",
    end="2026-06-30",
    fields=["revenue", "net_income"],
)
f = f[f["fiscal_period"].isin(["Q1", "Q2", "Q3"])].copy()
f["lag_days"] = (f["filing_date"] - f["period_end"]).dt.days

print(f"quarters: {len(f)}   companies: {f['entity_id'].nunique()}")
print("days from period end to filing")
for q in (0.25, 0.50, 0.75):
    print(f"  {int(q * 100)}th percentile: {f['lag_days'].quantile(q):.0f}")
for d in (30, 35, 40):
    print(f"  filed within {d} days: {(f['lag_days'] <= d).mean() * 100:.1f}%")

print("\nrebalance date   companies whose newest period_end was not yet filed")
for d in ("2025-11-14", "2026-01-15", "2026-02-13", "2026-03-31", "2026-05-15"):
    form = pd.Timestamp(d)
    newest = (f[f["period_end"] <= form].sort_values("period_end")
              .groupby("entity_id").tail(1).set_index("entity_id"))
    filed = (f[f["filing_date"] <= form].sort_values("period_end")
             .groupby("entity_id").tail(1).set_index("entity_id"))
    j = newest[["period_end"]].join(filed[["period_end"]], rsuffix="_filed", how="inner")
    print(f"  {d}       {(j['period_end'] != j['period_end_filed']).sum()} of {len(j)}")

aapl = f[(f["ticker"] == "AAPL") & (f["period_end"] >= "2025-12-01")]
print("\nAAPL rows around a 2026-03-31 rebalance")
print(aapl[["period_end", "filing_date", "revenue"]].to_string(index=False))
