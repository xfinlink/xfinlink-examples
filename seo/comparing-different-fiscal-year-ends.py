# Full write-up: https://xfinlink.com/blog/comparing-different-fiscal-year-ends
"""Aligning companies whose fiscal years end on different dates."""

import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

ids = xfl.index("sp500")["entity_id"].dropna().astype(int).tolist()
f = pd.concat([xfl.fundamentals(entity_id=ids[i:i + 100], period_type="annual",
                                start="2024-09-01", fields=["revenue"])
               for i in range(0, len(ids), 100)], ignore_index=True)
latest = f.sort_values("period_end").groupby("entity_id", as_index=False).tail(1)
month = pd.to_datetime(latest["period_end"]).dt.month

print(f"S&P 500 companies: {len(latest)}")
print(f"December year end: {(month == 12).sum()}")
print(f"Other month:       {(month != 12).sum()} "
      f"({100 * (month != 12).mean():.0f}%)")
print("\nMost common non-December year ends")
names = {1: "January", 6: "June", 9: "September", 10: "October"}
for m, n in month[month != 12].value_counts().head(4).items():
    print(f"  {names.get(m, m):<10} {n}")

windows = {}
print("\nThe same label covers different twelve months")
print(f"{'ticker':<8}{'fiscal_year':>12}{'period_end':>14}{'twelve months to':>20}")
for t in ["WMT", "HD", "NKE", "MSFT", "COST", "AAPL"]:
    r = (xfl.fundamentals(t, period_type="annual", start="2024-09-01", fields=["revenue"])
         .sort_values("period_end").iloc[-1])
    end = pd.to_datetime(r["period_end"])
    start = end - pd.DateOffset(months=12) + pd.Timedelta(days=1)
    windows[t] = (start, end)
    print(f"{t:<8}{r['fiscal_year']:>12}{str(end.date()):>14}"
          f"{start.strftime('%b %Y') + ' - ' + end.strftime('%b %Y'):>20}")

overlap = ((min(windows["WMT"][1], windows["MSFT"][1])
            - max(windows["WMT"][0], windows["MSFT"][0])).days) / 30.44
print(f"\nWalmart and Microsoft both label it fiscal 2026. "
      f"The two windows share {overlap:.0f} months.")

# Align on period_end: each company's most recent close on or before a cut date.
CUT = "2026-06-30"
aligned = f[pd.to_datetime(f["period_end"]) <= CUT]
aligned = aligned.sort_values("period_end").groupby("entity_id", as_index=False).tail(1)
lag = (pd.to_datetime(CUT) - pd.to_datetime(aligned["period_end"])).dt.days
print(f"\nAligned on period_end at {CUT}: {len(aligned)} companies")
print(f"  staleness in days: median {lag.median():.0f}, 90th percentile {lag.quantile(.9):.0f}")
