# Full write-up: https://xfinlink.com/blog/quarterly-cash-flow-year-to-date
import json
import urllib.request

import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

TAG = "NetCashProvidedByUsedInOperatingActivities"
HEADERS = {"User-Agent": "your name your.email@example.com"}  # SEC requires this


def as_filed(cik, fiscal_start):
    """Operating cash flow exactly as the company tagged it with the SEC."""
    url = f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/us-gaap/{TAG}.json"
    req = urllib.request.Request(url, headers=HEADERS)
    facts = json.load(urllib.request.urlopen(req))["units"]["USD"]
    return sorted({(f["start"], f["end"], f["val"], f["form"])
                   for f in facts if f["start"] == fiscal_start})


cases = [
    ("Apple fiscal 2025", "AAPL", "0000320193", "2024-09-29", "2024-10-01", "2025-10-01"),
    ("Walmart fiscal 2026", "WMT", "0000104169", "2025-02-01", "2025-02-05", "2026-02-05"),
]

for label, ticker, cik, fiscal_start, start, end in cases:
    print(label)
    print("  as filed with the SEC")
    for s, e, val, form in as_filed(cik, fiscal_start):
        months = round((pd.Timestamp(e) - pd.Timestamp(s)).days / 30.44)
        print(f"    {s} to {e}  {months:2d} months  {val / 1e6:>9,.0f}m  {form}")

    q = xfl.fundamentals(ticker, period_type="quarterly", start=start, end=end,
                         fields=["operating_cash_flow", "capital_expenditures",
                                 "free_cash_flow"])
    a = xfl.fundamentals(ticker, period_type="annual", start=start, end=end,
                         fields=["operating_cash_flow"])
    print("  standalone quarters from xfinlink")
    for _, r in q.iterrows():
        print(f"    {r.fiscal_period} to {r.period_end.date()}  "
              f"ocf {r.operating_cash_flow:>8,.0f}m  "
              f"capex {r.capital_expenditures:>6,.0f}m  "
              f"fcf {r.free_cash_flow:>8,.0f}m")
    print(f"    quarters sum to {q['operating_cash_flow'].sum():>9,.0f}m")
    print(f"    annual row      {a['operating_cash_flow'].iloc[0]:>9,.0f}m\n")
