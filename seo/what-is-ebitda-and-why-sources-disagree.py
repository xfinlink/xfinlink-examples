# Full write-up: https://xfinlink.com/blog/what-is-ebitda-and-why-sources-disagree
#
# Two defensible EBITDA builds for one company-year, from filed statement lines.

import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

fields = [
    "operating_income",
    "depreciation_amortization",
    "net_income",
    "interest_expense",
    "income_tax_expense",
    "stock_based_compensation",
    "ebitda",
]

df = xfl.fundamentals(
    "UBER",
    start="2025-01-01",
    end="2025-12-31",
    period_type="annual",
    fields=fields,
)
r = df.iloc[-1]

from_operating = r.operating_income + r.depreciation_amortization
from_net_income = (
    r.net_income + r.interest_expense + r.income_tax_expense + r.depreciation_amortization
)

print(f"period_end                     {r.period_end.date()}")
print(f"operating income               {r.operating_income:>9,.0f}")
print(f"depreciation and amortisation  {r.depreciation_amortization:>9,.0f}")
print(f"net income                     {r.net_income:>9,.0f}")
print(f"interest expense               {r.interest_expense:>9,.0f}")
print(f"income tax expense             {r.income_tax_expense:>9,.0f}")
print(f"stock-based compensation       {r.stock_based_compensation:>9,.0f}")
print()
print(f"EBITDA from operating income   {from_operating:>9,.0f}")
print(f"EBITDA from net income         {from_net_income:>9,.0f}")
print(f"served ebitda field            {r.ebitda:>9,.0f}")
print(f"difference between the builds  {from_net_income - from_operating:>9,.0f}")
