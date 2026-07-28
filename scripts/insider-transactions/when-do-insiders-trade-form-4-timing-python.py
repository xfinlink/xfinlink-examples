# Full write-up: https://xfinlink.com/blog/when-do-insiders-trade-form-4-timing-python
"""When do S&P 500 insiders actually place open-market trades?

Maps every Form 4 open-market buy and sell to the number of days elapsed since
the filer's own fiscal quarter end, then compares the daily transaction rate
across the quarter. Built from SEC EDGAR public filings and market data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

START, END = "2021-07-01", "2026-06-30"
BATCH = 25

tickers = sorted(xfl.index("sp500")["ticker"].dropna().unique().tolist())
batches = [tickers[i:i + BATCH] for i in range(0, len(tickers), BATCH)]

trades, quarters = [], []
for b in batches:
    t = xfl.insiders(b, start=START, end=END,
                     transaction_type=["open_market_buy", "open_market_sell"],
                     fields=["ticker", "transaction_date", "transaction_type",
                             "insider_role", "insider_name"],
                     max_rows=200000)
    if len(t):
        trades.append(t)
    q = xfl.fundamentals(b, start="2021-01-01", end="2026-07-28",
                         period_type="quarterly", max_rows=200000)
    if len(q):
        quarters.append(q[["ticker", "period_end", "fiscal_period", "filing_date"]])

trades = pd.concat(trades, ignore_index=True)
quarters = pd.concat(quarters, ignore_index=True)

# Typical interim reporting lag. Only the 10-Q quarters are used: the fourth
# quarter is reported on the annual 10-K, which carries a different deadline.
q13 = quarters[quarters["fiscal_period"].isin(["Q1", "Q2", "Q3"])].copy()
q13["lag"] = (pd.to_datetime(q13["filing_date"]).dt.tz_localize(None)
              - pd.to_datetime(q13["period_end"])).dt.days
lag = q13.loc[q13["lag"].between(1, 120), "lag"]
report_day = int(lag.median())

# Attach each transaction to the most recent fiscal quarter end of its own filer.
qe = (quarters[["ticker", "period_end"]].drop_duplicates()
      .assign(period_end=lambda d: pd.to_datetime(d["period_end"]))
      .sort_values("period_end"))
trades["transaction_date"] = pd.to_datetime(trades["transaction_date"]).dt.tz_localize(None)
trades = trades.sort_values("transaction_date")

m = pd.merge_asof(trades, qe, left_on="transaction_date", right_on="period_end",
                  by="ticker", direction="backward")
m["day"] = (m["transaction_date"] - m["period_end"]).dt.days
m = m[m["day"].between(0, 91)].copy()

sells = m[m["transaction_type"] == "open_market_sell"]
buys = m[m["transaction_type"] == "open_market_buy"]

phases = [("quarter closed, report not out", 0, report_day - 1),
          ("first month after the report", report_day, report_day + 29),
          ("run-up to the next quarter end", report_day + 30, 91)]

print(f"S&P 500 open-market insider transactions, {START} to {END}")
print(f"Universe: {len(tickers)} tickers   "
      f"Transactions placed in a mapped quarter: {len(m):,}")
print(f"Median 10-Q reporting lag: {report_day} days after fiscal quarter end "
      f"(n={len(lag):,}, IQR {lag.quantile(.25):.0f}-{lag.quantile(.75):.0f})\n")

print(f"{'Days after quarter end':<38}{'Sells':>8}{'/day':>9}{'Buys':>8}{'/day':>8}")
for name, lo, hi in phases:
    n = hi - lo + 1
    ns = int(sells["day"].between(lo, hi).sum())
    nb = int(buys["day"].between(lo, hi).sum())
    print(f"{f'{lo:>2}-{hi:<2}  {name}':<38}{ns:>8,}{ns / n:>9.1f}{nb:>8,}{nb / n:>8.1f}")

peak = sells.groupby("day").size()
quiet = int(sells["day"].between(80, 91).sum()) / 12
busy = int(sells["day"].between(report_day, report_day + 29).sum()) / 30
print(f"\nBusiest selling day: day {int(peak.idxmax())} ({peak.max():,} transactions)")
print(f"Quietest stretch: days 80-91 at {quiet:.1f} sells/day, "
      f"{100 * quiet / busy:.0f}% of the post-report rate")
print(f"Median transaction day: sells {sells['day'].median():.0f}, "
      f"buys {buys['day'].median():.0f}")
print(f"Sells per buy: {len(sells) / len(buys):.1f}")

officer = sells["insider_role"].str.contains(
    "CEO|CFO|COO|President|Officer", na=False)
director = sells["insider_role"].str.contains("Director", na=False) & ~officer
for name, sub in [("Officers", sells[officer]), ("Directors", sells[director])]:
    on = 100 * (sub["day"] >= report_day).mean()
    print(f"{name}: {len(sub):,} sells, median day {sub['day'].median():.0f}, "
          f"{on:.1f}% on or after day {report_day}")

d012 = sells[sells["day"] <= 2]
print(f"\nSells on days 0-2: {len(d012):,} ({100 * len(d012) / len(sells):.1f}% of sells) "
      f"from {d012['insider_name'].nunique()} insiders; "
      f"top 3 account for {d012['insider_name'].value_counts().head(3).sum():,}")

# ---- chart -----------------------------------------------------------------
def share(d):
    c = d.groupby("day").size().reindex(range(92), fill_value=0)
    return 100 * c / c.sum()

plt.style.use("dark_background")
fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, facecolor="#0a0a0a")
panels = [(axes[0], sells, "#3b82f6", f"Open-market sells ({len(sells):,})"),
          (axes[1], buys, "#f59e0b", f"Open-market buys ({len(buys):,})")]
for ax, data, colour, label in panels:
    ax.set_facecolor("#0a0a0a")
    ax.bar(range(92), share(data), color=colour, width=0.9)
    ax.axvline(report_day, color="#e0e0e0", ls="--", lw=1)
    ax.set_ylabel("Share of transactions (%)")
    ax.set_xlim(-1, 92)
    ax.tick_params(colors="#e0e0e0")
    ax.text(0.99, 0.92, label, transform=ax.transAxes, ha="right",
            color="#e0e0e0", fontsize=10)
    for s in ax.spines.values():
        s.set_color("#333333")
axes[0].text(report_day - 1.5, axes[0].get_ylim()[1] * 0.88,
             f"typical quarterly report, day {report_day}", color="#e0e0e0",
             fontsize=9, ha="right")
axes[0].set_title("When S&P 500 insiders place open-market trades")
axes[1].set_xlabel("Days after the fiscal quarter ends")
plt.tight_layout()
plt.savefig("when-do-insiders-trade-form-4-timing-python.png", dpi=150, facecolor="#0a0a0a")
