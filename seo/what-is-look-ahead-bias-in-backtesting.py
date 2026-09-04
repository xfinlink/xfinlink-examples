"""How much price action sits between a fiscal period end and the date the
annual report is certainly public (the 60-day Form 10-K deadline for a large
accelerated filer). Companion script for the guide
"What Is Look-Ahead Bias in Backtesting?".
"""

import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

tickers = ["MSFT", "ORCL", "NVDA", "WMT", "AAPL", "COST"]

f = xfl.fundamentals(tickers, period_type="annual", fields=["revenue"], start="2025-01-01")
f["period_end"] = pd.to_datetime(f["period_end"])
f = f.sort_values("period_end").groupby("ticker").tail(1)
f["usable_from"] = f["period_end"] + pd.Timedelta(days=60)  # Form 10-K deadline

p = xfl.prices(tickers, start="2025-08-01", end="2026-09-03", fields=["adj_close"])
p["date"] = pd.to_datetime(p["date"])

rows = []
for r in f.itertuples():
    s = p[p["ticker"] == r.ticker].sort_values("date")
    at_end = s[s["date"] >= r.period_end].iloc[0]["adj_close"]
    at_pub = s[s["date"] >= r.usable_from].iloc[0]["adj_close"]
    rows.append({"ticker": r.ticker, "fiscal_year": r.fiscal_year,
                 "period_end": r.period_end.date(), "revenue_musd": r.revenue,
                 "usable_from": r.usable_from.date(),
                 "price_move_pct": round(100 * (at_pub / at_end - 1), 1)})

print(pd.DataFrame(rows).sort_values("period_end").to_string(index=False))

# Same measurement across the current S&P 500 roster: for each company's most
# recent annual period whose 60-day window has closed, the absolute price move
# from period end to that date. 495 of 504 members measure; median 11.1%.
roster = xfl.index("sp500")
eids = [int(e) for e in roster["entity_id"].unique()]

fun = pd.concat([xfl.fundamentals(entity_id=eids[i:i + 100], period_type="annual",
                                  fields=["revenue"], start="2025-06-01")
                 for i in range(0, len(eids), 100)], ignore_index=True)
fun["period_end"] = pd.to_datetime(fun["period_end"])
fun = fun[fun["period_end"] + pd.Timedelta(days=60) <= pd.Timestamp("2026-09-03")]
latest = fun.sort_values("period_end").groupby("entity_id").tail(1)

moves = []
for pe, grp in latest.groupby("period_end"):
    ids = [int(e) for e in grp["entity_id"]]
    px = pd.concat([xfl.prices(entity_id=ids[i:i + 100], start=pe.date().isoformat(),
                               end=(pe + pd.Timedelta(days=75)).date().isoformat(),
                               fields=["adj_close"], max_rows=100000)
                    for i in range(0, len(ids), 100)], ignore_index=True)
    px["date"] = pd.to_datetime(px["date"])
    for eid in ids:
        s = px[px["entity_id"] == eid].sort_values("date").dropna(subset=["adj_close"])
        a = s[s["date"] >= pe]
        b = s[s["date"] >= pe + pd.Timedelta(days=60)]
        if a.empty or b.empty or a.iloc[0]["adj_close"] <= 0:
            continue
        moves.append(abs(100 * (b.iloc[0]["adj_close"] / a.iloc[0]["adj_close"] - 1)))

m = pd.Series(moves)
print(f"\nsample {len(m)} | median {m.median():.1f}% | quartiles {m.quantile(0.25):.1f}%"
      f" / {m.quantile(0.75):.1f}% | share above 10% {100 * (m > 10).mean():.1f}%")
