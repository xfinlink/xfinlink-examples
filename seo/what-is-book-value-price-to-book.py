# Full write-up: https://xfinlink.com/blog/what-is-book-value-price-to-book
#
# Book value, price-to-book, and what happens to P/B when research spending is
# treated as an asset instead of an expense. Fiscal 2024 annual filings, market
# capitalisation taken on each company's fiscal year-end trading day.

import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

TICKERS = ["MSFT", "GOOGL", "META", "MRK", "PFE", "AMGN", "XOM", "CAT", "KO", "PG"]
LIFE = 5  # years over which capitalised research is written off

f = xfl.fundamentals(TICKERS, period_type="annual", start="2019-01-01", end="2025-01-31",
                     fields=["total_equity", "goodwill", "research_and_development"])
f = f[f["fiscal_year"].between(2020, 2024)]
base = f[f["fiscal_year"] == 2024].set_index("ticker")

m = xfl.metrics(TICKERS, period_type="annual", fields=["market_cap", "pb_ratio"],
                start="2024-01-01", end="2025-01-31")

rows = []
for t in TICKERS:
    b = base.loc[t]
    hist = f[f["ticker"] == t].set_index("fiscal_year")["research_and_development"]
    # straight-line: this year's spend fully on the books, each earlier year 20% less
    rd_cap = sum(float(hist.get(2024 - i)) * (LIFE - i) / LIFE
                 for i in range(LIFE) if pd.notna(hist.get(2024 - i)))
    q = m[(m["ticker"] == t) & (m["period_end"] == b["period_end"])]
    mcap, eq, gw = float(q["market_cap"].iloc[0]), float(b["total_equity"]), float(b["goodwill"])
    rows.append(dict(ticker=t, sector=b["gics_sector"], equity=eq, goodwill=gw, rd_capital=rd_cap,
                     pb=mcap / eq, pb_api=float(q["pb_ratio"].iloc[0]),
                     pb_rd=mcap / (eq + rd_cap), goodwill_share=gw / eq))

t = pd.DataFrame(rows).set_index("ticker").sort_values("pb")
t["change_pct"] = (t["pb_rd"] / t["pb"] - 1) * 100

print(t[["sector", "equity", "goodwill", "rd_capital", "pb", "pb_rd",
         "change_pct", "goodwill_share"]].round(2).to_string())

# sanity check: the ratio computed here must match the API's own pb_ratio
print("\nmax gap vs xfl.metrics pb_ratio:", (t["pb"] - t["pb_api"]).abs().max().round(4))

# ---- chart ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 7), facecolor="#0a0a0a")
ax.set_facecolor("#0a0a0a")
y = range(len(t))
ax.hlines(y, t["pb_rd"], t["pb"], color="#3f3f46", linewidth=2, zorder=1)
ax.scatter(t["pb_rd"], y, s=70, color="#3b82f6", zorder=2, label="P/B with research capitalised")
ax.scatter(t["pb"], y, s=110, facecolors="none", edgecolors="#e0e0e0", linewidths=1.8,
           zorder=3, label="Reported P/B")
ax.set_yticks(list(y))
ax.set_yticklabels(t.index, color="#e0e0e0")
ax.set_xlabel("Price to book value", color="#e0e0e0")
ax.set_title("What capitalising research does to price-to-book\nFiscal 2024, 10 large caps",
             color="#e0e0e0", loc="left")
ax.tick_params(colors="#e0e0e0")
for s in ax.spines.values():
    s.set_color("#3f3f46")
ax.grid(axis="x", color="#1c1c1c", linewidth=0.8)
ax.set_axisbelow(True)
leg = ax.legend(facecolor="#0a0a0a", edgecolor="#3f3f46", loc="lower right")
for txt in leg.get_texts():
    txt.set_color("#e0e0e0")
plt.tight_layout()
plt.savefig("what-is-book-value-price-to-book.png", dpi=150, facecolor="#0a0a0a")
