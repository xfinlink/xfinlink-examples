# Full write-up: https://xfinlink.com/blog/goodwill-adjusted-price-to-book-python
"""Does goodwill distort the price-to-book screen?

Strips goodwill out of shareholders' equity for the non-financial S&P 500 and
measures how far the resulting value ranking moves.

Built from SEC EDGAR public filings and market data.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

EXCLUDED_SECTORS = ["Financials", "Real Estate"]
MIN_MARKET_CAP = 1_000.0   # $m
OUT_PNG = "goodwill-adjusted-price-to-book-python.png"

# ---------------------------------------------------------------- data

members = xfl.index("sp500")
tickers = sorted(members["ticker"].dropna().unique().tolist())

fund = xfl.fundamentals(tickers, period_type="all", period="2y",
                        fields=["total_equity", "goodwill"])

# One balance sheet per company: the latest filing stating both figures.
stated = fund[fund["total_equity"].notna() & fund["goodwill"].notna()]
book = stated.sort_values("period_end").groupby("entity_id").tail(1)

caps = xfl.metrics(tickers, period_type="daily", fields=["market_cap"], period="1w")
caps = caps.sort_values("period_end").groupby("entity_id").tail(1)
price_date = caps["period_end"].max()
caps = caps[["entity_id", "market_cap"]]

# Symbols come off the traded price series, keyed on entity identifier.
px = xfl.prices(tickers, period="1w", fields=["close"])
sym = px.sort_values("date").groupby("entity_id").tail(1)[["entity_id", "ticker"]]

cutoff = price_date - pd.DateOffset(months=13)
book = book[book["period_end"] >= cutoff]
book = book[~book["gics_sector"].isin(EXCLUDED_SECTORS)]

df = book.merge(sym, on="entity_id", how="left", suffixes=("_roster", ""))
df["ticker"] = df["ticker"].fillna(df["ticker_roster"])
df = df.merge(caps, on="entity_id")
df = df[(df["total_equity"] > 0) & (df["market_cap"] >= MIN_MARKET_CAP)].copy()

# ---------------------------------------------------------------- measures

df["gw_share"] = df["goodwill"] / df["total_equity"]
df["book_ex"] = df["total_equity"] - df["goodwill"]
df["pb"] = df["market_cap"] / df["total_equity"]
df["q_all"] = pd.qcut(df["pb"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)

wiped = df[df["book_ex"] <= 0]

# Names keeping tangible net worth carry both ratios, so both rankings exist.
tan = df[df["book_ex"] > 0].copy()
tan["pb_ex"] = tan["market_cap"] / tan["book_ex"]
tan["q_pb"] = pd.qcut(tan["pb"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
tan["q_ex"] = pd.qcut(tan["pb_ex"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)

rho, pval = spearmanr(tan["pb"], tan["pb_ex"])
cheap = tan[tan["q_pb"] == 1]
landing = cheap["q_ex"].value_counts().reindex([1, 2, 3, 4, 5], fill_value=0)
moved = int((tan["q_ex"] != tan["q_pb"]).sum())

sector = (df.groupby("gics_sector")
            .agg(n=("ticker", "size"),
                 gw_share=("gw_share", "median"),
                 wiped=("book_ex", lambda s: int((s <= 0).sum())),
                 pb=("pb", "median"))
            .sort_values("gw_share", ascending=False))
sector["pb_ex"] = tan.groupby("gics_sector")["pb_ex"].median()

# ---------------------------------------------------------------- output

print(f"Goodwill and the price-to-book screen, S&P 500, market data {price_date:%Y-%m-%d}")
print("Book equity: latest filing stating both shareholders' equity and goodwill, "
      f"on or after {cutoff:%Y-%m-%d}")
print(f"Sample: {len(df)} members with positive book equity and a market capitalisation "
      f"above ${MIN_MARKET_CAP:,.0f}m, outside {' and '.join(EXCLUDED_SECTORS)}")
print()
print(f"Median goodwill as a share of book equity: {df['gw_share'].median():.2f}")
print(f"Goodwill larger than the whole of book equity: {len(wiped)} names "
      f"({len(wiped) / len(df):.1%}) -- no tangible net worth, no ratio to compute")
print(f"Of the {(df['q_all'] == 1).sum()} names in the cheapest reported-P/B quintile, "
      f"{int(((df['q_all'] == 1) & (df['book_ex'] <= 0)).sum())} are in that group")
print()

print(f"Ranking test on the {len(tan)} names that keep tangible net worth")
print(f"  median P/B {tan['pb'].median():.2f}, median P/B ex-goodwill {tan['pb_ex'].median():.2f}")
print(f"  Spearman rank correlation between the two: {rho:.3f}  p = {pval:.1e}")
print(f"  {moved} of {len(tan)} names change quintile")
print(f"  cheapest P/B quintile ({len(cheap)} names) lands as:")
for q in [1, 2, 3, 4, 5]:
    label = {1: " (cheapest)", 5: " (most expensive)"}.get(q, "")
    print(f"     quintile {q}{label:<19}{landing[q]:>4}")
print()

screened = df[(df["q_all"] == 1) & (df["book_ex"] <= 0)].sort_values("gw_share", ascending=False)
print("Cheapest reported-P/B quintile, names whose book equity is entirely goodwill")
print(f"{'Ticker':<8}{'Sector':<24}{'P/B':>6}{'GW/Book':>9}{'Book $m':>11}{'Book ex-GW $m':>15}")
for _, r in screened.iterrows():
    print(f"{r['ticker']:<8}{r['gics_sector'][:23]:<24}{r['pb']:>6.2f}{r['gw_share']:>9.2f}"
          f"{r['total_equity']:>11,.1f}{r['book_ex']:>15,.1f}")
print()

top = cheap.sort_values("gw_share", ascending=False).head(10)
print("Cheapest-quintile names that survive the adjustment, most goodwill first")
print(f"{'Ticker':<8}{'Sector':<24}{'P/B':>6}{'GW/Book':>9}{'P/B ex-GW':>11}{'Quintile':>10}")
for _, r in top.iterrows():
    print(f"{r['ticker']:<8}{r['gics_sector'][:23]:<24}{r['pb']:>6.2f}{r['gw_share']:>9.2f}"
          f"{r['pb_ex']:>11.2f}{r['q_ex']:>10}")
print()

print("Goodwill in book equity by sector")
print(f"{'Sector':<24}{'n':>5}{'GW/Book':>9}{'No tangible':>13}{'P/B':>8}{'P/B ex-GW':>11}")
for name, r in sector.iterrows():
    print(f"{name[:23]:<24}{int(r['n']):>5}{r['gw_share']:>9.2f}{int(r['wiped']):>13}"
          f"{r['pb']:>8.2f}{r['pb_ex']:>11.2f}")

# ---------------------------------------------------------------- chart

BG, FG, ACCENT, WARN = "#0a0a0a", "#e0e0e0", "#3b82f6", "#f59e0b"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": FG, "axes.labelcolor": FG, "xtick.color": FG, "ytick.color": FG,
    "axes.edgecolor": "#333333", "font.size": 10,
})

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7),
                               gridspec_kw={"height_ratios": [1.25, 1]})

pan = cheap.sort_values("gw_share").tail(12)
y = np.arange(len(pan))
ax1.hlines(y, pan["pb"], pan["pb_ex"], color="#3f3f46", linewidth=2, zorder=1)
ax1.scatter(pan["pb"], y, s=52, color=ACCENT, zorder=2, label="P/B as reported")
ax1.scatter(pan["pb_ex"], y, s=52, color=WARN, zorder=2, label="P/B after removing goodwill")
ax1.set_yticks(y)
ax1.set_yticklabels(pan["ticker"])
ax1.set_xscale("log")
ax1.set_xlabel("Price to book value (log scale)")
ax1.set_title("Cheapest S&P 500 price-to-book quintile: where goodwill was doing the work",
              color=FG, fontsize=12, pad=10)
ax1.legend(frameon=False, loc="lower right", fontsize=9)
ax1.grid(axis="x", color="#262626", linewidth=0.6)
ax1.set_axisbelow(True)
for s in ("top", "right", "left"):
    ax1.spines[s].set_visible(False)

sec = sector.sort_values("gw_share")
ax2.barh(sec.index, sec["gw_share"], color=ACCENT, height=0.66)
ax2.set_xlabel("Median goodwill as a share of book equity")
ax2.set_xlim(0, sec["gw_share"].max() * 1.15)
ax2.set_title("How much of book equity is goodwill, by sector", color=FG, fontsize=12, pad=10)
ax2.grid(axis="x", color="#262626", linewidth=0.6)
ax2.set_axisbelow(True)
for s in ("top", "right", "left"):
    ax2.spines[s].set_visible(False)
for i, v in enumerate(sec["gw_share"]):
    ax2.text(v + 0.015, i, f"{v:.2f}", va="center", color=FG, fontsize=9)

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150)
print(f"\nchart saved: {OUT_PNG}")
