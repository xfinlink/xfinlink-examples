# Full write-up: https://xfinlink.com/blog/sp500-index-weight-concentration-python
"""How concentrated is the S&P 500? Index weight analysis in Python.

Takes the S&P 500 roster by entity id, joins each member to its market value on
a single trading day, and measures how much of the index a handful of members
carry: the weight of each rank band, the Herfindahl-Hirschman index of the
weights, and the effective number of members that implies.
"""
import warnings

import matplotlib
import numpy as np
import pandas as pd
import xfinlink as xfl

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.simplefilter("ignore")
xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

SNAPSHOT = "2026-08-21"
WINDOW_START = "2026-08-14"

# A member's market value here is its total share count at the price of its
# principal listed line. That holds wherever a company's classes trade at
# comparable prices, which is the normal case. Berkshire Hathaway is the
# exception in this index: its Class A and Class B lines trade at a ratio of
# about 1,500 to 1, so no single per-share price values the combined count,
# and the company sits outside the sample.
NON_COMPARABLE_CLASSES = {"BRK"}

# ── Data ──────────────────────────────────────────────────────────────
# Members are addressed by entity id, not by ticker, so a symbol change
# between the roster and the market data cannot break the join.
roster = xfl.index("sp500").dropna(subset=["entity_id"])
ids = sorted(set(roster["entity_id"].astype(int)))

caps, sectors = [], []
for i in range(0, len(ids), 100):
    batch = ids[i:i + 100]
    caps.append(
        xfl.metrics(entity_id=batch, period_type="daily", fields=["market_cap"],
                    start=WINDOW_START, end=SNAPSHOT)
    )
    sectors.append(
        xfl.prices(entity_id=batch, start=WINDOW_START, end=SNAPSHOT,
                   fields=["close", "gics_sector"])
    )

cap = pd.concat([c for c in caps if len(c)], ignore_index=True)
cap = cap.dropna(subset=["market_cap"])
cap = cap[cap["market_cap"] > 0]
cap = cap.sort_values("period_end").groupby("entity_id", as_index=False).last()

sec = pd.concat([s for s in sectors if len(s)], ignore_index=True)
sec = sec.sort_values("date").groupby("entity_id", as_index=False).last()

df = cap[["entity_id", "ticker", "entity_name", "market_cap"]].merge(
    sec[["entity_id", "gics_sector"]], on="entity_id", how="left")
df = df[~df["ticker"].isin(NON_COMPARABLE_CLASSES)]
df = df.sort_values("market_cap", ascending=False).reset_index(drop=True)

# ── Concentration measures ────────────────────────────────────────────
total = df["market_cap"].sum()
df["weight"] = 100 * df["market_cap"] / total
df["cumulative"] = df["weight"].cumsum()

n = len(df)
hhi = float((df["weight"] ** 2).sum())            # weights in percentage points
effective_n = 10000 / hhi                          # equally weighted equivalent
half_n = n // 2
smallest_half = float(df["weight"].tail(half_n).sum())
bands = (1, 3, 5, 10, 25, 50, 100)

print(f"S&P 500 market value concentration, {SNAPSHOT}")
print(f"members on the roster       : {len(ids)}")
print(f"members in the sample       : {n}")
print(f"combined market value       : ${total / 1e6:,.2f}tn")
print()
print(f"{'rank band':<22}{'combined weight':>16}")
print("-" * 38)
for k in bands:
    print(f"{'largest ' + str(k):<22}{df['weight'].head(k).sum():>15.1f}%")
print(f"{'smallest ' + str(half_n):<22}{smallest_half:>15.1f}%")
print()
print(f"Herfindahl-Hirschman index  : {hhi:>8.1f}")
print(f"effective number of members : {effective_n:>8.1f}")
print(f"equal weight per member     : {100 / n:>8.3f}%")
print(f"median member weight        : {df['weight'].median():>8.3f}%")
print(f"members above 5% of index   : {int((df['weight'] > 5).sum()):>8}")
print(f"members above 1% of index   : {int((df['weight'] > 1).sum()):>8}")

print("\nFive largest members")
print(f"{'#':<3}{'ticker':<8}{'market value':>16}{'weight':>9}{'cumulative':>12}")
for i in range(5):
    r = df.loc[i]
    print(f"{i + 1:<3}{r['ticker']:<8}{r['market_cap'] / 1e6:>15,.2f}t"
          f"{r['weight']:>8.2f}%{r['cumulative']:>11.2f}%")

by_sector = df.groupby("gics_sector").agg(
    members=("weight", "size"), value_share=("weight", "sum")).sort_values(
    "value_share", ascending=False)
by_sector["member_share"] = 100 * by_sector["members"] / n
print("\nSector split: share of members against share of index value")
print(f"{'sector':<26}{'members':>9}{'of members':>12}{'of value':>10}")
print("-" * 57)
for name, r in by_sector.iterrows():
    print(f"{name:<26}{int(r['members']):>9}{r['member_share']:>11.1f}%{r['value_share']:>9.1f}%")

# ── Chart ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
    "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
    "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
    "axes.edgecolor": "#333333", "font.size": 10,
})
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7),
                               gridspec_kw={"height_ratios": [1.25, 1]})

rank = np.arange(1, n + 1)
mark = 25
ax1.plot(rank, df["cumulative"], color="#3b82f6", lw=2.2)
ax1.plot([1, n], [100 / n, 100], color="#9ca3af", lw=1.4, ls="--",
         label="Equal weighting")
ax1.plot([mark], [df["cumulative"].iloc[mark - 1]], "o", color="#3b82f6", ms=6)
ax1.annotate(f"the largest {mark} members hold "
             f"{df['cumulative'].iloc[mark - 1]:.1f}% of index value",
             xy=(mark, df["cumulative"].iloc[mark - 1]),
             xytext=(mark + 60, 33), color="#e0e0e0", fontsize=10,
             arrowprops=dict(arrowstyle="-", color="#4b5563", lw=1))
ax1.set_xlabel("Members, largest first")
ax1.set_ylabel("Share of index value (%)")
ax1.set_xlim(0, n)
ax1.set_ylim(0, 100)
ax1.set_title(f"{n} S&P 500 members carry the concentration of "
              f"{effective_n:.0f} equally sized companies",
              color="#e0e0e0", fontsize=12, pad=12)
ax1.legend(frameon=False, loc="lower right")

order = by_sector.index[::-1]
y = np.arange(len(order))
ax2.barh(y + 0.19, by_sector.loc[order, "value_share"], height=0.38,
         color="#3b82f6", label="Share of index value")
ax2.barh(y - 0.19, by_sector.loc[order, "member_share"], height=0.38,
         color="#9ca3af", label="Share of members")
ax2.set_yticks(y)
ax2.set_yticklabels(order, fontsize=9)
ax2.set_xlabel("Percent")
ax2.legend(frameon=False, loc="lower right")

for ax in (ax1, ax2):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("sp500-index-weight-concentration-python.png", dpi=150,
            facecolor="#0a0a0a")
