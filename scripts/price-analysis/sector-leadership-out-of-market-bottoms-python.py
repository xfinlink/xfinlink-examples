# Full write-up: https://xfinlink.com/blog/sector-leadership-out-of-market-bottoms-python
"""Which sectors lead the market out of a bottom?

Finds every S&P 500 drawdown of 15% or more since 1999, dates the trough, then measures
what each sector fund returned over the following 3, 6 and 12 months against the index.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

SECTORS = {
    "XLK": "Technology", "XLF": "Financials", "XLE": "Energy", "XLV": "Health Care",
    "XLI": "Industrials", "XLY": "Cons. Discretionary", "XLP": "Cons. Staples",
    "XLU": "Utilities", "XLB": "Materials", "XLRE": "Real Estate",
    "XLC": "Communication Svs",
}
THRESHOLD = 0.15
HORIZONS = {"3m": 63, "6m": 126, "12m": 252}

px = xfl.prices(["SPY"] + list(SECTORS), start="1999-01-01", end="2026-08-21",
                fields=["return_daily"], max_rows=300000)
px["date"] = pd.to_datetime(px["date"])
ret = px.pivot_table(index="date", columns="ticker", values="return_daily").sort_index()
spy = ret["SPY"].dropna()

# Cumulative total return index for SPY, then peak-to-trough episodes.
level = (1 + spy).cumprod()
peak = level.cummax()
drawdown = level / peak - 1

episodes = []
in_ep, ep_peak_date = False, None
for d, dd in drawdown.items():
    if not in_ep and dd <= -THRESHOLD:
        in_ep = True
        # The peak that started this decline is the last date at a running high.
        ep_peak_date = peak.loc[:d].idxmax()
        ep_peak_date = level.loc[:d][level.loc[:d] == peak.loc[d]].index[-1]
    elif in_ep and dd >= 0:
        # Recovered to a new high: the episode's trough is the low in between.
        seg = drawdown.loc[ep_peak_date:d]
        episodes.append((ep_peak_date, seg.idxmin(), seg.min()))
        in_ep = False
if in_ep:
    seg = drawdown.loc[ep_peak_date:]
    episodes.append((ep_peak_date, seg.idxmin(), seg.min()))

print("S&P 500 drawdowns of 15% or more since 1999")
print(f"{'peak':>12} {'trough':>12} {'depth':>8}")
for p, t, m in episodes:
    print(f"{p.date()!s:>12} {t.date()!s:>12} {m * 100:7.1f}%")

dates = ret.index
pos = {d: i for i, d in enumerate(dates)}

rows = []
for _, trough, depth in episodes:
    i = pos[trough]
    for h, n in HORIZONS.items():
        if i + n >= len(dates):
            continue
        window = ret.iloc[i + 1:i + 1 + n]
        # A fund only enters an episode if it traded for the whole forward window.
        fwd = (1 + window).prod() - 1
        valid = window.notna().all()
        bench = fwd["SPY"]
        for tk in SECTORS:
            if valid.get(tk, False):
                rows.append({"trough": trough.date(), "horizon": h, "ticker": tk,
                             "sector": SECTORS[tk], "excess": (fwd[tk] - bench) * 100})

panel = pd.DataFrame(rows)
table = panel.pivot_table(index="sector", columns="horizon", values="excess", aggfunc="mean")
counts = panel[panel["horizon"] == "12m"].groupby("sector").size()
table = table[["3m", "6m", "12m"]]
table["episodes_12m"] = counts
table = table.sort_values("12m", ascending=False)

print("\nMean return against SPY from the trough (percentage points)")
print(table.round(2).to_string())

print("\nHit rate: share of episodes each sector beat SPY over 12 months")
hit = panel[panel["horizon"] == "12m"].groupby("sector")["excess"].agg(
    beat=lambda s: (s > 0).mean() * 100, n="size").round(1).sort_values("beat", ascending=False)
print(hit.to_string())

print("\n12-month excess by episode (percentage points)")
by_ep = panel[panel["horizon"] == "12m"].pivot_table(
    index="trough", columns="sector", values="excess")
print(by_ep.round(1).to_string())

# ── chart ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
    "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
    "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
    "axes.edgecolor": "#333333", "font.size": 9,
})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 7))
order = table.sort_values("12m")
colours = ["#3b82f6" if v > 0 else "#6b7280" for v in order["12m"]]
ax1.barh(order.index, order["12m"], color=colours)
ax1.axvline(0, color="#333333", lw=0.8)
ax1.set_xlabel("Mean 12-month return vs SPY (pp)")
ax1.set_title("Average leadership after a bottom")

h = hit.sort_values("beat")
ax2.barh(h.index, h["beat"], color="#f59e0b")
ax2.axvline(50, color="#666666", lw=0.8, ls="--")
ax2.set_xlabel("Share of episodes beating SPY over 12 months (%)")
ax2.set_title("How often, not just how much")
ax2.set_xlim(0, 100)

plt.tight_layout()
plt.savefig("sector-leadership-out-of-market-bottoms-python.png", dpi=150, facecolor="#0a0a0a")
