# Full write-up: https://xfinlink.com/blog/technical-indicator-signal-correlation-python
"""Six standard technical rules, one price series: how often do they disagree?

Each rule is reduced to the same thing a trader acts on, a long or flat state
for the next session. Agreement is then measured pairwise, across the whole
toolkit, and separately on the most volatile days.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

TICKERS = ["SPY", "IWM", "EFA", "EEM", "TLT", "GLD"]
START, END = "2005-01-01", "2024-12-31"
RULES = ["50/200 cross", "price vs 200d", "12m momentum",
         "20d breakout", "MACD", "RSI(14)"]

px = xfl.prices(TICKERS, start="2003-06-01", end=END,
                fields=["adj_close", "return_daily"], max_rows=200000)
px = px.drop_duplicates(["ticker", "date"]).sort_values(["ticker", "date"])


def states(c):
    """Long (1) or flat (0) for each rule, from a split-adjusted close series."""
    s = pd.DataFrame(index=c.index)
    s["50/200 cross"] = (c.rolling(50).mean() > c.rolling(200).mean()).astype(float)
    s["price vs 200d"] = (c > c.rolling(200).mean()).astype(float)
    s["12m momentum"] = (c / c.shift(252) - 1 > 0).astype(float)

    hi, lo = c.shift(1).rolling(20).max(), c.shift(1).rolling(20).min()
    br = pd.Series(np.where(c > hi, 1.0, np.where(c < lo, 0.0, np.nan)), index=c.index)
    s["20d breakout"] = br.ffill()

    macd = c.ewm(span=12).mean() - c.ewm(span=26).mean()
    s["MACD"] = (macd > macd.ewm(span=9).mean()).astype(float)

    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / 14, min_periods=14).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, min_periods=14).mean()
    s["RSI(14)"] = (100 - 100 / (1 + up / dn) > 50).astype(float)
    return s[RULES]


panels, rows = {}, []
for t in TICKERS:
    d = px[px["ticker"] == t].set_index("date")
    s = states(d["adj_close"]).loc[START:END].dropna()
    panels[t] = (s, d["return_daily"].reindex(s.index))

    pairs = [(s[a] == s[b]).mean() for i, a in enumerate(RULES) for b in RULES[i + 1:]]
    agree_all = (s.sum(axis=1).isin([0, 6])).mean()
    vol = d["return_daily"].rolling(21).std().reindex(s.index)
    stress = vol >= vol.quantile(0.90)
    pairs_stress = [(s[a] == s[b])[stress].mean() for i, a in enumerate(RULES)
                    for b in RULES[i + 1:]]
    long_share = s.mean().mean()
    rows.append({"ticker": t, "n": len(s), "agree": np.mean(pairs),
                 "unanimous": agree_all, "stress": np.mean(pairs_stress),
                 "long": long_share})

res = pd.DataFrame(rows)
print(f"Six price-based rules, {START} to {END}, daily long or flat states\n")
print("ticker   sessions   mean pairwise    all six      same, top-decile   days")
print("                       agreement     agree        volatility         long")
for r in res.itertuples():
    print(f"{r.ticker:<9}{r.n:>7}{r.agree * 100:14.1f}%{r.unanimous * 100:12.1f}%"
          f"{r.stress * 100:17.1f}%{r.long * 100:14.1f}%")

s, ret = panels["SPY"]
mat = pd.DataFrame([[(s[a] == s[b]).mean() for b in RULES] for a in RULES],
                   index=RULES, columns=RULES)
print("\nSPY pairwise agreement, share of sessions in the same state")
print("                 " + "".join(f"{r[:11]:>13}" for r in RULES))
for a in RULES:
    print(f"{a:<17}" + "".join(f"{mat.loc[a, b] * 100:12.1f}%" for b in RULES))

share = s.mean()
base = np.mean([share[a] * share[b] + (1 - share[a]) * (1 - share[b])
                for i, a in enumerate(RULES) for b in RULES[i + 1:]])
print("\nSPY share of sessions each rule is long: "
      + ", ".join(f"{a} {share[a] * 100:.1f}%" for a in RULES))
print(f"mean pairwise agreement if the six rules were independent: {base * 100:.1f}%")

SLOW, FAST = RULES[:3], RULES[3:]
pair = lambda g, h: np.mean([mat.loc[a, b] for i, a in enumerate(g)
                             for b in (h[i + 1:] if g is h else h)])
print(f"\nSPY agreement within the three trend rules {pair(SLOW, SLOW) * 100:.1f}%, "
      f"within the three faster rules {pair(FAST, FAST) * 100:.1f}%, "
      f"between the two groups {pair(SLOW, FAST) * 100:.1f}%")

held = s.shift(1).dropna()
vote = (held.sum(axis=1) >= 4).astype(float)
solo = held["price vs 200d"]
r = ret.reindex(held.index)
print(f"\nSPY: the 200-day rule is long on {solo.mean() * 100:.1f}% of sessions, "
      f"a 4-of-6 vote on {vote.mean() * 100:.1f}%")
print(f"the two hold the same position on {(vote == solo).mean() * 100:.1f}% of sessions")
for name, pos in [("200-day rule", solo), ("4-of-6 vote", vote), ("buy and hold", 1.0)]:
    x = r * pos
    n = len(x)
    cagr = (1 + x).prod() ** (252 / n) - 1
    print(f"  {name:<14} return {cagr * 100:6.2f}%  volatility "
          f"{x.std() * np.sqrt(252) * 100:5.2f}%  Sharpe {cagr / (x.std() * np.sqrt(252)):.2f}")

# ── chart ─────────────────────────────────────────────────────────────
plt.rcParams.update({"figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
                     "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
                     "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
                     "axes.edgecolor": "#333333", "font.size": 8.5})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

im = ax1.imshow(mat.values * 100, cmap="Blues", vmin=50, vmax=100)
ax1.set_xticks(range(6), RULES, rotation=45, ha="right")
ax1.set_yticks(range(6), RULES)
for i in range(6):
    for j in range(6):
        v = mat.values[i, j] * 100
        ax1.text(j, i, f"{v:.0f}", ha="center", va="center",
                 color="#e0e0e0" if v > 80 else "#0a0a0a", fontsize=8)
ax1.set_title("SPY: sessions in the same state (%)")
fig.colorbar(im, ax=ax1, fraction=0.046)

x = np.arange(len(res))
ax2.bar(x - 0.2, res["agree"] * 100, width=0.38, color="#3b82f6", label="All sessions")
ax2.bar(x + 0.2, res["stress"] * 100, width=0.38, color="#f59e0b",
        label="Top-decile volatility")
ax2.set_xticks(x, res["ticker"])
ax2.set_ylim(55, 75)
ax2.set_ylabel("Mean pairwise agreement (%)")
ax2.set_title("Mean pairwise agreement, calm days and volatile days")
ax2.legend(frameon=False, labelcolor="#e0e0e0")
for ax in (ax1, ax2):
    for sp in ax.spines.values():
        sp.set_color("#333333")
plt.tight_layout()
plt.savefig("/home/user/xfinlink/worker/src/site/blog-images/"
            "technical-indicator-signal-correlation-python.png",
            dpi=120, facecolor="#0a0a0a")
