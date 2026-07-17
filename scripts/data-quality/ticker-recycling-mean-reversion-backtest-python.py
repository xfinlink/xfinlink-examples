# Full write-up: https://xfinlink.com/blog/ticker-recycling-mean-reversion-backtest-python
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

START, END = "2016-08-01", "2017-02-28"
WINDOW, THRESH, HOLD = 20, 2.0, 5
SPLICE = pd.Timestamp("2016-11-01")

# --- 1. Entity history: who has held the symbol "AA"? ---
print("=" * 74)
print("ENTITY HISTORY FOR SYMBOL 'AA'")
print("=" * 74)
for e in xfl.resolve("AA")["data"]["AA"]["entities"]:
    print(f"  {e['name']:22s} CIK {e['cik']}  {e['ticker_valid_from']} -> "
          f"{e['ticker_valid_to'] or 'present'}  ({e['classifications']['gics_sector']})")

# --- 2. Pull both entities. prices('AA') returns ONLY the current holder
#        (Alcoa Corp). The prior holder is reached via its own symbol, HWM. ---
old = xfl.prices("HWM", start=START, end=END, fields=["adj_close"])   # entity 294
new = xfl.prices("AA",  start=START, end=END, fields=["adj_close"])   # entity 15812
both = pd.concat([old, new], ignore_index=True)

# NAIVE = select on the SYMBOL. This reconstructs symbol-keyed data:
# whatever company held "AA" on each date. The ticker column is point-in-time.
naive = both[both["ticker"] == "AA"].sort_values("date").reset_index(drop=True)
# RESOLVED = select on the ENTITY. One company, no splice.
resolved = both[both["entity_id"] == new["entity_id"].iloc[0]].sort_values("date").reset_index(drop=True)

def zscore(df):
    d = df.copy()
    m = d["adj_close"].rolling(WINDOW).mean()
    s = d["adj_close"].rolling(WINDOW).std()
    d["z"] = (d["adj_close"] - m) / s
    return d

naive, resolved = zscore(naive), zscore(resolved)

print("\nNaive symbol-keyed series: %d rows, %s -> %s" %
      (len(naive), naive["date"].min().date(), naive["date"].max().date()))
for name, g in naive.groupby("entity_name"):
    print(f"   {name:22s} {g['date'].min().date()} -> {g['date'].max().date()}  ({len(g)} days)")
print("Entity-resolved series:    %d rows, %s -> %s" %
      (len(resolved), resolved["date"].min().date(), resolved["date"].max().date()))

# --- 3. The splice ---
prev = naive[naive["date"] < SPLICE].iloc[-1]
first = naive[naive["date"] == SPLICE].iloc[0]
print("\n" + "=" * 74)
print("THE SPLICE (no gap: consecutive trading days)")
print("=" * 74)
print(f"  {prev['date'].date()}  AA = {prev['adj_close']:6.2f}   {prev['entity_name']}")
print(f"  {first['date'].date()}  AA = {first['adj_close']:6.2f}   {first['entity_name']}")
print(f"  Implied one-day return: {100*(first['adj_close']/prev['adj_close']-1):+.2f}%  "
      f"-- earned by no shareholder of either company")

# --- 4. Signals in the 20 days after the splice ---
win = naive[naive["date"] >= SPLICE].head(WINDOW)
winr = resolved[resolved["date"] >= SPLICE].head(WINDOW)
sigs = win[win["z"].abs() > THRESH]
print("\n" + "=" * 74)
print(f"FIRST {WINDOW} TRADING DAYS AFTER SPLICE ({win['date'].min().date()} -> {win['date'].max().date()})")
print("=" * 74)
print(f"  Naive    : {win['z'].notna().sum():2d} days scored, {len(sigs)} signals (|z|>{THRESH}), max|z| = {win['z'].abs().max():.2f}")
print(f"  Resolved : {winr['z'].notna().sum():2d} days scored, "
      f"{int((winr['z'].abs() > THRESH).sum())} signals -- insufficient history to score the rest")

# --- 5. What the phantom signals would have traded (on real Alcoa Corp prices) ---
real = resolved.set_index("date")["adj_close"]
print("\n  Phantom trades (z > +2 => mean-reversion SHORT, %d-day hold):" % HOLD)
pnl = []
for _, r in sigs.iterrows():
    i = real.index.get_loc(r["date"])
    if i + HOLD < len(real):
        entry, exit_ = real.iloc[i], real.iloc[i + HOLD]
        p = -(exit_ / entry - 1) * 100
        pnl.append(p)
        print(f"    {r['date'].date()}  z=+{r['z']:.2f}  short {entry:5.2f} -> cover {exit_:5.2f}   {p:+7.2f}%")
print(f"\n  {len(pnl)} phantom shorts | mean {np.mean(pnl):+.2f}% | cumulative {np.sum(pnl):+.2f}% | win rate {100*np.mean([x>0 for x in pnl]):.0f}%")
print(f"  Alcoa Corp over the same stretch: {real.loc['2016-11-01']:.2f} -> {real.loc['2016-11-17']:.2f} "
      f"({100*(real.loc['2016-11-17']/real.loc['2016-11-01']-1):+.1f}%)")

# --- 6. Other S&P 500 symbols with the same structure. Each leg was checked by hand
#        against SEC EDGAR: two distinct filer CIKs, a rename or exit event on the prior
#        filer consistent with the handover date, and the current holder of the symbol
#        confirmed in EDGAR's company_tickers registry. ---
print("\n" + "=" * 74)
print("OTHER S&P 500 SYMBOLS WITH AN EDGAR-CONFIRMED HANDOVER BETWEEN DISTINCT FILERS")
print("=" * 74)
print("  (entity_name is the filer's CURRENT legal name, not its name at the time)")
for t in ["GM", "DELL", "BG"]:
    print(f"  {t}:")
    for e in xfl.resolve(t)["data"][t]["entities"]:
        print(f"     {e['name'][:48]:48s} CIK {e['cik']}  "
              f"{e['ticker_valid_from']} -> {e['ticker_valid_to'] or 'present'}")

# --- Chart ---
plt.rcParams.update({
    "figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
    "savefig.facecolor": "#0a0a0a", "text.color": "#e0e0e0",
    "axes.labelcolor": "#e0e0e0", "xtick.color": "#e0e0e0",
    "ytick.color": "#e0e0e0", "axes.edgecolor": "#3a3a3a", "font.size": 9,
})
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

pre = naive[naive["date"] <= SPLICE]
ax1.plot(pre["date"], pre["adj_close"], color="#ef4444", lw=1.5)
ax1.plot(naive[naive["date"] >= SPLICE]["date"], naive[naive["date"] >= SPLICE]["adj_close"],
         color="#ef4444", lw=1.5, label="Naive symbol-keyed 'AA' (two companies spliced)")
ax1.plot(resolved["date"], resolved["adj_close"], color="#3b82f6", lw=2.2,
         label="Entity-resolved: Alcoa Corp only")
ax1.axvline(SPLICE, color="#e0e0e0", ls="--", lw=1)
ax1.annotate("Symbol handed over\nAlcoa Inc -> Alcoa Corp", xy=(SPLICE, 21.5),
             xytext=(pd.Timestamp("2016-08-20"), 27), color="#e0e0e0",
             arrowprops=dict(arrowstyle="->", color="#e0e0e0", lw=1))
ax1.set_ylabel("Adjusted close (USD)")
ax1.set_title("Ticker recycling corrupts a mean-reversion signal: symbol 'AA', 2016", color="#e0e0e0")
ax1.legend(facecolor="#0a0a0a", edgecolor="#3a3a3a", labelcolor="#e0e0e0", loc="upper left")
for s in ax1.spines.values(): s.set_color("#3a3a3a")

ax2.plot(naive["date"], naive["z"], color="#ef4444", lw=1.5, label="Naive z-score")
ax2.plot(resolved["date"], resolved["z"], color="#3b82f6", lw=2.2, label="Entity-resolved z-score")
ax2.scatter(sigs["date"], sigs["z"], color="#fbbf24", s=45, zorder=5,
            label=f"Phantom signals (|z|>{THRESH:.0f}): {len(sigs)}")
ax2.axhline(THRESH, color="#666", ls=":", lw=1)
ax2.axhline(-THRESH, color="#666", ls=":", lw=1)
ax2.axvline(SPLICE, color="#e0e0e0", ls="--", lw=1)
ax2.set_ylabel(f"{WINDOW}-day z-score")
ax2.set_xlabel("Date")
ax2.legend(facecolor="#0a0a0a", edgecolor="#3a3a3a", labelcolor="#e0e0e0", loc="upper left")
for s in ax2.spines.values(): s.set_color("#3a3a3a")

plt.tight_layout()
out = "ticker-recycling-mean-reversion-backtest-python.png"
plt.savefig(out, dpi=150)
print(f"\nchart -> {out}")
