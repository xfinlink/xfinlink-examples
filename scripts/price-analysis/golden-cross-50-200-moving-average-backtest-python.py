# Full write-up: https://xfinlink.com/blog/golden-cross-50-200-moving-average-backtest-python
"""Does the golden cross work? A 50/200 moving average crossover test on eight
liquid ETFs, 2005-2024, measured against buy and hold on return, risk and drawdown."""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

TICKERS = ["SPY", "IWM", "EFA", "EEM", "XLK", "XLE", "TLT", "LQD"]
LABEL = {"SPY": "US large cap", "IWM": "US small cap", "EFA": "Developed ex-US",
         "EEM": "Emerging markets", "XLK": "Technology", "XLE": "Energy",
         "TLT": "Long Treasuries", "LQD": "IG corporate bonds"}
WARMUP, START, END = "2004-01-01", "2005-01-01", "2024-12-31"
IMG = "golden-cross-50-200-moving-average-backtest-python.png"

# ── Data ──────────────────────────────────────────────────────────────
px = xfl.prices(TICKERS, start=WARMUP, end=END,
                fields=["close", "adj_close", "return_daily"])
px = px.drop_duplicates(["ticker", "date"]).sort_values(["ticker", "date"])
px = px[px["adj_close"] > 0]

sessions = set(px.loc[px["ticker"] == "SPY", "date"])
panel = {}
for t in TICKERS:
    g = px[px["ticker"] == t].set_index("date")
    have = set(g.index)
    missing = len(sessions - have)
    extreme = int((g["return_daily"].abs() > 0.50).sum())
    if missing or extreme or g["return_daily"].isna().sum():
        print(f"  {t} set aside: {missing} sessions absent, {extreme} out-of-band returns")
        continue
    panel[t] = g

# ── Signal and performance ────────────────────────────────────────────
def stats(r):
    """CAGR, annualised volatility, Sharpe (cash at zero) and worst drawdown."""
    n = len(r)
    cagr = (1 + r).prod() ** (252 / n) - 1
    vol = r.std() * np.sqrt(252)
    curve = (1 + r).cumprod()
    return cagr, vol, cagr / vol, (curve / curve.cummax() - 1).min()


def run(g, fast=50, slow=200, start=START, end=END):
    """Long when the fast average sits above the slow one, otherwise flat."""
    sma_f = g["adj_close"].rolling(fast).mean()
    sma_s = g["adj_close"].rolling(slow).mean()
    hold = (sma_f > sma_s).shift(1)          # yesterday's close decides today
    w = g.loc[start:end]
    hold = hold.loc[start:end].fillna(False)
    bh = w["return_daily"]
    rule = bh.where(hold, 0.0)
    return bh, rule, hold


rows, curves = [], {}
for t, g in panel.items():
    bh, rule, hold = run(g)
    trades = int((hold.astype(int).diff().abs() == 1).sum())
    rows.append([t, *stats(bh), *stats(rule), hold.mean(), trades])
    curves[t] = ((1 + bh).cumprod(), (1 + rule).cumprod())

cols = ["ticker", "bh_cagr", "bh_vol", "bh_sharpe", "bh_dd",
        "r_cagr", "r_vol", "r_sharpe", "r_dd", "exposure", "trades"]
res = pd.DataFrame(rows, columns=cols).set_index("ticker")

# ── Output ────────────────────────────────────────────────────────────
print(f"\nGolden cross (50/200 day) versus buy and hold, {START} to {END}")
print(f"{len(res)} ETFs, daily total returns, out-of-market days earn nothing\n")
print(f"{'':4} {'':19} {'buy and hold':>29}   {'50/200 crossover rule':>29}")
print(f"{'':4} {'':19} {'CAGR':>7}{'Vol':>8}{'Sharpe':>8}{'MaxDD':>7}   "
      f"{'CAGR':>7}{'Vol':>8}{'Sharpe':>8}{'MaxDD':>7}{'InMkt':>7}{'Trades':>7}")
for t, r in res.iterrows():
    print(f"{t:4} {LABEL[t]:19} {r.bh_cagr:6.2%}{r.bh_vol:8.2%}{r.bh_sharpe:8.2f}"
          f"{r.bh_dd:7.1%}   {r.r_cagr:6.2%}{r.r_vol:8.2%}{r.r_sharpe:8.2f}"
          f"{r.r_dd:7.1%}{r.exposure:7.1%}{r.trades:7.0f}")
m = res.mean()
print(f"{'mean':4} {'':19} {m.bh_cagr:6.2%}{m.bh_vol:8.2%}{m.bh_sharpe:8.2f}"
      f"{m.bh_dd:7.1%}   {m.r_cagr:6.2%}{m.r_vol:8.2%}{m.r_sharpe:8.2f}"
      f"{m.r_dd:7.1%}{m.exposure:7.1%}{m.trades:7.1f}")

print("\nAveraged across the eight funds")
print(f"  return given up a year          {m.bh_cagr - m.r_cagr:7.2%}")
print(f"  volatility removed              {m.bh_vol - m.r_vol:7.2%}")
print(f"  worst drawdown, buy and hold    {m.bh_dd:7.1%}")
print(f"  worst drawdown, crossover rule  {m.r_dd:7.1%}")
print(f"  Sharpe, buy and hold            {m.bh_sharpe:7.2f}")
print(f"  Sharpe, crossover rule          {m.r_sharpe:7.2f}")

print("\nMoving average pair, averaged over the same eight funds")
print(f"{'fast/slow':>10}{'CAGR':>9}{'Vol':>8}{'Sharpe':>8}{'MaxDD':>8}{'InMkt':>7}{'Trades':>8}")
for fast, slow in [(20, 100), (50, 150), (50, 200), (100, 300)]:
    agg = []
    for t, g in panel.items():
        _, rule, hold = run(g, fast, slow)
        agg.append([*stats(rule), hold.mean(),
                    int((hold.astype(int).diff().abs() == 1).sum())])
    a = np.array(agg).mean(axis=0)
    print(f"{fast:>4}/{slow:<5}{a[0]:9.2%}{a[1]:8.2%}{a[2]:8.2f}{a[3]:8.1%}{a[4]:7.1%}{a[5]:8.1f}")

print("\nSub-periods, averaged over the eight funds")
print(f"{'window':>14}{'BH CAGR':>10}{'Rule CAGR':>11}{'BH MaxDD':>10}{'Rule MaxDD':>12}")
for lo, hi, name in [("2005-01-01", "2009-12-31", "2005-2009"),
                     ("2010-01-01", "2024-12-31", "2010-2024"),
                     ("2015-01-01", "2024-12-31", "2015-2024")]:
    agg = []
    for t, g in panel.items():
        bh, rule, _ = run(g, start=lo, end=hi)
        agg.append([stats(bh)[0], stats(rule)[0], stats(bh)[3], stats(rule)[3]])
    a = np.array(agg).mean(axis=0)
    print(f"{name:>14}{a[0]:10.2%}{a[1]:11.2%}{a[2]:10.1%}{a[3]:12.1%}")

# Whipsaws: crosses that reverse within three months
print("\nSignal life")
short_lived, spells = 0, 0
for t, g in panel.items():
    _, _, hold = run(g)
    block = (hold != hold.shift()).cumsum()
    for _, s in hold.groupby(block):
        if s.iloc[0]:
            spells += 1
            short_lived += int(len(s) < 63)
print(f"  long spells across the eight funds        {spells}")
print(f"  spells lasting under three months         {short_lived} ({short_lived / spells:.0%})")

# ── Independent checks ────────────────────────────────────────────────
spy = panel["SPY"]
cal = spy.loc[START:END, "return_daily"].groupby(spy.loc[START:END].index.year)
print("\nChecks")
print(f"  SPY 2008 calendar total return   {(1 + cal.get_group(2008)).prod() - 1:7.2%}  (published: -36.8%)")
print(f"  SPY 2013 calendar total return   {(1 + cal.get_group(2013)).prod() - 1:7.2%}  (published: +32.3%)")
mo = xfl.prices("SPY", start=START, end=END, interval="1mo", fields=["return_daily"])
mo_cagr = (1 + mo["return_daily"].dropna()).prod() ** (12 / len(mo["return_daily"].dropna())) - 1
print(f"  SPY CAGR from daily returns      {res.loc['SPY', 'bh_cagr']:7.2%}")
print(f"  SPY CAGR from monthly returns    {mo_cagr:7.2%}")
sig = (spy["adj_close"].rolling(50).mean() > spy["adj_close"].rolling(200).mean())
flips = sig[sig != sig.shift()].loc["2020-01-01":"2020-12-31"]
print("  SPY 2020 crossovers              " +
      ", ".join(f"{d.date()} {'golden' if v else 'death'}" for d, v in flips.items()))

# ── Chart ─────────────────────────────────────────────────────────────
plt.rcParams.update({"figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
                     "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
                     "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
                     "axes.edgecolor": "#333333", "font.size": 10})
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
x = np.arange(len(res))
ax1.bar(x - 0.2, res["bh_dd"] * 100, 0.4, color="#6b7280", label="Buy and hold")
ax1.bar(x + 0.2, res["r_dd"] * 100, 0.4, color="#3b82f6", label="50/200 crossover")
ax1.set_ylabel("Worst drawdown (%)")
ax1.set_title("Golden cross versus buy and hold, eight ETFs, 2005-2024")
ax1.legend(frameon=False, loc="lower right")
ax2.bar(x - 0.2, res["bh_cagr"] * 100, 0.4, color="#6b7280")
ax2.bar(x + 0.2, res["r_cagr"] * 100, 0.4, color="#3b82f6")
ax2.set_ylabel("Return a year (%)")
ax2.axhline(0, color="#333333", lw=0.8)
ax2.set_xticks(x)
ax2.set_xticklabels(res.index)
for ax in (ax1, ax2):
    ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(IMG, dpi=150, facecolor="#0a0a0a")
