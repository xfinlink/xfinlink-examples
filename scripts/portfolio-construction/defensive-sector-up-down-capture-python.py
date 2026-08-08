# Full write-up: https://xfinlink.com/blog/defensive-sector-up-down-capture-python
import numpy as np
import pandas as pd
import xfinlink as xfl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

SECTORS = {"XLP": "Consumer Staples", "XLU": "Utilities", "XLV": "Health Care",
           "XLE": "Energy", "XLF": "Financials", "XLI": "Industrials",
           "XLK": "Technology", "XLY": "Cons. Discretionary", "XLB": "Materials"}
START, END = "1999-01-01", "2026-06-30"

px = xfl.prices(["SPY"] + list(SECTORS), start=START, end=END,
                fields=["close", "return_daily"], max_rows=200000)
px["date"] = pd.to_datetime(px["date"])
daily = px.pivot_table(index="date", columns="ticker", values="return_daily").dropna()
close = px.pivot_table(index="date", columns="ticker", values="close").reindex(daily.index)

monthly = (1 + daily).resample("ME").prod() - 1
print(f"{len(monthly)} common months, {monthly.index.min():%Y-%m} to {monthly.index.max():%Y-%m}")
print(f"{len(daily)} daily sessions")
print()

up = monthly["SPY"] > 0
dn = monthly["SPY"] < 0
print(f"up months: {up.sum()}   down months: {dn.sum()}")
print()

# Capture ratios: average sector return in up/down months, scaled by the market's own average.
rows = []
for t, name in SECTORS.items():
    r = monthly[t]
    uc = r[up].mean() / monthly["SPY"][up].mean() * 100
    dc = r[dn].mean() / monthly["SPY"][dn].mean() * 100
    # dual beta: slope against the market, fitted separately in up and down months
    bu = np.polyfit(monthly["SPY"][up], r[up], 1)[0]
    bd = np.polyfit(monthly["SPY"][dn], r[dn], 1)[0]
    yrs = (monthly.index[-1] - monthly.index[0]).days / 365.25
    cagr = ((1 + r).prod() ** (1 / yrs) - 1) * 100
    rows.append((t, name, uc, dc, uc - dc, bu, bd, cagr))

tab = pd.DataFrame(rows, columns=["ticker", "sector", "up_cap", "down_cap",
                                  "spread", "beta_up", "beta_dn", "cagr"])
tab = tab.sort_values("down_cap")

print(f"{'':5s} {'sector':21s} {'Up cap':>7s} {'Dn cap':>7s} {'Spread':>7s} "
      f"{'B up':>6s} {'B dn':>6s} {'CAGR':>7s}")
for _, x in tab.iterrows():
    print(f"{x.ticker:5s} {x.sector:21s} {x.up_cap:6.1f}% {x.down_cap:6.1f}% "
          f"{x.spread:6.1f}pp {x.beta_up:6.2f} {x.beta_dn:6.2f} {x.cagr:6.2f}%")
print()
spy_yrs = (monthly.index[-1] - monthly.index[0]).days / 365.25
print(f"SPY   {'S&P 500':21s} {100.0:6.1f}% {100.0:6.1f}% {0.0:6.1f}pp "
      f"{1.00:6.2f} {1.00:6.2f} {((1 + monthly['SPY']).prod() ** (1 / spy_yrs) - 1) * 100:6.2f}%")
print()

# Participation in the market's four deepest daily drawdowns.
curve = (1 + daily["SPY"]).cumprod()
dd = curve / curve.cummax() - 1
episodes = []
in_dd = False
for d, v in dd.items():
    if v < 0 and not in_dd:
        in_dd, start = True, d
    elif v == 0 and in_dd:
        in_dd = False
        seg = dd.loc[start:d]
        episodes.append((start, seg.idxmin(), seg.min()))
if in_dd:
    seg = dd.loc[start:]
    episodes.append((start, seg.idxmin(), seg.min()))
episodes = sorted(episodes, key=lambda e: e[2])[:4]
episodes = sorted(episodes, key=lambda e: e[0])

print("Sector return over the market's four deepest peak-to-trough declines")
hdr = "      " + "".join(f"{p.year:>8d}" for p, _, _ in episodes)
print(hdr)
print("      " + "".join(f"{d * 100:7.1f}%" for _, _, d in episodes) + "   <- SPY")
for t in tab["ticker"]:
    cells = ""
    for p, tr, _ in episodes:
        seg = daily[t].loc[p:tr]
        cells += f"{((1 + seg).prod() - 1) * 100:7.1f}%"
    print(f"{t:5s} {cells}")
print()

corr = np.corrcoef(tab["up_cap"], tab["down_cap"])[0, 1]
print(f"cross-sector correlation between up capture and down capture: {corr:.3f}")
best = tab.iloc[0]
print(f"lowest down capture: {best.ticker} at {best.down_cap:.1f}%, "
      f"giving up {100 - best.up_cap:.1f}pp of upside")

# Chart -------------------------------------------------------------------
plt.rcParams.update({"figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
                     "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
                     "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
                     "axes.edgecolor": "#3f3f3f"})
fig, ax = plt.subplots(figsize=(10, 7))
lo = min(tab["down_cap"].min(), tab["up_cap"].min()) - 12
hi = max(tab["down_cap"].max(), tab["up_cap"].max()) + 12
lim = [lo, hi]
ax.plot(lim, lim, color="#6b7280", lw=1, ls="--", zorder=1)
ax.axhline(100, color="#3f3f3f", lw=0.8, zorder=1)
ax.axvline(100, color="#3f3f3f", lw=0.8, zorder=1)
ax.scatter(tab["down_cap"], tab["up_cap"], s=90, color="#3b82f6", zorder=3)
NUDGE = {"XLU": (0, -19, "center"), "XLI": (-9, -4, "right"),
         "XLE": (-9, -4, "right")}
for _, x in tab.iterrows():
    dx, dy, ha = NUDGE.get(x.ticker, (9, -4, "left"))
    ax.annotate(f"{x.ticker}  {x.sector}", (x.down_cap, x.up_cap),
                textcoords="offset points", xytext=(dx, dy), ha=ha,
                fontsize=9, color="#e0e0e0")
ax.scatter([100], [100], s=110, marker="s", color="#e0e0e0", zorder=3)
ax.annotate("SPY", (100, 100), textcoords="offset points", xytext=(9, -4),
            fontsize=9, color="#e0e0e0")
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_xlabel("Share of the market's average down month captured (%)")
ax.set_ylabel("Share of the market's average up month captured (%)")
ax.set_title("Defensive sectors give up upside in proportion to the downside they avoid\n"
             "S&P 500 sector SPDRs, monthly returns 1999-2026",
             color="#e0e0e0", fontsize=12)
ax.text(0.985, 0.03, "points above the dashed line captured more up than down",
        transform=ax.transAxes, ha="right", fontsize=8.5, color="#9ca3af")
plt.tight_layout()
plt.savefig("defensive-sector-up-down-capture-python.png", dpi=150,
            facecolor="#0a0a0a")
print("\nchart written")
