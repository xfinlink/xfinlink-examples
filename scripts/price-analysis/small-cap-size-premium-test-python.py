# Full write-up: https://xfinlink.com/blog/small-cap-size-premium-test-python
import numpy as np
import pandas as pd
import xfinlink as xfl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

FUNDS = {"SPY": "S&P 500 (large)", "MDY": "S&P 400 (mid)",
         "IJR": "S&P 600 (small)", "IWM": "Russell 2000 (small)"}
START, END = "2000-06-01", "2024-12-31"


def series(t):
    d = xfl.prices(t, start=START, end=END, fields=["close", "return_daily"]).sort_values("date")
    d["date"] = pd.to_datetime(d["date"])
    return d.dropna(subset=["return_daily"]).set_index("date")["return_daily"]


rets = pd.DataFrame({t: series(t) for t in FUNDS}).dropna()
yrs = (rets.index[-1] - rets.index[0]).days / 365.25
print(f"{len(rets)} common sessions, {rets.index.min().date()} to {rets.index.max().date()} ({yrs:.1f} years)")
print()


def stats(r):
    cagr = (1 + r).prod() ** (1 / yrs) - 1
    vol = r.std() * np.sqrt(252)
    curve = (1 + r).cumprod()
    return cagr * 100, vol * 100, cagr / vol, (curve / curve.cummax() - 1).min() * 100


print(f"{'':4s} {'index':22s} {'CAGR':>7s} {'Vol':>7s} {'Ret/Vol':>8s} {'MaxDD':>8s}")
for t, label in FUNDS.items():
    c, v, s, d = stats(rets[t])
    print(f"{t:4s} {label:22s} {c:6.2f}% {v:6.2f}% {s:8.2f} {d:7.2f}%")

# rolling five-year annualised gap, small minus large
W = 252 * 5
cum = (1 + rets).cumprod()
roll = {}
for t in ["IJR", "IWM", "MDY"]:
    a = (cum[t] / cum[t].shift(W)) ** (252 / W) - 1
    b = (cum["SPY"] / cum["SPY"].shift(W)) ** (252 / W) - 1
    roll[t] = ((a - b) * 100).dropna()
roll = pd.DataFrame(roll)
print()
for t in roll.columns:
    g = roll[t]
    print(f"{t} minus SPY, rolling 5y: positive in {(g > 0).mean()*100:5.1f}% of {len(g)} windows  "
          f"median {g.median():6.2f}pp  best {g.max():6.2f}pp  worst {g.min():7.2f}pp")

# by period
print()
for lo, hi in [(2000, 2010), (2011, 2024)]:
    sub = rets[(rets.index.year >= lo) & (rets.index.year <= hi)]
    y = (sub.index[-1] - sub.index[0]).days / 365.25
    line = []
    for t in ["SPY", "MDY", "IJR", "IWM"]:
        line.append(f"{t} {(((1+sub[t]).prod()) ** (1/y) - 1)*100:6.2f}%")
    print(f"{lo}-{hi} ({y:4.1f}y): " + "  ".join(line))

# how much of the full-period gap is one stretch
print()
last = rets[rets.index.year >= 2014]
yl = (last.index[-1] - last.index[0]).days / 365.25
print("2014-2024 only: " + "  ".join(
    f"{t} {(((1+last[t]).prod()) ** (1/yl) - 1)*100:.2f}%" for t in ["SPY", "IJR", "IWM"]))

plt.rcParams.update({"figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
                     "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
                     "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
                     "axes.edgecolor": "#3a3a3a", "font.size": 10})
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={"width_ratios": [1.2, 1]})
for t, col in [("SPY", "#3b82f6"), ("MDY", "#f59e0b"), ("IJR", "#10b981"), ("IWM", "#9ca3af")]:
    a1.plot(cum.index, cum[t].values, label=t, linewidth=1.2, color=col)
a1.set_yscale("log")
a1.set_ylabel("Growth of 1 unit (log scale)")
a1.set_title("Growth by size bucket, 2000-2024")
a1.legend(facecolor="#0a0a0a", edgecolor="#3a3a3a", labelcolor="#e0e0e0", fontsize=8)
a2.plot(roll.index, roll["IJR"], color="#10b981", linewidth=1.1, label="IJR minus SPY")
a2.plot(roll.index, roll["IWM"], color="#9ca3af", linewidth=1.1, label="IWM minus SPY")
a2.axhline(0, color="#e0e0e0", linewidth=0.9)
a2.set_ylabel("Annualised gap over 5 years (pp)")
a2.set_title("Rolling five-year size premium")
a2.legend(facecolor="#0a0a0a", edgecolor="#3a3a3a", labelcolor="#e0e0e0", fontsize=8)
plt.tight_layout()
plt.savefig("small-cap-size-premium-test-python.png", dpi=150, facecolor="#0a0a0a")
print("\nchart written")
