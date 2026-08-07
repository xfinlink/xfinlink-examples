# Full write-up: https://xfinlink.com/blog/portfolio-rebalancing-premium-python
import numpy as np
import pandas as pd
import xfinlink as xfl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

SLEEVES = {"SPY": "US equity", "EFA": "Developed intl", "EEM": "Emerging mkts",
           "TLT": "Long Treasuries", "VNQ": "US real estate"}
START, END = "2005-01-01", "2024-12-31"


def daily_returns(tickers, start, end):
    frames = {}
    for t in tickers:
        d = xfl.prices(t, start=start, end=end, fields=["close", "return_daily"]).sort_values("date")
        d["date"] = pd.to_datetime(d["date"])
        frames[t] = d.set_index("date")["return_daily"]
    return pd.DataFrame(frames).dropna()


def simulate(rets, weights, rule):
    """rule: 'none', 'M'/'Q'/'A' calendar, or ('band', pct)."""
    w0 = np.array([weights[c] for c in rets.columns], dtype=float)
    w = w0.copy()
    vals, trades, turnover = [], 0, 0.0
    if isinstance(rule, tuple):
        marks = None
    elif rule == "none":
        marks = None
    else:
        marks = set(rets.index.to_series().resample(rule).last().dropna())
    total = 1.0
    for dt, row in rets.iterrows():
        w = w * (1.0 + row.values)
        total_growth = w.sum()
        vals.append(total * total_growth)
        total = total * total_growth
        w = w / total_growth
        hit = False
        if isinstance(rule, tuple) and np.abs(w - w0).max() > rule[1]:
            hit = True
        elif marks is not None and dt in marks:
            hit = True
        if hit:
            turnover += np.abs(w - w0).sum() / 2.0
            trades += 1
            w = w0.copy()
    return pd.Series(vals, index=rets.index), trades, turnover


def perf(curve, trades, turnover):
    r = curve.pct_change().dropna()
    yrs = (curve.index[-1] - curve.index[0]).days / 365.25
    cagr = curve.iloc[-1] ** (1 / yrs) - 1
    vol = r.std() * np.sqrt(252)
    dd = (curve / curve.cummax() - 1).min()
    return {"CAGR": cagr * 100, "Vol": vol * 100, "Ret/Vol": cagr / vol,
            "MaxDD": dd * 100, "Rebals": trades, "Turnover/yr": turnover / yrs * 100}


rets = daily_returns(list(SLEEVES), START, END)
print(f"{len(rets)} common sessions, {rets.index.min().date()} to {rets.index.max().date()}")
print("sleeves:", ", ".join(f"{k} ({v})" for k, v in SLEEVES.items()))
print()

eq = {c: 0.2 for c in rets.columns}
RULES = [("Never (drift)", "none"), ("Monthly", "ME"), ("Quarterly", "QE"),
         ("Annually", "YE"), ("5% band", ("band", 0.05)), ("10% band", ("band", 0.10))]

rows, curves = [], {}
for name, rule in RULES:
    c, t, to = simulate(rets, eq, rule)
    curves[name] = c
    rows.append({"rule": name, **perf(c, t, to)})
tab = pd.DataFrame(rows)
print("Equal-weight five-sleeve portfolio")
print(f"{'rule':14s} {'CAGR':>7s} {'Vol':>7s} {'Ret/Vol':>8s} {'MaxDD':>8s} {'Rebals':>7s} {'Turn/yr':>8s}")
for r in tab.itertuples():
    print(f"{r.rule:14s} {r.CAGR:6.2f}% {r.Vol:6.2f}% {r._4:8.2f} {r.MaxDD:7.2f}% {r.Rebals:7d} {r._7:7.1f}%")

# 60/40 two-asset version
rets2 = rets[["SPY", "TLT"]]
w6040 = {"SPY": 0.6, "TLT": 0.4}
print("\n60/40 SPY/TLT")
print(f"{'rule':14s} {'CAGR':>7s} {'Vol':>7s} {'Ret/Vol':>8s} {'MaxDD':>8s} {'Rebals':>7s} {'Turn/yr':>8s}")
rows2 = []
for name, rule in RULES:
    c, t, to = simulate(rets2, w6040, rule)
    p = perf(c, t, to)
    rows2.append({"rule": name, **p})
    print(f"{name:14s} {p['CAGR']:6.2f}% {p['Vol']:6.2f}% {p['Ret/Vol']:8.2f} {p['MaxDD']:7.2f}% "
          f"{p['Rebals']:7d} {p['Turnover/yr']:7.1f}%")

# where did the drift portfolio end up?
w = np.array([0.2] * len(rets.columns))
for _, row in rets.iterrows():
    w = w * (1 + row.values)
w = w / w.sum()
print("\nDrift portfolio final weights:",
      ", ".join(f"{c} {x*100:.1f}%" for c, x in zip(rets.columns, w)))
print("Sleeve total returns:",
      ", ".join(f"{c} {((1+rets[c]).prod()-1)*100:.0f}%" for c in rets.columns))


# sub-period stability
print()
for lo, hi in [(2005, 2014), (2015, 2024)]:
    sub = rets[(rets.index.year >= lo) & (rets.index.year <= hi)]
    line = []
    for name in ["Never (drift)", "Annually", "10% band"]:
        rule = dict(RULES)[name]
        c, t, to = simulate(sub, eq, rule)
        line.append(f"{name} {perf(c, t, to)['CAGR']:.2f}%")
    print(f"Five sleeves {lo}-{hi}: " + "  ".join(line))

# cost sensitivity for the monthly rule
mo = tab[tab["rule"] == "Monthly"].iloc[0]
never = tab[tab["rule"] == "Never (drift)"].iloc[0]
for bps in [1, 5, 10]:
    drag = mo["Turnover/yr"] / 100 * 2 * bps / 10000 * 100
    print(f"Monthly rule net of {bps} bps per unit traded: {mo['CAGR'] - drag:.2f}% "
          f"vs {never['CAGR']:.2f}% for never rebalancing")

plt.rcParams.update({"figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
                     "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
                     "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
                     "axes.edgecolor": "#3a3a3a", "font.size": 10})
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={"width_ratios": [1.4, 1]})
for name in ["Never (drift)", "Monthly", "Annually"]:
    a1.plot(curves[name].index, curves[name].values,
            label=name, linewidth=1.3,
            color={"Never (drift)": "#6b7280", "Monthly": "#3b82f6", "Annually": "#f59e0b"}[name])
a1.set_ylabel("Growth of 1 unit")
a1.set_title("Equal-weight portfolio, 2005-2024")
a1.legend(facecolor="#0a0a0a", edgecolor="#3a3a3a", labelcolor="#e0e0e0")
a2.scatter(tab["Vol"], tab["CAGR"], color="#3b82f6", s=45)
for r in tab.itertuples():
    a2.annotate(r.rule, (r.Vol, r.CAGR), fontsize=7.5, xytext=(4, 3),
                textcoords="offset points", color="#e0e0e0")
a2.set_xlabel("Annualised volatility (%)")
a2.set_ylabel("Annual return (%)")
a2.set_title("Return against risk by rule")
plt.tight_layout()
plt.savefig("portfolio-rebalancing-premium-python.png", dpi=150, facecolor="#0a0a0a")
print("\nchart written")
