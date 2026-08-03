# Full write-up: https://xfinlink.com/blog/value-at-risk-backtest-kupiec-christoffersen-python
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xfinlink as xfl
from scipy import stats

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

OUT = "value-at-risk-backtest-kupiec-christoffersen-python.png"

ASSETS = {
    "SPY": "US large cap", "IWM": "US small cap", "EFA": "Developed ex-US",
    "EEM": "Emerging markets", "TLT": "20y+ Treasuries", "LQD": "Investment grade",
    "HYG": "High yield", "VNQ": "US REITs",
}
WINDOW, ALPHA = 500, 0.01
EVAL_START, END = "2016-01-01", "2026-07-31"
CRISIS = ("2020-02-20", "2020-04-30")

px = xfl.prices(list(ASSETS), start="2013-06-01", end=END, fields=["adj_close"])

print("Series check")
for t in ASSETS:
    s = px[px["ticker"] == t].sort_values("date")
    r = s["adj_close"].pct_change()
    assert s["date"].is_monotonic_increasing and s["date"].is_unique
    assert s["adj_close"].notna().all() and (s["adj_close"] > 0).all()
    print(f"  {t:4s} {len(s):5d} rows  {s['date'].min().date()} to {s['date'].max().date()}"
          f"  worst day {r.min():+.2%}  best day {r.max():+.2%}  missing values {s['adj_close'].isna().sum()}")


def _xlogy(a, b):
    return 0.0 if a == 0 else a * np.log(b)


def kupiec(hits, p):
    """Unconditional coverage: is the breach rate p? LR ~ chi2(1)."""
    n, x = len(hits), int(hits.sum())
    pi = x / n
    lr = 2 * ((_xlogy(n - x, 1 - pi) + _xlogy(x, pi))
              - (_xlogy(n - x, 1 - p) + _xlogy(x, p)))
    return lr, 1 - stats.chi2.cdf(lr, 1)


def christoffersen(hits):
    """Independence: does a breach today change the odds of one tomorrow?"""
    a, b = hits[:-1], hits[1:]
    n00 = int(((a == 0) & (b == 0)).sum()); n01 = int(((a == 0) & (b == 1)).sum())
    n10 = int(((a == 1) & (b == 0)).sum()); n11 = int(((a == 1) & (b == 1)).sum())
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    p01 = n01 / (n00 + n01) if n00 + n01 else 0.0
    p11 = n11 / (n10 + n11) if n10 + n11 else 0.0
    lr = 2 * ((_xlogy(n00, 1 - p01) + _xlogy(n01, p01)
               + _xlogy(n10, 1 - p11) + _xlogy(n11, p11))
              - (_xlogy(n00 + n10, 1 - pi) + _xlogy(n01 + n11, pi)))
    return lr, 1 - stats.chi2.cdf(lr, 1)


def var_series(ret):
    """One-day-ahead VaR from a window that ends the previous session."""
    roll = ret.shift(1).rolling(WINDOW)
    return (roll.mean() + roll.std(ddof=1) * stats.norm.ppf(ALPHA),
            roll.quantile(ALPHA))


rows, breach_dates, eval_index = [], {}, None
for t in ASSETS:
    s = px[px["ticker"] == t].sort_values("date").reset_index(drop=True)
    s["ret"] = s["adj_close"].pct_change()
    s = s.dropna(subset=["ret"]).reset_index(drop=True)
    s["gauss"], s["hist"] = var_series(s["ret"])
    ev = s[(s["date"] >= EVAL_START) & s["gauss"].notna()].reset_index(drop=True)
    eval_index = ev["date"]
    for name, col in (("Gaussian", "gauss"), ("Historical", "hist")):
        h = (ev["ret"] < ev[col]).to_numpy().astype(int)
        lr_uc, p_uc = kupiec(h, ALPHA)
        lr_ind, p_ind = christoffersen(h)
        rows.append(dict(ticker=t, model=name, n=len(h), breaches=int(h.sum()),
                         rate=h.mean(), p_uc=p_uc, p_ind=p_ind,
                         p_cc=1 - stats.chi2.cdf(lr_uc + lr_ind, 2),
                         avg_var=ev[col].mean()))
        breach_dates[(t, name)] = ev.loc[h == 1, "date"].to_numpy()

res = pd.DataFrame(rows)
assert res["n"].nunique() == 1
n_eval = int(res["n"].iloc[0])

print(f"\nOne-day {int((1 - ALPHA) * 100)}% value-at-risk backtest, daily price returns")
print(f"Rolling {WINDOW}-day estimation window, evaluated "
      f"{eval_index.min().date()} to {eval_index.max().date()}")
print(f"{n_eval} trading days per fund, {n_eval * ALPHA:.1f} breaches expected "
      f"if the model is right\n")

star = lambda p: "*" if p < 0.05 else " "
for name in ("Gaussian", "Historical"):
    sub = res[res["model"] == name]
    print(f"{name} VaR" + ("  (window mean and standard deviation, normal quantile)"
                           if name == "Gaussian" else "  (1st percentile of the window)"))
    print(f"{'':24s}{'avg VaR':>8s}{'breaches':>10s}{'rate':>8s}"
          f"{'Kupiec p':>11s}{'Indep p':>10s}{'Joint p':>10s}")
    for _, r in sub.iterrows():
        print(f"{r.ticker:5s}{ASSETS[r.ticker]:19s}{r.avg_var:8.2%}{r.breaches:10d}"
              f"{r.rate:8.2%}{r.p_uc:10.4f}{star(r.p_uc)}{r.p_ind:9.4f}{star(r.p_ind)}"
              f"{r.p_cc:9.4f}{star(r.p_cc)}")
    print(f"  mean breach rate {sub['rate'].mean():.2%}   "
          f"funds failing Kupiec at 5%: {(sub['p_uc'] < 0.05).sum()}/8   "
          f"failing independence: {(sub['p_ind'] < 0.05).sum()}/8\n")

# ---- how concentrated are the breaches in time? ------------------------
crisis_days = int(((eval_index >= CRISIS[0]) & (eval_index <= CRISIS[1])).sum())
print(f"Breach timing, historical simulation VaR "
      f"({CRISIS[0]} to {CRISIS[1]} is {crisis_days} of the {n_eval} sessions, "
      f"{crisis_days / n_eval:.1%})")
for t in ASSETS:
    d = pd.Series(breach_dates[(t, "Historical")])
    crisis = int(((d >= CRISIS[0]) & (d <= CRISIS[1])).sum())
    near = int((d.diff().dt.days <= 7).sum())
    print(f"  {t:5s}{len(d):4d} breaches{crisis:5d} in the crisis window ({crisis / len(d):5.1%})"
          f"{near:6d} within a week of the previous one ({near / len(d):5.1%})")

allb = pd.Series(np.concatenate([breach_dates[(t, "Historical")] for t in ASSETS]))
nc = int(((allb >= CRISIS[0]) & (allb <= CRISIS[1])).sum())
counts = allb.value_counts()
alldays = sorted(pd.Timestamp(d).date() for d in counts[counts == 8].index)
print(f"  all funds: {len(allb)} breaches, {nc} ({nc / len(allb):.1%}) inside that window")
print(f"  days on which all 8 funds breached at once: "
      f"{', '.join(str(d) for d in alldays) if alldays else 'none'}")

# ---- are the estimators and the tests calibrated? ----------------------
rng = np.random.default_rng(20260803)
sim = pd.Series(rng.standard_normal(60000) / 100)
sg, sh = var_series(sim)
m = sg.notna()
print("\nCalibration on data with no fat tails and no volatility clustering")
print(f"  60,000 simulated independent normal returns: Gaussian VaR breached "
      f"{(sim[m] < sg[m]).mean():.2%} of days, historical simulation {(sim[m] < sh[m]).mean():.2%}")

REPS = 2000
iid_uc = iid_ind = cl_uc = cl_ind = 0
for _ in range(REPS):
    h = (rng.random(n_eval) < ALPHA).astype(int)
    iid_uc += kupiec(h, ALPHA)[1] < 0.05
    iid_ind += christoffersen(h)[1] < 0.05
p11 = 0.35
p01 = ALPHA * (1 - p11) / (1 - ALPHA)
for _ in range(REPS // 4):
    h, s = np.zeros(n_eval, dtype=int), 0
    for i in range(n_eval):
        s = int(rng.random() < (p11 if s else p01))
        h[i] = s
    cl_uc += kupiec(h, ALPHA)[1] < 0.05
    cl_ind += christoffersen(h)[1] < 0.05
print(f"  simulated {n_eval}-day breach sequences, tests run at the 5% level:")
print(f"    genuinely independent 1% breaches: Kupiec rejects {iid_uc / REPS:.1%}, "
      f"independence rejects {iid_ind / REPS:.1%}")
print(f"    clustered breaches, same 1% rate  : Kupiec rejects {cl_uc / (REPS // 4):.1%}, "
      f"independence rejects {cl_ind / (REPS // 4):.1%}")

# ---- chart -------------------------------------------------------------
plt.rcParams.update({"figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
                     "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
                     "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
                     "axes.edgecolor": "#3a3a3a", "font.size": 10})
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))
names = list(ASSETS)
x = np.arange(len(names))
pick = lambda model: [res[(res.ticker == t) & (res.model == model)]["rate"].iloc[0] * 100
                      for t in names]
ax1.bar(x - 0.2, pick("Gaussian"), 0.4, color="#3b82f6", label="Gaussian VaR")
ax1.bar(x + 0.2, pick("Historical"), 0.4, color="#a16207", label="Historical simulation VaR")
ax1.axhline(1.0, color="#e0e0e0", ls="--", lw=1)
ax1.text(len(names) - 0.45, 1.06, "1% promised", color="#e0e0e0", fontsize=9, ha="right")
ax1.set_xticks(x); ax1.set_xticklabels(names)
ax1.set_ylabel("Days breaching the loss limit (%)")
ax1.set_title("A 99% one-day loss limit, and how often it actually broke, 2016-2026",
              color="#e0e0e0", fontsize=12)
ax1.legend(facecolor="#0a0a0a", edgecolor="#3a3a3a", labelcolor="#e0e0e0", fontsize=9)
for sp in ("top", "right"):
    ax1.spines[sp].set_visible(False)

ax2.eventplot([[pd.Timestamp(d).toordinal() for d in breach_dates[(t, "Historical")]]
               for t in names[::-1]], colors="#3b82f6", lineoffsets=np.arange(len(names)),
              linelengths=0.8, linewidths=1.2)
ax2.set_yticks(np.arange(len(names))); ax2.set_yticklabels(names[::-1], fontsize=9)
years = [pd.Timestamp(f"{y}-01-01") for y in range(2016, 2027)]
ax2.set_xticks([d.toordinal() for d in years]); ax2.set_xticklabels([d.year for d in years])
ax2.set_xlim(pd.Timestamp("2015-11-01").toordinal(), pd.Timestamp("2026-09-15").toordinal())
ax2.set_title("Every day the limit broke, historical simulation VaR", color="#e0e0e0", fontsize=12)
for sp in ("top", "right"):
    ax2.spines[sp].set_visible(False)
plt.tight_layout()
plt.savefig(OUT, dpi=150, facecolor="#0a0a0a")
print(f"\nchart -> {OUT}")
