# Full write-up: https://xfinlink.com/blog/calendar-anomaly-multiple-testing-bootstrap-python
#
# A calendar anomaly is normally reported one rule at a time, with a t-statistic
# from a single test. But the rule was picked out of a family of candidates, and
# the best of many draws is large even when every draw is noise. This searches
# all 60 month-by-weekday trading rules on three index funds, records the winning
# t-statistic, then rebuilds the distribution of that winning t-statistic under
# the null with a circular block bootstrap. The gap between the two is the size
# of the data-snooping bias. A power check re-runs the machinery on a series with
# a known effect added, to confirm the test can still find one.

import numpy as np
import pandas as pd
import xfinlink as xfl
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

FUNDS = {"SPY": "S&P 500", "QQQ": "Nasdaq 100", "IWM": "Russell 2000"}
START, END = "1996-01-01", "2024-12-31"
DRAWS, BLOCK, SEED = 2000, 10, 7
PNG = "calendar-anomaly-multiple-testing-bootstrap-python.png"

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]

# ------------------------------------------------------------------ rules ---


def rule_matrix(dates):
    """One row per (month, weekday) rule; True where that rule is invested."""
    month, weekday = dates.month.values, dates.dayofweek.values
    rows, names = [], []
    for m in range(1, 13):
        for w in range(5):
            rows.append((month == m) & (weekday == w))
            names.append(f"{MONTHS[m - 1]}-{DAYS[w]}")
    return np.array(rows, dtype=float), names


def welch_t(r, M):
    """Welch t of rule days against every other day, for all rules at once."""
    n = len(r)
    n_in = M.sum(1)
    n_out = n - n_in
    s1, s2 = M @ r, M @ (r ** 2)
    t1, t2 = r.sum(), (r ** 2).sum()
    m_in, m_out = s1 / n_in, (t1 - s1) / n_out
    v_in = (s2 - n_in * m_in ** 2) / (n_in - 1)
    v_out = ((t2 - s2) - n_out * m_out ** 2) / (n_out - 1)
    return (m_in - m_out) / np.sqrt(v_in / n_in + v_out / n_out)


def bootstrap_max_t(r, M, rng):
    """Distribution of the best t across the family when no rule is real.

    Circular blocks keep volatility clustering and autocorrelation intact
    while breaking the alignment between returns and the calendar.
    """
    n = len(r)
    n_blocks = int(np.ceil(n / BLOCK))
    out = np.empty(DRAWS)
    for b in range(DRAWS):
        starts = rng.integers(0, n, n_blocks)
        idx = (starts[:, None] + np.arange(BLOCK)[None, :]).ravel()[:n] % n
        out[b] = welch_t(r[idx], M).max()
    return out


def search(r, M, names, rng):
    t = welch_t(r, M)
    best = int(np.argmax(t))
    boot = bootstrap_max_t(r, M, rng)
    return {
        "t": t, "best": best, "name": names[best],
        "days": int(M[best].sum()), "mean": float(r[M[best] == 1].mean()),
        "t_best": float(t[best]),
        "p_naive": float(2 * (1 - stats.norm.cdf(abs(t[best])))),
        "p_snoop": float((boot >= t[best]).mean()),
        "crit95": float(np.percentile(boot, 95)),
        "boot": boot,
    }


# ------------------------------------------------------------------- run ----

rng = np.random.default_rng(SEED)
results = {}

print(f"Best-of-60 calendar rules | month x weekday | {START} to {END}")
print(f"Bootstrap: {DRAWS:,} circular block draws, block length {BLOCK}\n")

for fund, label in FUNDS.items():
    px = xfl.prices(fund, start=START, end=END, fields=["return_daily"])
    px = px.dropna(subset=["return_daily"]).sort_values("date").reset_index(drop=True)
    dates = pd.DatetimeIndex(px["date"])
    r = px["return_daily"].to_numpy()
    M, names = rule_matrix(dates)
    res = search(r, M, names, rng)
    res.update(label=label, first=dates.min().date(), last=dates.max().date(),
               n=len(r), cell=int(M.sum(1).min()))
    results[fund] = res

    print(f"{fund} ({label}) {res['first']} to {res['last']}  "
          f"{res['n']:,} trading days, smallest rule cell {res['cell']}")
    print(f"  best rule           {res['name']}  ({res['days']} days, "
          f"mean {100 * res['mean']:+.3f}% per day)")
    print(f"  t-statistic         {res['t_best']:.2f}")
    print(f"  p-value, one test   {res['p_naive']:.4f}")
    print(f"  p-value, best of 60 {res['p_snoop']:.4f}")
    print(f"  5% hurdle for t     {res['crit95']:.2f}  (single-test hurdle 1.96)")
    print(f"  rules with |t|>1.96 {int((np.abs(res['t']) > 1.96).sum())}  "
          f"(3.0 expected by chance)\n")

# Power check: plant a known effect on one cell of the SPY series and re-search.
spy = xfl.prices("SPY", start=START, end=END, fields=["return_daily"])
spy = spy.dropna(subset=["return_daily"]).sort_values("date").reset_index(drop=True)
r_spy = spy["return_daily"].to_numpy()
M, names = rule_matrix(pd.DatetimeIndex(spy["date"]))
planted = M[names.index("Sep-Tue")] == 1

print("Power check: a known effect added to every September Tuesday in SPY")
for size in (0.0025, 0.0050):
    r_plant = r_spy + planted * size
    res = search(r_plant, M, names, rng)
    print(f"  +{100 * size:.2f}% per day planted -> winner {res['name']}, "
          f"t {res['t_best']:.2f}, p-value best of 60 {res['p_snoop']:.4f}")

# ----------------------------------------------------------------- chart ----

spy_res = results["SPY"]
plt.rcParams.update({"figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
                     "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
                     "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
                     "axes.edgecolor": "#3a3a3a", "font.size": 10})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

order = np.argsort(spy_res["t"])[::-1]
ax1.bar(range(60), spy_res["t"][order], color="#3b82f6", width=0.85)
ax1.axhline(1.96, color="#e0e0e0", lw=1, ls="--")
ax1.axhline(spy_res["crit95"], color="#f59e0b", lw=1, ls="--")
ax1.set_ylim(spy_res["t"].min() - 0.2, spy_res["crit95"] + 0.6)
ax1.text(59, 2.08, "single-test 5% hurdle", ha="right", fontsize=8, color="#e0e0e0")
ax1.text(59, spy_res["crit95"] + 0.12, "best-of-60 5% hurdle", ha="right",
         fontsize=8, color="#f59e0b")
ax1.set_title("The 60 calendar rules, ranked", fontsize=11)
ax1.set_xlabel("Rule, best to worst")
ax1.set_ylabel("t-statistic against all other days")

counts, _, _ = ax2.hist(spy_res["boot"], bins=45, color="#3b82f6", alpha=0.85)
ax2.set_ylim(0, counts.max() * 1.22)
ax2.axvline(spy_res["t_best"], color="#f59e0b", lw=1.6)
ax2.text(spy_res["t_best"], counts.max() * 1.16,
         f" best real rule, t = {spy_res['t_best']:.2f}", fontsize=8,
         color="#f59e0b", va="top")
ax2.set_title("Best t-statistic when no rule is real", fontsize=11)
ax2.set_xlabel("Winning t-statistic in a bootstrap draw")
ax2.set_ylabel("Number of draws")

fig.suptitle("Calendar anomalies against a multiple-testing hurdle, SPY 1996-2024",
             fontsize=12.5)
plt.tight_layout()
plt.savefig(PNG, dpi=150, facecolor="#0a0a0a")
print(f"\nChart saved to {PNG}")
