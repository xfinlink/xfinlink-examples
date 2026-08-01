# Full write-up: https://xfinlink.com/blog/stock-bond-correlation-60-40-portfolio-risk-python
"""Stock-bond correlation through time and what it costs a 60/40 portfolio.

The break date is estimated, not assumed: every interior day is tried as a
split point and the one maximising the Fisher-z difference between the two
sub-sample correlations wins. Portfolio risk is then rebuilt from the
two-asset variance formula under each regime's correlation.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

START = "2002-07-26"          # first common trading day of the bond funds
TICKERS = ["SPY", "TLT", "IEF", "LQD", "TIP"]
LABELS = {"TLT": "Long Treasuries (TLT)", "IEF": "7-10y Treasuries (IEF)",
          "LQD": "Investment grade credit (LQD)"}
WIN = 126                     # six months of trading days
ANN = np.sqrt(252)

# ---------------------------------------------------------------- data
px = pd.concat([xfl.prices(t, start=START, fields=["close", "return_daily"])
                for t in TICKERS])
assert not px.duplicated(subset=["ticker", "date"]).any()
assert (px["close"] > 0).all()
assert px["return_daily"].abs().max() < 1.0

wide = px.pivot_table(index="date", columns="ticker", values="return_daily")
d = wide[["SPY", "TLT", "IEF", "LQD"]].dropna()      # common calendar

# ------------------------------------------------- break-date search
def fisher(r):
    return 0.5 * np.log((1 + r) / (1 - r))


def break_date(x, y, trim=0.15):
    n = len(x)
    out = []
    for i in range(int(n * trim), int(n * (1 - trim))):
        r1 = np.corrcoef(x[:i], y[:i])[0, 1]
        r2 = np.corrcoef(x[i:], y[i:])[0, 1]
        z = (fisher(r1) - fisher(r2)) / np.sqrt(1 / (i - 3) + 1 / (n - i - 3))
        out.append((i, abs(z)))
    return pd.DataFrame(out, columns=["i", "absz"])


prof = break_date(d["SPY"].values, d["TLT"].values)
split = int(prof.loc[prof["absz"].idxmax(), "i"])
BREAK = d.index[split]
peak = prof["absz"].max()
flat = d.index[prof.loc[prof["absz"] > peak - 1.0, "i"].values]

pre, post = d[d.index < BREAK], d[d.index >= BREAK]

# ------------------------------------------------ portfolio arithmetic
def port_vol(se, sb, rho, we=0.60):
    wb = 1 - we
    return np.sqrt(we ** 2 * se ** 2 + wb ** 2 * sb ** 2
                   + 2 * we * wb * rho * se * sb)


# ------------------------------- inflation environment (breakeven proxy)
tip = wide[["SPY", "TLT", "IEF", "TIP"]].dropna()
be = ((1 + tip["TIP"]).rolling(252).apply(np.prod, raw=True)
      - (1 + tip["IEF"]).rolling(252).apply(np.prod, raw=True))
cond = tip.loc[be.dropna().index].assign(be=be.dropna())
cond["q"] = pd.qcut(cond["be"], 4, labels=["Q1 lowest", "Q2", "Q3", "Q4 highest"])
cond_pre = cond[cond.index < BREAK].copy()
cond_pre["q"] = pd.qcut(cond_pre["be"], 4, labels=["Q1 lowest", "Q2", "Q3", "Q4 highest"])

# ---------------------------------------------------------- output
print(f"Sample: {d.index[0]:%Y-%m-%d} to {d.index[-1]:%Y-%m-%d}  ({len(d)} trading days)")
print(f"Estimated break in the SPY/TLT correlation: {BREAK:%Y-%m-%d}   |z| = {peak:.1f}")
print(f"Split points within 1.0 of the peak: {flat[0]:%Y-%m-%d} to {flat[-1]:%Y-%m-%d}")
print(f"Pre-break {len(pre)} days, post-break {len(post)} days\n")

print(f"{'':32}{'correlation with SPY':>22}")
print(f"{'':32}{'pre':>10}{'post':>10}{'change':>10}")
for b in ["TLT", "IEF", "LQD"]:
    r1, r2 = pre["SPY"].corr(pre[b]), post["SPY"].corr(post[b])
    print(f"{LABELS[b]:<32}{r1:>+10.3f}{r2:>+10.3f}{r2 - r1:>+10.3f}")

roll = d["SPY"].rolling(WIN).corr(d["TLT"])
print(f"\nShare of {WIN}-day windows with a positive SPY/TLT correlation:")
print(f"  pre-break  {(roll[d.index < BREAK] > 0).mean():.1%}")
print(f"  post-break {(roll[d.index >= BREAK] > 0).mean():.1%}")

print(f"\n{'':32}{'annualised volatility':>22}")
print(f"{'':32}{'pre':>10}{'post':>10}")
for t in ["SPY", "TLT", "IEF"]:
    print(f"{t:<32}{pre[t].std() * ANN * 100:>9.2f}%{post[t].std() * ANN * 100:>9.2f}%")

for b in ["TLT", "IEF"]:
    se_a, sb_a = pre["SPY"].std() * ANN, pre[b].std() * ANN
    se_b, sb_b = post["SPY"].std() * ANN, post[b].std() * ANN
    r_a, r_b = pre["SPY"].corr(pre[b]), post["SPY"].corr(post[b])
    v_a, v_b = port_vol(se_a, sb_a, r_a), port_vol(se_b, sb_b, r_b)
    v_cf = port_vol(se_b, sb_b, r_a)
    print(f"\n60/40 SPY/{b}, annualised volatility")
    print(f"  pre-break  : {v_a * 100:5.2f}%   (realised {(0.6 * pre['SPY'] + 0.4 * pre[b]).std() * ANN * 100:.2f}%)")
    print(f"  post-break : {v_b * 100:5.2f}%   (realised {(0.6 * post['SPY'] + 0.4 * post[b]).std() * ANN * 100:.2f}%)")
    print(f"  post-break volatilities at the old correlation of {r_a:+.3f}: {v_cf * 100:5.2f}%")
    print(f"  risk added by the correlation change alone: {(v_b - v_cf) * 100:+5.2f} pp")
    print(f"  diversification saving vs a weighted average of the two legs:"
          f"  pre {(0.6 * se_a + 0.4 * sb_a - v_a) * 100:.2f} pp,"
          f"  post {(0.6 * se_b + 0.4 * sb_b - v_b) * 100:.2f} pp")

print(f"\nSPY/TLT correlation by trailing 1-year breakeven inflation proxy (TIP minus IEF)")
print(f"  {'bucket':<12}{'n':>6}{'corr':>9}{'share after break':>20}")
for q, g in cond.groupby("q", observed=True):
    print(f"  {q:<12}{len(g):>6}{g['SPY'].corr(g['TLT']):>+9.3f}"
          f"{np.mean(g.index >= BREAK):>19.0%}")
print(f"  same buckets, pre-break sample only ({len(cond_pre)} days):")
for q, g in cond_pre.groupby("q", observed=True):
    print(f"  {q:<12}{len(g):>6}{g['SPY'].corr(g['TLT']):>+9.3f}")

# ----------------------------------------------------------- chart
plt.rcParams.update({
    "figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
    "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
    "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
    "axes.edgecolor": "#333333", "font.size": 10,
})
roll_ief = d["SPY"].rolling(WIN).corr(d["IEF"])
se_r = d["SPY"].rolling(WIN).std() * ANN
sb_r = d["TLT"].rolling(WIN).std() * ANN
rho_pre = pre["SPY"].corr(pre["TLT"])
actual = port_vol(se_r, sb_r, roll) * 100
counter = port_vol(se_r, sb_r, rho_pre) * 100

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
ax1.axhline(0, color="#666666", lw=0.9)
ax1.plot(roll.index, roll, color="#3b82f6", lw=1.3, label="S&P 500 vs long Treasuries")
ax1.plot(roll_ief.index, roll_ief, color="#9ca3af", lw=1.0, alpha=0.8,
         label="S&P 500 vs 7-10 year Treasuries")
ax1.axvline(BREAK, color="#f59e0b", lw=1.0, ls="--")
ax1.annotate(f"estimated break {BREAK:%b %Y}", xy=(BREAK, -0.72),
             xytext=(6, 0), textcoords="offset points", color="#f59e0b", fontsize=9)
ax1.set_ylabel("Correlation of daily returns\n(six-month window)")
ax1.set_title("Stock-bond correlation and the cost to a 60/40 portfolio")
ax1.legend(loc="upper left", frameon=False, fontsize=9)

ax2.plot(actual.index, actual, color="#3b82f6", lw=1.3, label="Realised 60/40 risk")
ax2.plot(counter.index, counter, color="#9ca3af", lw=1.0, ls="--",
         label=f"Same volatilities at the pre-break correlation of {rho_pre:+.2f}")
ax2.axvline(BREAK, color="#f59e0b", lw=1.0, ls="--")
ax2.set_ylabel("Annualised volatility of a\n60/40 stock-bond book (%)")
ax2.legend(loc="upper right", frameon=False, fontsize=9)
plt.tight_layout()
plt.savefig("stock-bond-correlation-60-40-portfolio-risk-python.png", dpi=150)
