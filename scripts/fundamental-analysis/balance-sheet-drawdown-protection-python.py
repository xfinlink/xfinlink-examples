# Full write-up: https://xfinlink.com/blog/balance-sheet-drawdown-protection-python
#
# Does a strong balance sheet cushion drawdowns?
# Point-in-time S&P 500 members, ranked on leverage known BEFORE each window opens,
# tested across three separate selloffs, with a sector-neutral control.
# Built from SEC EDGAR public filings and market data.

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xfinlink as xfl
from scipy import stats

warnings.filterwarnings("ignore")
xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

EPISODES = [
    ("2018 Q4 selloff", "2018-09-20", "2018-12-24"),
    ("COVID crash", "2020-02-19", "2020-03-23"),
    ("2022 bear market", "2022-01-03", "2022-10-12"),
]
# Leverage IS the business model for banks and REITs, not a sign of distress.
EXCLUDE = {"Financials", "Real Estate"}
QL = ["Q1 (least levered)", "Q2", "Q3", "Q4", "Q5 (most levered)"]

rows, diag = [], []

for name, start, end in EPISODES:
    # Rank only on fiscal periods that ended >=120 days before the window opens.
    # 120 days clears the 60-day 10-K and 40-day 10-Q deadlines, so every figure
    # used was genuinely public on the day the window starts. No lookahead.
    cutoff = (pd.Timestamp(start) - pd.Timedelta(days=120)).strftime("%Y-%m-%d")

    # Point-in-time membership: the index as it stood on the day, not today's list.
    universe = sorted(xfl.index("sp500", as_of=start)["ticker"].unique().tolist())

    f = xfl.fundamentals(
        universe,
        period_type="quarterly",
        start=(pd.Timestamp(cutoff) - pd.Timedelta(days=500)).strftime("%Y-%m-%d"),
        end=cutoff,
        fields=["ebitda", "total_debt", "cash_and_equivalents"],
    )
    f["period_end"] = pd.to_datetime(f["period_end"])
    f = f.dropna(subset=["ebitda", "total_debt", "cash_and_equivalents"])
    f = f.sort_values(["ticker", "period_end"])

    lev = []
    for tk, g in f.groupby("ticker"):
        g = g.drop_duplicates("period_end", keep="last").tail(4)
        if len(g) < 4:
            continue
        ttm_ebitda = g["ebitda"].sum()
        last = g.iloc[-1]
        net_debt = last["total_debt"] - last["cash_and_equivalents"]
        lev.append(
            {
                "ticker": tk,
                "gics_sector": last["gics_sector"],
                "ttm_ebitda": ttm_ebitda,
                "net_debt_to_ebitda": net_debt / ttm_ebitda,
            }
        )
    lev = pd.DataFrame(lev)
    lev = lev[~lev["gics_sector"].isin(EXCLUDE)]
    n_nonfin = len(lev)
    # The ratio is meaningless in sign when EBITDA is negative. Screen, do not winsorise.
    lev = lev[lev["ttm_ebitda"] > 0]
    n_pos = len(lev)

    # Max drawdown from return_daily. Never from raw close: it steps across splits.
    p = xfl.prices(universe, start=start, end=end,
                   fields=["close", "return_daily"], max_rows=500000)
    # prices() silently caps at max_rows. Fail loudly rather than analyse a truncated panel.
    cov = p["ticker"].nunique() / len(universe)
    assert cov > 0.95, f"{name}: price coverage {cov:.1%} — truncated, raise max_rows"

    p["date"] = pd.to_datetime(p["date"])
    p = p.sort_values(["ticker", "date"])
    ndays = p.groupby("ticker")["date"].count()
    min_days = int(ndays.median() * 0.5)

    dd = []
    for tk, g in p.groupby("ticker"):
        r = g["return_daily"].dropna()
        if len(r) < min_days:
            continue
        cum = (1 + r).cumprod()
        dd.append({"ticker": tk, "max_dd": float((cum / cum.cummax() - 1).min())})
    dd = pd.DataFrame(dd)

    m = lev.merge(dd, on="ticker", how="inner")
    m["episode"] = name
    m["quintile"] = pd.qcut(m["net_debt_to_ebitda"], 5, labels=QL)
    rows.append(m)
    diag.append({"episode": name, "window": f"{start} to {end}", "ranked_on_data_to": cutoff,
                 "universe": len(universe), "ex_fin_re": n_nonfin,
                 "ebitda_pos": n_pos, "final_n": len(m)})

panel = pd.concat(rows, ignore_index=True)
assert panel.notna().all().all(), "NaN in panel"

print("=" * 92)
print("SAMPLE — point-in-time S&P 500 members, ex-Financials/Real Estate, TTM EBITDA > 0")
print("=" * 92)
print(pd.DataFrame(diag).to_string(index=False))

print("\n" + "=" * 92)
print("MAX DRAWDOWN BY NET DEBT / EBITDA QUINTILE  (leverage known before the window opens)")
print("=" * 92)
summary = {}
for name, _, _ in EPISODES:
    e = panel[panel["episode"] == name]
    t = e.groupby("quintile", observed=True).agg(
        n=("max_dd", "size"), lev=("net_debt_to_ebitda", "median"), dd=("max_dd", "mean"))
    print(f"\n{name}  (n={len(e)})")
    for q, r in t.iterrows():
        print(f"   {q:<20s} n={int(r['n']):3d}   median net debt/EBITDA {r['lev']:6.2f}x"
              f"   mean max drawdown {r['dd'] * 100:7.2f}%")
    q1 = e[e["quintile"] == QL[0]]["max_dd"]
    q5 = e[e["quintile"] == QL[4]]["max_dd"]
    ts, tp = stats.ttest_ind(q5, q1, equal_var=False)
    rho, rp = stats.spearmanr(e["net_debt_to_ebitda"], e["max_dd"])
    print(f"   Q5 minus Q1: {(q5.mean() - q1.mean()) * 100:+.2f} pp   Welch t={ts:+.2f}  p={tp:.4f}")
    print(f"   Spearman rho(leverage, max drawdown) = {rho:+.3f}  p={rp:.4f}")
    summary[name] = t["dd"].values * 100

print("\n" + "=" * 92)
print("SECTOR MIX OF THE EXTREME QUINTILES")
print("=" * 92)
for name, _, _ in EPISODES:
    e = panel[panel["episode"] == name]
    print(f"\n{name}")
    for q in [QL[0], QL[4]]:
        top = e[e["quintile"] == q]["gics_sector"].value_counts().head(3)
        print(f"   {q:<20s} " + ", ".join(f"{s} {v}" for s, v in top.items()))

print("\n" + "=" * 92)
print("SECTOR-NEUTRAL TEST — rank leverage and drawdown WITHIN sector, then correlate")
print("=" * 92)
sn = {}
for name, _, _ in EPISODES:
    e = panel[panel["episode"] == name].copy()
    e = e[e.groupby("gics_sector")["ticker"].transform("size") >= 5]
    e["lev_r"] = e.groupby("gics_sector")["net_debt_to_ebitda"].rank(pct=True)
    e["dd_r"] = e.groupby("gics_sector")["max_dd"].rank(pct=True)
    raw_rho, raw_p = stats.spearmanr(panel[panel["episode"] == name]["net_debt_to_ebitda"],
                                     panel[panel["episode"] == name]["max_dd"])
    sn_rho, sn_p = stats.spearmanr(e["lev_r"], e["dd_r"])
    print(f"{name:18s} n={len(e):3d}  raw rho={raw_rho:+.3f} (p={raw_p:.4f})"
          f"   sector-neutral rho={sn_rho:+.3f} (p={sn_p:.4f})")
    sn[name] = (raw_rho, sn_rho)

# ---------------------------------------------------------------- chart
plt.rcParams.update({"figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
                     "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
                     "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0"})
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))
cols = ["#6b7280", "#3b82f6", "#ef4444"]
x = np.arange(5)
w = 0.26
for i, (name, _, _) in enumerate(EPISODES):
    ax1.bar(x + (i - 1) * w, summary[name], w, label=name, color=cols[i])
ax1.set_xticks(x)
ax1.set_xticklabels(["Q1\nleast levered", "Q2", "Q3", "Q4", "Q5\nmost levered"])
ax1.set_ylabel("Mean maximum drawdown (%)")
ax1.set_title("Max drawdown by balance-sheet leverage quintile, S&P 500")
ax1.axhline(0, color="#e0e0e0", lw=0.8)
ax1.set_ylim(-52, 14)
ax1.legend(facecolor="#0a0a0a", edgecolor="#333", labelcolor="#e0e0e0",
           fontsize=8, loc="upper right", ncol=3)

names = [n for n, _, _ in EPISODES]
xr = np.arange(3)
ax2.bar(xr - 0.18, [sn[n][0] for n in names], 0.36, label="Raw", color="#3b82f6")
ax2.bar(xr + 0.18, [sn[n][1] for n in names], 0.36, label="Sector-neutral", color="#6b7280")
ax2.set_xticks(xr)
ax2.set_xticklabels(names)
ax2.set_ylabel("Correlation of leverage\nwith drawdown")
ax2.set_title("The effect mostly disappears once sector is controlled")
ax2.axhline(0, color="#e0e0e0", lw=0.8)
ax2.set_ylim(-0.33, 0.33)
ax2.legend(facecolor="#0a0a0a", edgecolor="#333", labelcolor="#e0e0e0",
           fontsize=8, loc="lower left")

for a in (ax1, ax2):
    for s in ["top", "right"]:
        a.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        a.spines[s].set_color("#333")
plt.tight_layout()
plt.savefig("balance-sheet-drawdown-protection-python.png", dpi=150, facecolor="#0a0a0a")
