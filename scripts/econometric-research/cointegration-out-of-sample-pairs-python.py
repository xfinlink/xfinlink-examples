# Full write-up: https://xfinlink.com/blog/cointegration-out-of-sample-pairs-python
#
# Does in-sample cointegration survive out of sample? Engle-Granger on every
# same-industry pair of S&P 500 members, formation window against holdout window.

import itertools

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xfinlink as xfl
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.stattools import coint

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

FORM_START, FORM_END, HOLD_END = "2020-08-05", "2023-08-04", "2026-08-04"
ENTRY, STOP, LEVEL = 2.0, 60, 0.05

# --- universe: index members as at the formation start, grouped by GICS industry
members = xfl.index("sp500", as_of=FORM_START).dropna(subset=["entity_id"])
members["entity_id"] = members["entity_id"].astype(int)

rows = []
tk = members["ticker"].tolist()
for i in range(0, len(tk), 10):
    for sym, hit in xfl.resolve(tk[i:i + 10])["data"].items():
        for e in hit["entities"]:
            rows.append({"ticker": sym, "entity_id": e["entity_id"],
                         "industry": e["classifications"].get("gics_industry")})
uni = pd.DataFrame(rows).merge(members[["entity_id", "ticker"]], on=["entity_id", "ticker"])

top10 = uni.groupby("industry")["ticker"].count().sort_values(ascending=False).head(10)
uni = uni[uni["industry"].isin(top10.index)].reset_index(drop=True)

ids = sorted(uni["entity_id"])
px = pd.concat([xfl.prices(entity_id=ids[i:i + 5], start=FORM_START, end=HOLD_END,
                           fields=["adj_close"], max_rows=200000)
                for i in range(0, len(ids), 5)], ignore_index=True)

# --- panel: one symbol per entity, complete daily history, positive prices
one = px.groupby("entity_id")["ticker"].nunique()
px = px[px["entity_id"].isin(one[one == 1].index)]
wide = px.pivot_table(index="date", columns="entity_id", values="adj_close").sort_index()
form, hold = wide.loc[:FORM_END], wide.loc[wide.index > FORM_END]
keep = [i for i in wide.columns
        if form[i].notna().all() and hold[i].notna().all() and wide[i].min() > 0]
uni = uni[uni["entity_id"].isin(keep)]
lf, lh = np.log(form[keep]), np.log(hold[keep])
sym = dict(zip(uni["entity_id"], uni["ticker"]))


def trades(z):
    """Enter on a fresh crossing of the band, exit at z = 0 or after STOP sessions."""
    out, i, n, armed = [], 0, len(z), abs(z[0]) < ENTRY
    while i < n:
        if armed and abs(z[i]) >= ENTRY:
            side, j = np.sign(z[i]), i + 1
            while j < n and j - i <= STOP and side * z[j] > 0:
                j += 1
            k = min(j, n - 1)
            out.append((j < n and j - i <= STOP, k - i, side * (z[i] - z[k])))
            i, armed = k + 1, False
        else:
            armed = armed or abs(z[i]) < ENTRY
            i += 1
    return out


pairs, book = [], []
for industry, grp in uni.groupby("industry"):
    for a, b in itertools.combinations(sorted(grp["entity_id"], key=lambda e: sym[e]), 2):
        p_form = coint(lf[b].values, lf[a].values, trend="c", autolag="AIC")[1]
        p_hold = coint(lh[b].values, lh[a].values, trend="c", autolag="AIC")[1]

        fit = OLS(lf[b].values, add_constant(lf[a].values)).fit()
        sd = fit.resid.std(ddof=1)
        z = (lh[b].values - fit.params[0] - fit.params[1] * lh[a].values
             - fit.resid.mean()) / sd

        selected = p_form < LEVEL
        pairs.append({"pair": f"{sym[a]}/{sym[b]}", "industry": industry,
                      "p_form": p_form, "p_hold": p_hold, "selected": selected})
        for converged, days, sigma in trades(z):
            book.append({"pair": f"{sym[a]}/{sym[b]}", "selected": selected,
                         "converged": converged, "days": days,
                         "sigma": sigma, "logret": sigma * sd})

res, tr = pd.DataFrame(pairs), pd.DataFrame(book)
sel, rej = res["selected"], ~res["selected"]

print(f"Panel: {len(keep)} names, {len(res)} same-industry pairs, "
      f"{len(form)} formation sessions, {len(hold)} holdout sessions")
print(f"Formation cointegrated at 5%: {sel.sum()} ({100 * sel.mean():.2f}%), "
      f"chance alone predicts {LEVEL * len(res):.1f}")
print(f"Holdout cointegrated at 5%:   {(res['p_hold'] < LEVEL).sum()} "
      f"({100 * (res['p_hold'] < LEVEL).mean():.2f}%)")
print(f"Repeat rate, selected pairs:  {(sel & (res['p_hold'] < LEVEL)).sum()}/{sel.sum()} "
      f"({100 * res.loc[sel, 'p_hold'].lt(LEVEL).mean():.1f}%)")
print(f"Repeat rate, rejected pairs:  {(rej & (res['p_hold'] < LEVEL)).sum()}/{rej.sum()} "
      f"({100 * res.loc[rej, 'p_hold'].lt(LEVEL).mean():.1f}%)")
print(f"At the 1% level: {(res['p_form'] < 0.01).sum()} selected, "
      f"{((res['p_form'] < 0.01) & (res['p_hold'] < 0.01)).sum()} repeat")

print("\nHoldout entry rule: enter at |z| >= 2, exit at z = 0 or after 60 sessions")
print(f"{'group':<14}{'trades':>7}{'converged':>11}{'median days':>13}"
      f"{'mean sigma':>12}{'mean log ret':>14}")
for label, mask in (("selected", tr["selected"]), ("rejected", ~tr["selected"])):
    d = tr[mask]
    print(f"{label:<14}{len(d):>7}{100 * d['converged'].mean():>10.1f}%"
          f"{d.loc[d['converged'], 'days'].median():>13.0f}"
          f"{d['sigma'].mean():>12.3f}{100 * d['logret'].mean():>13.2f}%")
for label, mask in (("converged", tr["converged"]), ("timed out", ~tr["converged"])):
    print(f"  {label} trades: mean {tr.loc[mask, 'sigma'].mean():+.2f} sigma")

# --- chart
plt.rcParams.update({"figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
                     "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
                     "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
                     "axes.edgecolor": "#3a3a3a", "font.size": 10})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

ax1.scatter(res.loc[rej, "p_form"], res.loc[rej, "p_hold"], s=9, c="#4b5563", alpha=.55,
            edgecolors="none", label="rejected in formation")
ax1.scatter(res.loc[sel, "p_form"], res.loc[sel, "p_hold"], s=22, c="#3b82f6",
            edgecolors="none", label="cointegrated in formation")
ax1.axvline(LEVEL, color="#e0e0e0", lw=.8, ls="--")
ax1.axhline(LEVEL, color="#e0e0e0", lw=.8, ls="--")
ax1.set_xscale("log"), ax1.set_yscale("log")
ax1.set_xlabel("Cointegration p-value, 2020-2023")
ax1.set_ylabel("Cointegration p-value, 2023-2026")
ax1.set_title("Same-industry pairs, tested twice", fontsize=11)
ax1.legend(fontsize=8, frameon=False, loc="lower left")

rates = [100 * tr.loc[tr["selected"], "converged"].mean(),
         100 * tr.loc[~tr["selected"], "converged"].mean()]
bars = ax2.bar(["cointegrated\nin formation", "rejected\nin formation"], rates,
               color=["#3b82f6", "#4b5563"], width=.55)
for bar, v in zip(bars, rates):
    ax2.text(bar.get_x() + bar.get_width() / 2, v + .5, f"{v:.1f}%", ha="center", fontsize=10)
ax2.set_ylabel("Divergences back at fair value within 60 sessions")
ax2.set_title("What the entry rule caught out of sample", fontsize=11)
ax2.set_ylim(0, max(rates) * 1.25)
ax2.spines[["top", "right"]].set_visible(False)
ax1.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig("cointegration-out-of-sample-pairs-python.png", dpi=150, facecolor="#0a0a0a")
