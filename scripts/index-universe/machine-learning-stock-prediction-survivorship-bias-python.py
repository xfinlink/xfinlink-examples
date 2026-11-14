# Full write-up: https://xfinlink.com/blog/machine-learning-stock-prediction-survivorship-bias-python
#
# Does survivorship bias in the training data change what a stock-return model learns?
# Two walk-forward classifiers predict whether an S&P 500 member beats the month's median
# return next month from four price-only features known at the month end. Model A trains on
# the roster in force at each month end (point-in-time); model B trains on today's roster
# applied to every month. Both are scored on the point-in-time universe, 2005-2024.
# Holdings that stop trading take their recorded delisting return in the exit month.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xfinlink as xfl
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup
xfl.set_timeout(300)
CHART = "machine-learning-stock-prediction-survivorship-bias-python.png"

# ---- data: monthly rosters, today's roster, monthly total returns by entity id ------------
dates = pd.date_range("1999-12-01", "2024-11-30", freq="BME")
rosters = {d.to_period("M") + 1: xfl.index("sp500", as_of=d.strftime("%Y-%m-%d")) for d in dates}
pit = {m: set(r.entity_id.dropna().astype(int)) for m, r in rosters.items()}   # roster sets the next month
now = xfl.index("sp500")                                                        # today's roster
today = set(now.entity_id.dropna().astype(int))
ids = sorted(set().union(*pit.values()) | today)
parts = [xfl.prices(entity_id=ids[i:i + 25], start="1998-12-01", end="2024-12-31",
                    interval="1mo", fields=["return_daily"], max_rows=100_000)
         for i in range(0, len(ids), 25)]
px = pd.concat(parts)
px["m"] = px["date"].dt.to_period("M")
R = px.drop_duplicates(["entity_id", "m"]).pivot(index="m", columns="entity_id", values="return_daily")
R = R.reindex(pd.period_range("1998-12", "2024-12", freq="M"))

# A holding that stops trading takes its recorded delisting return (meta.delisting) in the exit
# month: the month of its last bar when the exit falls in it, else the month after. Exit records
# are read one exit year at a time, one page per call
last = R.apply(pd.Series.last_valid_index)
ended = last[(last >= pd.Period("1999-12", "M")) & (last < pd.Period("2024-12", "M"))]
delist = {}
for y, g in ended.groupby(ended.map(lambda p: p.year)):
    g = g.index.tolist()
    for i in range(0, len(g), 150):
        d = xfl.prices(entity_id=g[i:i + 150], start=f"{y}-01-01", end="2024-12-31", interval="1mo",
                       fields=["return_daily"])
        delist.update({int(k): v for k, v in (d.attrs["meta"].get("delisting") or {}).items()})
applied = 0
for e, v in delist.items():
    dr = v.get("delisting_return")
    if e not in R or last[e] is None or dr is None or np.isnan(dr):
        continue
    m = last[e] if pd.Period(v["event_date"], "M") <= last[e] else last[e] + 1
    if m <= R.index[-1]:
        R.loc[m, e] = (1 + np.nan_to_num(R.loc[m, e])) * (1 + dr) - 1
        applied += 1

# ---- features at each month end t (price-only, known at t) and next month's return --------
lvl = (1 + R.fillna(0)).cumprod()
FEATS = {"ret_1m": R,                                                     # last month's return
         "mom_12_1": np.expm1(np.log1p(R).rolling(11).sum().shift(1)),    # months t-11 to t-1
         "vol_12m": R.rolling(12).std(),                                  # volatility of monthly returns
         "off_high": lvl / lvl.rolling(12).max() - 1}                     # distance below 12-month high
full = R.rolling(12).count() == 12                                        # 12 priced months to date
fwd = R.shift(-1)                                                         # the label month
X = pd.concat({k: f.where(full).stack() for k, f in FEATS.items()}, axis=1).dropna()
X["y_ret"] = fwd.stack().reindex(X.index)
X.index.names = ["t", "e"]
X = X.loc["1999-12":"2024-11"].reset_index()
X["hold"] = X["t"] + 1
X["pit"] = [e in pit[h] for e, h in zip(X["e"], X["hold"])]
X["today"] = X["e"].isin(today)
F = list(FEATS)


def universe(col):
    """Rows of one universe; features as cross-sectional ranks in (-0.5, 0.5] and the label
    'beat this universe's median return next month', all computed inside that universe."""
    d = X[X[col] & X["y_ret"].notna()].copy()
    g = d.groupby("t")
    d[F] = g[F].rank(pct=True) - 0.5
    d["y"] = (d["y_ret"] > g["y_ret"].transform("median")).astype(int)
    return d


U = {"pit": universe("pit"), "today": universe("today")}
MODELS = {"Logistic regression": lambda: LogisticRegression(C=1.0, max_iter=1000),
          "Gradient boosting": lambda: HistGradientBoostingClassifier(
              max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=200,
              early_stopping=False, random_state=0)}
YEARS = range(2005, 2025)

# ---- walk-forward: retrain each January on every label realised by the prior December -----
pred, coef = [], []
for y in YEARS:
    for train in ("pit", "today"):
        tr = U[train][U[train]["hold"].dt.year < y]
        for mdl, make in MODELS.items():
            clf = make().fit(tr[F], tr["y"])
            if mdl == "Logistic regression":
                coef.append(pd.Series(clf.coef_[0], F, name=(train, y)))
            for test in ("pit", "today"):
                te = U[test][U[test]["hold"].dt.year == y]
                pred.append(te[["t", "e", "y", "y_ret", "today"]].assign(
                    p=clf.predict_proba(te[F])[:, 1], model=mdl, train=train, test=test))
P = pd.concat(pred, ignore_index=True)
P["dec"] = P.groupby(["model", "train", "test", "t"])["p"].transform(
    lambda s: np.ceil(s.rank(pct=True) * 10))


def score(d):
    by = d.groupby("t")
    top = d[d.dec == 10].groupby("t")["y_ret"].mean()
    bot = d[d.dec == 1].groupby("t")["y_ret"].mean()
    s = top - bot
    ic = by.apply(lambda g: g["p"].corr(g["y_ret"], method="spearman"))
    return pd.Series({"months": len(s), "names": by.size().mean(), "top": top.mean() * 12,
                      "bottom": bot.mean() * 12, "spread": s.mean() * 12,
                      "t": s.mean() / s.std() * len(s) ** 0.5, "auc": roc_auc_score(d["y"], d["p"]),
                      "ic": ic.mean(), "ic_t": ic.mean() / ic.std() * len(ic) ** 0.5})


S = P.groupby(["model", "train", "test"]).apply(score)
C = pd.concat(coef, axis=1).T.groupby(level=0).mean()

# ---- output -------------------------------------------------------------------------------
LBL = {"pit": "point-in-time", "today": "today's roster"}
name = pd.concat(rosters.values()).drop_duplicates("entity_id").set_index("entity_id")["ticker"]
print(f"Rosters: {len(pit)} month ends, Dec 1999 to Nov 2024; today's roster: {len(now)} names "
      f"as of {now.attrs['meta']['as_of']}")
print(f"Securities: {len(ids)}; exit months given a recorded delisting return: {applied}")
print(f"Rows with features and a label, 2000-2024: point-in-time {len(U['pit']):,}, "
      f"today's roster {len(U['today']):,}")
ex = U["pit"][U["pit"]["e"].map(name).isin(["ENE", "WCOM", "LEH", "WM", "FRC"])]
ex = ex.sort_values("hold").groupby("e").tail(1)
print("Last month held, point-in-time: " + ", ".join(
    f"{name[e]} {h} {r:+.1%}" for e, h, r in zip(ex["e"], ex["hold"], ex["y_ret"])))
print(f"\nWalk-forward test {YEARS[0]}-{YEARS[-1]} ({S['months'].iloc[0]:.0f} months), retrained each January")
print(f"{'model':21}{'trained on':16}{'tested on':16}{'names':>6}{'top':>8}{'bottom':>8}"
      f"{'spread':>8}{'t':>7}{'AUC':>7}{'rank IC':>8}")
for (mdl, tr, te), r in S.iterrows():
    print(f"{mdl:21}{LBL[tr]:16}{LBL[te]:16}{r['names']:6.0f}{r['top']:8.1%}{r['bottom']:8.1%}"
          f"{r['spread'] * 100:+8.2f}{r['t']:7.2f}{r['auc']:7.3f}{r['ic']:+8.4f}")
print("\nLogistic regression coefficients (rank features, mean of 20 yearly fits)")
print(f"{'':16}" + "".join(f"{f:>10}" for f in F))
for tr in ("pit", "today"):
    print(f"{LBL[tr]:16}" + "".join(f"{C.loc[tr, f]:+10.3f}" for f in F))
print("\nTraining data 2000-2024, next month: share beating the median / mean return a year")
for f, label in [("vol_12m", "most volatile tenth"), ("off_high", "tenth furthest below its 12-month high")]:
    tenth = (lambda d: d[d[f] > 0.4]) if f == "vol_12m" else (lambda d: d[d[f] <= -0.4])
    pt = tenth(U["pit"])
    print(f"  {label}")
    for a, d in [(LBL["today"], tenth(U["today"])), (LBL["pit"], pt), ("  still in the index today", pt[pt["today"]]),
                 (f"  since removed ({(~pt['today']).mean():.0%})", pt[~pt["today"]])]:
        print(f"    {a:30}{d['y'].mean():7.1%}{d['y_ret'].mean() * 12:8.1%}")
T = P[(P["test"] == "pit") & (P["dec"] == 10)].merge(U["pit"][["t", "e", "vol_12m"]], on=["t", "e"])
print("\nTop decile on the point-in-time universe, 2005-2024: share since removed / volatility rank")
for (mdl, tr), d in T.groupby(["model", "train"]):
    print(f"  {mdl:21}trained on {LBL[tr]:16}{(~d['today']).mean():7.1%}{d['vol_12m'].mean():+8.3f}")

# ---- chart: top-minus-bottom decile spread by training set and test universe ---------------
plt.rcParams.update({"figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a", "text.color": "#e0e0e0",
                     "axes.labelcolor": "#e0e0e0", "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
                     "axes.edgecolor": "#444444", "font.size": 11})
cases = [("today", "today", "Trained and tested on\ntoday's roster"),
         ("today", "pit", "Trained on today's roster,\ntested point-in-time"),
         ("pit", "pit", "Trained and tested\npoint-in-time")]
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(cases))
for off, (mdl, colour) in zip([-0.2, 0.2], [("Logistic regression", "#3b82f6"), ("Gradient boosting", "#d97706")]):
    v = [S.loc[(mdl, tr, te), "spread"] * 100 for tr, te, _ in cases]
    t = [S.loc[(mdl, tr, te), "t"] for tr, te, _ in cases]
    ax.bar(x + off, v, width=0.38, color=colour, label=mdl, edgecolor="#0a0a0a", linewidth=2)
    for xi, vi, ti in zip(x + off, v, t):
        ax.text(xi, vi + (0.25 if vi >= 0 else -0.25), f"{vi:+.1f}\n(t {ti:.2f})", ha="center",
                va="bottom" if vi >= 0 else "top", fontsize=9)
ax.axhline(0, color="#666666", linewidth=1)
ax.set_xticks(x)
ax.set_xticklabels([c[2] for c in cases])
ax.set_ylabel("Top minus bottom decile, points a year")
ax.set_ylim(-5, 10.5)
ax.set_title("Survivorship bias in machine learning stock prediction, S&P 500, 2005-2024")
ax.legend(frameon=False, loc="upper right")
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
plt.tight_layout()
plt.savefig(CHART, dpi=150)
