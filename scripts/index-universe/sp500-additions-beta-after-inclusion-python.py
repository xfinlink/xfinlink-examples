# Full write-up: https://xfinlink.com/blog/sp500-additions-beta-after-inclusion-python
#
# S&P 500 additions and removals, 1995-2023: does a stock's weekly beta against SPY rise
# after it joins the index, and fall after it leaves? Additions are compared with incumbent
# members measured over the same calendar weeks.

import numpy as np
import pandas as pd
import xfinlink as xfl
from scipy import stats

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

LAST_WEEK = pd.Timestamp("2024-12-23")     # every post window ends inside 2024
SKIP, N = 4, 52                            # skip 4 weeks either side of the event, then 52 weeks
rng = np.random.default_rng(0)


def weekly(px):
    """Weekly total returns keyed on the Monday of each week, one column per security."""
    px = px.dropna(subset=["return_daily"]).copy()
    px["week"] = px["date"] - pd.to_timedelta(px["date"].dt.day_of_week, unit="D")
    return px.groupby(["entity_id", "week"])["return_daily"].apply(lambda r: np.prod(1 + r) - 1).unstack(0)


# 1. Every S&P 500 addition and removal since 1990 (offset paging, 1,000 rows a call; the
# "returned 1000 of 1675 rows" warning on the first call is expected, the second call fetches the rest)
pages = [xfl.index_events("sp500", start="1990-01-01", end="2024-12-31", limit=1000, offset=o) for o in (0, 1000)]
ev = pd.concat(pages, ignore_index=True)
ev["d"] = pd.to_datetime(ev["effective_date"])
ev["year"] = ev["d"].dt.year


def member(e, t):
    """Was security e in the index on date t, read off its own dated events."""
    x = ev[ev["entity_id"] == e].sort_values("d")
    before, after = x[x["d"] <= t], x[x["d"] > t]
    if len(before):
        return before["event_type"].iloc[-1] == "added"
    return len(after) > 0 and after["event_type"].iloc[0] == "removed"


# 2. Weekly returns: SPY, the event stocks and 60 incumbents per cohort year
spy = weekly(xfl.prices("SPY", start="1993-01-01", end="2024-12-31", interval="1w", fields=["return_daily"])).iloc[:, 0]
stock_r, ctrl_r = {}, {}
for y in range(1995, 2024):
    start, end = f"{y - 2}-10-01", min(f"{y + 2}-03-31", "2024-12-31")
    ids = sorted(set(ev.loc[ev["year"] == y, "entity_id"].astype(int)))
    stock_r[y] = weekly(xfl.prices(entity_id=ids, start=start, end=end, interval="1w",
                                   fields=["return_daily"], max_rows=200_000))
    ro = xfl.index("sp500", as_of=start)                          # roster when the cohort's window opens
    inc = ro[ro["removed_date"].isna() | (ro["removed_date"] >= end)]  # members for the whole window
    pick = sorted(rng.choice(inc["entity_id"].dropna().astype(int).unique(), 60, replace=False).tolist())
    ctrl_r[y] = weekly(xfl.prices(entity_id=pick, start=start, end=end, interval="1w",
                                  fields=["return_daily"], max_rows=200_000))


def beta(r, m):
    return np.cov(r, m)[0, 1] / np.var(m, ddof=1), np.corrcoef(r, m)[0, 1] ** 2


# 3. Beta and R-squared over weeks -56..-5 and +5..+56, for the stock and for each incumbent
rows = []
for x in ev[(ev["year"] >= 1995) & (ev["year"] <= 2023)].itertuples():
    w0 = x.d - pd.Timedelta(days=x.d.day_of_week)
    pre = pd.date_range(w0 - pd.Timedelta(weeks=SKIP + N), periods=N, freq="7D")
    post = pd.date_range(w0 + pd.Timedelta(weeks=SKIP + 1), periods=N, freq="7D")
    if post[-1] > LAST_WEEK:
        continue
    rec = {"e": x.entity_id, "ticker": x.ticker, "type": x.event_type, "d": x.d, "year": x.year}
    s = stock_r[x.year].get(x.entity_id)
    a, b = (s.reindex(pre), s.reindex(post)) if s is not None else (None, None)
    full = s is not None and a.notna().all() and b.notna().all()  # 52 priced weeks on each side
    # additions: outside the index before, inside after; removals: the reverse
    m0, m1 = member(x.entity_id, pre[0]), member(x.entity_id, post[-1] + pd.Timedelta(days=4))
    rec["kept"] = full and ((m0, m1) == ((False, True) if x.event_type == "added" else (True, False)))
    if rec["kept"]:
        mp, mq = spy.reindex(pre).values, spy.reindex(post).values
        (rec["b0"], rec["r0"]), (rec["b1"], rec["r1"]) = beta(a.values, mp), beta(b.values, mq)
        rec["excess_ret_pre"] = np.prod(1 + a) - np.prod(1 + spy.reindex(pre))
        c = ctrl_r[x.year].drop(columns=[x.entity_id], errors="ignore")
        ca, cb = c.reindex(pre), c.reindex(post)
        ok = ca.notna().all() & cb.notna().all()
        f0 = np.array([beta(ca[k].values, mp) for k in ok[ok].index])
        f1 = np.array([beta(cb[k].values, mq) for k in ok[ok].index])
        cb0, cb1 = f0[:, 0], f1[:, 0]
        rec["ctrl_db"] = (cb1 - cb0).mean()                          # incumbents, same weeks
        rec["ctrl_dr2"] = (f1[:, 1] - f0[:, 1]).mean()
        slope, icpt = np.polyfit(cb0, cb1 - cb0, 1)                  # incumbents' change given their pre-beta
        rec["ctrl_db_matched"] = icpt + slope * rec["b0"]
    rows.append(rec)
R = pd.DataFrame(rows)
K = R[R["kept"]].copy()
K["db"] = K["b1"] - K["b0"]
K["excess"] = K["db"] - K["ctrl_db"]
K["excess_matched"] = K["db"] - K["ctrl_db_matched"]
K["excess_r2"] = (K["r1"] - K["r0"]) - K["ctrl_dr2"]
today = set(xfl.index("sp500")["entity_id"].dropna().astype(int))   # today's roster
K["member_today"] = K["e"].isin(today)


def test(v):
    t = stats.ttest_1samp(v, 0)
    return v.mean(), v.median(), t.statistic, t.pvalue


add, rem = K[K["type"] == "added"], K[K["type"] == "removed"]
add_all, rem_all = R[R["type"] == "added"], R[R["type"] == "removed"]
print(f"Additions 1995-2023 whose post window ends in 2024 or earlier: {len(add_all)}; "
      f"kept (52 priced weeks each side, outside the index before, inside after): {len(add)}")
print(f"Removals on the same terms (inside before, outside after, still trading 56 weeks on): {len(rem)} of {len(rem_all)}")
for name, g in (("Additions", add), ("Removals", rem)):
    print(f"\n{name}: beta {g['b0'].mean():.3f} -> {g['b1'].mean():.3f}, R-squared {g['r0'].mean():.3f} -> {g['r1'].mean():.3f}; "
          f"incumbents over the same weeks: beta {g['ctrl_db'].mean():+.3f}, R-squared {g['ctrl_dr2'].mean():+.3f}")

print("\nChange after the event, net of incumbents         n    mean  median      t       p  Wilcoxon p")
tests = [("Added: beta", add["excess"]), ("Added: beta, incumbents matched on pre-beta", add["excess_matched"]),
         ("Added: R-squared", add["excess_r2"]), ("Added: beta, 29 cohort-year means", add.groupby("year")["excess"].mean()),
         ("Added: R-squared, 29 cohort-year means", add.groupby("year")["excess_r2"].mean()),
         ("Added: beta, matched, 29 cohort-year means", add.groupby("year")["excess_matched"].mean()),
         ("Removed: beta", rem["excess"]), ("Removed: beta, incumbents matched on pre-beta", rem["excess_matched"]),
         ("Removed: R-squared", rem["excess_r2"])]
for label, v in tests:
    m, md, t, p = test(v)
    print(f"{label:<46} {len(v):>4}  {m:+.3f}  {md:+.3f}  {t:+5.2f}  {p:.4f}  {stats.wilcoxon(v).pvalue:.4f}")

early = add["year"] <= 2007
hi_ret = add["excess_ret_pre"] > add["excess_ret_pre"].median()
groups = {"All additions": early | ~early, "Added 1995-2007": early, "Added 2008-2023": ~early,
          "Added outside 1998-2001": ~add["year"].between(1998, 2001),
          "Top half by pre-addition return vs SPY": hi_ret, "Bottom half by pre-addition return vs SPY": ~hi_ret,
          "1995-2007, still in the index today": early & add["member_today"],
          "1995-2007, removed since": early & ~add["member_today"],
          "2008-2023, still in the index today": ~early & add["member_today"],
          "2008-2023, removed since": ~early & ~add["member_today"]}
print("\nAdditions, net of incumbents (mean, t, p)          n  beta                      beta, pre-beta matched    R-squared")
cell = lambda v: "{:+.3f} ({:+.2f}, p={:.3f})".format(*[test(v)[j] for j in (0, 2, 3)])
sub = []
for label, mask in groups.items():
    g = add[mask]
    sub.append((label, test(g["excess"])[0], g["excess"].sem(), test(g["excess_matched"])[0], g["excess_matched"].sem()))
    print(f"{label:<46} {mask.sum():>4}  {cell(g['excess']):<25} {cell(g['excess_matched']):<25} {cell(g['excess_r2'])}")

# ---- chart: beta and R-squared change net of incumbents, by group ----
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a", "text.color": "#e0e0e0",
                     "axes.labelcolor": "#e0e0e0", "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
                     "axes.edgecolor": "#444444", "font.size": 9})
rows = [(label, add[mask]) for label, mask in groups.items()] + [("Removals (left the index)", rem)]
labels = [r[0] for r in rows][::-1]
fig, axes = plt.subplots(1, 2, figsize=(10, 7), sharey=True)
for ax, col, name in zip(axes, ["excess", "excess_r2"], ["Beta change", "R-squared change"]):
    vals = [test(g[col]) for _, g in rows][::-1]
    sems = [g[col].sem() for _, g in rows][::-1]
    colors = ["#3b82f6" if v[3] < 0.05 else "#3b4a63" for v in vals]
    y = np.arange(len(rows))
    ax.barh(y, [v[0] for v in vals], height=0.6, color=colors, xerr=[1.96 * s for s in sems],
            error_kw={"ecolor": "#9ca3af", "elinewidth": 1, "capsize": 2})
    ax.axvline(0, color="#6b7280", lw=0.8)
    hi = max(v[0] + 1.96 * e for v, e in zip(vals, sems))
    lo = min(0, min(v[0] - 1.96 * e for v, e in zip(vals, sems)))
    ax.set_xlim(lo - 0.05 * (hi - lo), hi + 0.3 * (hi - lo))
    for yi, v in zip(y, vals):
        ax.text(hi + 0.28 * (hi - lo), yi, f"{v[0]:+.3f}", ha="right", va="center", fontsize=8)
    ax.set_yticks(y, labels)
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.set_xlabel(f"{name}, net of incumbent members")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
fig.suptitle("S&P 500 additions 1995-2023: change in weekly beta and R-squared vs SPY\n"
             "Bright bars: p < 0.05; faded bars: not significant; whiskers: 95% interval", fontsize=11)
plt.tight_layout()
plt.savefig("sp500-additions-beta-after-inclusion-python.png", dpi=150)
