# Full write-up: https://xfinlink.com/blog/stock-recovery-after-50-percent-drop-python
#
# Do S&P 500 stocks recover after a 50% drop? Every time a member's total-return index
# first closed a month 50% or more below its running peak, 1995-2019, is followed for
# 60 months: did it regain the peak, stop trading first, or stay below? The same rule
# is then run on today's S&P 500 list, the way a survivor-list study would run it.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

DROP, HORIZON = 0.50, 60                      # 50% below the peak; 60 months to recover
FIRST, LAST = pd.Period("1995-01", "M"), pd.Period("2019-12", "M")
CHART = "stock-recovery-after-50-percent-drop-python.png"

# Membership spells: the January 1995 roster, plus every addition to 2019 paired with
# its next removal
ev = pd.concat([xfl.index_events("sp500", end="2024-12-31", offset=o)
                for o in range(0, 4000, 1000)], ignore_index=True).dropna(subset=["entity_id"])
ev["entity_id"] = ev["entity_id"].astype(int)
ev["d"] = pd.to_datetime(ev["effective_date"])
r95 = xfl.index("sp500", as_of="1995-01-03").dropna(subset=["entity_id"])
spells = [(int(r.entity_id), pd.Timestamp(r.added_date), pd.Timestamp(r.removed_date))
          for r in r95.itertuples()]
rem = ev[ev["event_type"] == "removed"]
for a in ev[(ev["event_type"] == "added") & ev["d"].between("1995-01-04", "2019-12-31")].itertuples():
    spells.append((a.entity_id, a.d, rem.loc[(rem["entity_id"] == a.entity_id) & (rem["d"] >= a.d), "d"].min()))
spells = pd.DataFrame(spells, columns=["entity_id", "start", "end"])
today = set(xfl.index("sp500")["entity_id"].dropna().astype(int))

# Monthly total returns, 1990-2024, for every 1995-2019 member and every member today
ids = sorted(set(spells["entity_id"]) | today)
px = pd.concat([xfl.prices(entity_id=ids[i:i + 300], start="1990-01-01", end="2024-12-31",
                           interval="1mo", fields=["return_daily"],
                           max_rows=200_000) for i in range(0, len(ids), 300)], ignore_index=True)
px["m"] = px["date"].dt.to_period("M")
px = px.drop_duplicates(["entity_id", "m"]).sort_values(["entity_id", "m"])

def member_months(eid, months):
    """True for each month the stock ends as an S&P 500 member."""
    ends = months.to_timestamp(how="end").normalize()
    ok = np.zeros(len(months), bool)
    for s in spells[spells["entity_id"] == eid].itertuples():
        ok |= (ends >= s.start) & (pd.isna(s.end) | (ends <= s.end))
        if pd.notna(s.end):  # removed in the month its shares stopped trading
            ok |= (months == s.end.to_period("M")) & (months == months[-1])
    return ok

def episodes(eid, g, need_member):
    """One event per drawdown episode: the first month-end 50%+ below the running peak."""
    out = []
    # A month without a return ends a price series; prices that resume later start a new one
    g = g.assign(return_daily=g["return_daily"].fillna({g.index[0]: 0.0}))
    seg = g["return_daily"].isna().cumsum()
    for _, s in g[g["return_daily"].notna()].groupby(seg):
        r = s.set_index("m")["return_daily"]
        months = pd.PeriodIndex(r.index)
        w = (1 + r).cumprod().to_numpy()                 # total-return index
        peak = np.maximum.accumulate(w)
        mem = member_months(eid, months) if need_member else np.ones(len(w), bool)
        k = 0
        while True:
            hit = np.flatnonzero(w[k:] <= (1 - DROP) * peak[k:])
            if not len(hit):
                break
            t = k + hit[0]                               # first month-end 50% below the peak
            back = np.flatnonzero(w[t + 1:] >= peak[t])
            if FIRST <= months[t] <= LAST and mem[t]:
                if len(back) and back[0] < HORIZON:
                    outcome, mo = "regained the peak", back[0] + 1
                elif len(w) - 1 - t < HORIZON:
                    outcome, mo = "series ended first", np.nan
                else:
                    outcome, mo = "still below at 60 months", np.nan
                out.append({"entity_id": eid, "ticker": s["ticker"].iloc[t], "month": months[t],
                            "peak_month": months[int(np.argmax(w[:t + 1]))], "outcome": outcome,
                            "months": mo, "trough": w[t:t + 1 + int(np.nan_to_num(mo, nan=HORIZON))].min() / peak[t] - 1,
                            "end_vs_peak": w[min(t + HORIZON, len(w) - 1)] / peak[t] - 1})
            if not len(back):
                break
            k = t + 1 + back[0]                          # a new episode only after the peak is regained
    return out

books = {}
pit_ids = set(spells["entity_id"])
for name, universe, need_member in [("Point-in-time rosters", pit_ids, True),
                                    ("Today's list", today, False),
                                    ("Today's list, member months only", today, True)]:
    rows = []
    for eid, g in px[px["entity_id"].isin(universe)].groupby("entity_id"):
        rows += episodes(eid, g.reset_index(drop=True), need_member)
    book = pd.DataFrame(rows)
    # One share class per company: two classes in the index crash in the same month
    book["company"] = book["entity_id"].map(px.groupby("entity_id")["entity_name"].last())
    books[name] = book.drop_duplicates(["company", "month"]).reset_index(drop=True)

# ---- output ------------------------------------------------------------------
OUT = ["regained the peak", "series ended first", "still below at 60 months"]
pit, surv = books["Point-in-time rosters"], books["Today's list"]
print(f"S&P 500 spells 1995-2019: {len(spells)} ({len(pit_ids)} securities); today's list: {len(today)}")
print("Events: the first month-end close 50%+ below the running peak of the total-return index, "
      "1995-2019, followed 60 months\n")
print(f"{'':34}{'events':>7}{'cos':>5}{'regained':>10}{'ended':>9}{'below':>7}{'median mo':>11}")
for name, b in books.items():
    sh = b["outcome"].value_counts(normalize=True).reindex(OUT, fill_value=0)
    print(f"{name:<34}{len(b):>7}{b['entity_id'].nunique():>5}{sh.iloc[0]:>10.1%}{sh.iloc[1]:>9.1%}"
          f"{sh.iloc[2]:>7.1%}{b['months'].median():>11.1f}")

print("\nShare back at the old peak within N months of the 50% close")
for name, b in books.items():
    print(f"  {name:<34}" + "".join(f"{n:>3}m {(b['months'] <= n).mean():6.1%} " for n in (12, 24, 36, 60)))

on = pit["entity_id"].isin(today)
print(f"\nPoint-in-time events, split by today's list")
for label, b in [("company on today's list", pit[on]), ("company not on today's list", pit[~on])]:
    sh = b["outcome"].value_counts(normalize=True).reindex(OUT, fill_value=0)
    print(f"  {label:<28} {len(b):>4} events: regained {sh.iloc[0]:.1%}, ended {sh.iloc[1]:.1%}, "
          f"below {sh.iloc[2]:.1%}")
print(f"Today's-list events in months the company was not yet (or no longer) a member: "
      f"{len(surv) - len(books[list(books)[2]])}")
st = pit[pit["outcome"] == "series ended first"]
print(f"Series ended first ({len(st)}): last close vs old peak, median {st['end_vs_peak'].median():.1%}; "
      f"within 25% of the peak {(st['end_vs_peak'] > -0.25).sum()}; 90%+ below {(st['end_vs_peak'] <= -0.9).sum()}")

print("\nRegained the peak within 60 months, by the deepest month-end before regaining")
bins, labels = [0.0, 0.6, 0.75, 1.0], ["50-60% down", "60-75% down", "75%+ down"]
for name, b in books.items():
    d = pd.cut(-b["trough"], bins, labels=labels)
    t = b.groupby(d, observed=True).agg(n=("outcome", "size"),
                                         reg=("outcome", lambda x: (x == OUT[0]).mean()))
    print(f"  {name:<34}" + "".join(f"{i} {r.reg:6.1%} (n={int(r.n):>3})  " for i, r in t.iterrows()))

print("\nNamed events (point-in-time book; ticker in force at the event)")
for tk, mo in [("LU", "2000-09"), ("CSCO", "2000-12"), ("AAPL", "2000-09"), ("ENE", "2001-08"),
               ("WCOM", "2000-10"), ("C", "2008-02"), ("AIG", "2008-02"), ("LEH", "2008-03"),
               ("WB", "2008-03"), ("BBY", "2008-10"), ("AMZN", "2008-11"), ("NFLX", "2011-09")]:
    r = pit[(pit["ticker"] == tk) & (pit["month"] == pd.Period(mo, "M"))].iloc[0]
    took = f"{r['months']:.0f} months" if r["outcome"] == OUT[0] else ""
    print(f"  {tk:<5} {r['month']}  peak {r['peak_month']}  deepest {r['trough']:7.1%}  "
          f"{r['outcome']:<25}{took}")

# ---- chart -------------------------------------------------------------------
plt.rcParams.update({"figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
                     "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
                     "xtick.color": "#a3a3a3", "ytick.color": "#a3a3a3", "axes.edgecolor": "#404040"})
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(0, HORIZON + 1)
for name, colour, style in [("Today's list", "#d97706", "-"),
                            ("Today's list, member months only", "#a3a3a3", "--"),
                            ("Point-in-time rosters", "#3b82f6", "-")]:
    b = books[name]
    y = [(b["months"] <= m).mean() * 100 for m in x]
    ax.plot(x, y, color=colour, linestyle=style, linewidth=2, label=f"{name} ({len(b)} events)")
    ax.annotate(f"{y[-1]:.1f}%", (x[-1], y[-1]), xytext=(6, 0), textcoords="offset points",
                va="center", color="#e0e0e0", fontsize=10)
ax.set_xlim(0, HORIZON + 7)
ax.set_xticks(range(0, HORIZON + 1, 12))
ax.set_ylim(0, 70)
ax.set_xlabel("Months after the first month-end 50% below the peak")
ax.set_ylabel("Share of events back at the old peak (%)")
ax.set_title("S&P 500 stocks after a 50% drop, 1995-2019: share back at the old peak\n"
             "point-in-time rosters vs today's list (total return, month-end closes)", fontsize=12)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, loc="upper left")
plt.tight_layout()
plt.savefig(CHART, dpi=150)
