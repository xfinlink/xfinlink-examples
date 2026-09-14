"""How many trading days are in a year? Count them instead of assuming 252.

Counts actual US equity sessions per calendar year from daily bars, compares
them against the weekday count, and shows what the difference does to an
annualisation factor.

    pip install -U xfinlink matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

START, END = "1996-01-01", "2025-12-31"

# One row per session. SPY has traded every US equity session in the window.
spy = xfl.prices("SPY", start=START, end=END, fields=["close"])
sessions = spy["date"]

# Cross-check: three long-lived listings should share the same session dates.
ref = set(sessions)
checks = {}
for t in ("KO", "JNJ", "IBM"):
    other = set(xfl.prices(t, start=START, end=END, fields=["close"])["date"])
    checks[t] = (len(other), len(ref - other), len(other - ref))

per_year = sessions.groupby(sessions.dt.year).size()
weekdays = pd.Series(
    {y: int(np.busday_count(f"{y}-01-01", f"{y + 1}-01-01")) for y in per_year.index}
)
gap = weekdays - per_year

# NYSE holidays for 2026, as published on nyse.com/markets/hours-calendars.
HOLIDAYS_2026 = [
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25",
    "2026-06-19", "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25",
]

print(f"US equity sessions per calendar year, {START} to {END}")
print("Source: daily bars, one row per session. Weekdays = Monday to Friday.\n")
print(f"{'Year':<6}{'Sessions':>9}{'Weekdays':>10}{'Gap':>6}   "
      f"{'Year':<6}{'Sessions':>9}{'Weekdays':>10}{'Gap':>6}")
years = list(per_year.index)
half = (len(years) + 1) // 2
for i in range(half):
    left = years[i]
    row = f"{left:<6}{per_year[left]:>9}{weekdays[left]:>10}{gap[left]:>6}"
    if i + half < len(years):
        r = years[i + half]
        row += f"   {r:<6}{per_year[r]:>9}{weekdays[r]:>10}{gap[r]:>6}"
    print(row)

print(f"\nSessions per year: mean {per_year.mean():.1f}, median {per_year.median():.0f}, "
      f"min {per_year.min()} ({per_year.idxmin()}), max {per_year.max()} ({per_year.idxmax()})")
print(f"Years landing on exactly 252: {(per_year == 252).sum()} of {len(per_year)}")
print(f"Weekdays per year: mean {weekdays.mean():.1f}, "
      f"min {weekdays.min()}, max {weekdays.max()}")
print(f"Window totals: {len(sessions):,} sessions against {weekdays.sum():,} weekdays, "
      f"a difference of {weekdays.sum() - len(sessions):,} days over {len(years)} years")

print("\nSame session dates across three long-lived listings")
for t, (n, missing, extra) in checks.items():
    print(f"  {t:<5}{n:>7} sessions   dates absent vs SPY: {missing}   dates not in SPY: {extra}")

print("\nThe observed-holiday list is not fixed")
jan_third_monday = {}
for y in years:
    days = pd.date_range(f"{y}-01-01", f"{y}-01-31", freq="D")
    mondays = [d for d in days if d.dayofweek == 0]
    jan_third_monday[y] = mondays[2]
traded_mlk = [y for y in years if jan_third_monday[y] in ref]
print(f"  Sessions on the third Monday of January: {', '.join(str(y) for y in traded_mlk)}")
def observed(ts):
    """Saturday holidays move to the Friday before, Sunday holidays to the Monday after."""
    return ts - pd.Timedelta(days=1) if ts.dayofweek == 5 else (
        ts + pd.Timedelta(days=1) if ts.dayofweek == 6 else ts)

june = {y: observed(pd.Timestamp(f"{y}-06-19")) for y in years}
missing_june = [y for y in years if june[y] not in ref]
print(f"  No session on the 19 June observance: "
      f"{', '.join(str(y) for y in missing_june)}")

wk26 = int(np.busday_count("2026-01-01", "2027-01-01"))
print(f"\n2026 scheduled: {wk26} weekdays minus {len(HOLIDAYS_2026)} published holidays "
      f"= {wk26 - len(HOLIDAYS_2026)} sessions")

vol = xfl.prices("SPY", start="2015-01-01", end="2025-12-31", fields=["volume"])
ratios = {}
for y in range(2015, 2026):
    nov = pd.date_range(f"{y}-11-01", periods=30, freq="D")
    thanksgiving = [x for x in nov if x.month == 11 and x.dayofweek == 3][3]
    row = vol[vol["date"] == thanksgiving + pd.Timedelta(days=1)]
    median = vol[vol["date"].dt.year == y]["volume"].median()
    ratios[y] = float(row["volume"].iloc[0]) / median

print("\nDay after Thanksgiving (an early close), SPY volume against that year's median session")
items = [f"{y} {r:.2f}x" for y, r in ratios.items()]
print("  " + "   ".join(items[:6]))
print("  " + "   ".join(items[6:]))
below = sum(1 for r in ratios.values() if r < 1)
print(f"  Below the year's median in {below} of {len(ratios)} years; "
      f"mean {sum(ratios.values()) / len(ratios):.2f}x")

print("\nWhat the gap does to an annualisation factor")
f_actual, f_weekday = per_year.mean(), weekdays.mean()
print(f"  sqrt({f_actual:.1f}) = {np.sqrt(f_actual):.3f}    "
      f"sqrt({f_weekday:.1f}) = {np.sqrt(f_weekday):.3f}")
for v in (0.15, 0.20, 0.30):
    scaled = v * np.sqrt(f_weekday / f_actual)
    print(f"  {v:.0%} annualised on the session count becomes "
          f"{scaled:.2%} on the weekday count  (+{scaled - v:.2%})")

fig, ax = plt.subplots(figsize=(10, 5), facecolor="#0a0a0a")
ax.set_facecolor("#0a0a0a")
ax.plot(years, weekdays.values, color="#e0e0e0", lw=1.4, ls="--", marker="o", ms=3.5,
        label="Weekdays (Mon-Fri)")
ax.plot(years, per_year.values, color="#3b82f6", lw=1.8, marker="o", ms=5,
        label="Actual sessions")
ax.axhline(252, color="#f59e0b", lw=1.1, label="The 252 convention")
ax.set_ylim(246, 264)
ax.set_xlabel("Calendar year", color="#e0e0e0")
ax.set_ylabel("Days", color="#e0e0e0")
ax.set_title("US equity trading days per year, 1996-2025", color="#e0e0e0", pad=12)
ax.tick_params(colors="#e0e0e0")
for s in ax.spines.values():
    s.set_color("#333333")
ax.grid(axis="y", color="#222222", lw=0.7)
ax.set_axisbelow(True)
leg = ax.legend(facecolor="#0a0a0a", edgecolor="#333333", labelcolor="#e0e0e0", loc="center", ncol=3, fontsize=9)
plt.tight_layout()
plt.savefig("trading-days-vs-calendar-days.png", dpi=150, facecolor="#0a0a0a")
