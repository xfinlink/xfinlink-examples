# Full write-up: https://xfinlink.com/blog/fomc-event-study-spy-python

import xfinlink as xfl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

xfl.api_key = "YOUR_API_KEY"  # free at https://xfinlink.com/signup

# -- FOMC announcement dates (2023-2026) --------------------------------------
fomc_dates = [
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14",
    "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
    "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-17",
    "2026-01-28", "2026-03-18", "2026-05-06",
]
fomc_dates = pd.to_datetime(fomc_dates)

# -- SPY daily returns from xfinlink -------------------------------------------
spy_df = xfl.prices("SPY", start="2022-12-01", fields=["close", "return_daily"])
spy_df["date"] = pd.to_datetime(spy_df["date"])
spy = spy_df.set_index("date").sort_index()

# -- Build event windows: [-5, +5] trading days around each FOMC date ---------
trading_days = spy.index
results = []
for fomc in fomc_dates:
    idx = trading_days.get_indexer([fomc], method="nearest")[0]
    if idx < 5 or idx + 5 >= len(trading_days):
        continue
    window = spy.iloc[idx - 5 : idx + 6]["return_daily"].values
    car = np.cumsum(window) * 100  # cumulative abnormal return in %
    results.append(car)

results = np.array(results)
n_events = len(results)
days = list(range(-5, 6))

# -- Print results -------------------------------------------------------------
print(f"Events: {n_events} FOMC meetings (2023-2026)\n")
print(f"{'Day':>4s}  {'Avg CAR':>8s}  {'SE':>7s}  {'t-stat':>7s}")
for i, d in enumerate(days):
    mean_car = results[:, i].mean()
    se = results[:, i].std() / np.sqrt(n_events)
    t = mean_car / se if se > 0 else 0
    print(f"{d:+4d}   {mean_car:+.3f}%  {se:.3f}%  {t:+.2f}")

# Summary lines
pre_drift = results[:, 5].mean() - results[:, 0].mean() if results.shape[1] > 5 else 0
# CAR at day -1 (index 4) is the pre-announcement drift
pre = results[:, 4].mean()
day0_reaction = results[:, 5].mean() - results[:, 4].mean()
post = results[:, -1].mean() - results[:, 5].mean()
print(f"\nPre-announcement drift (day -5 to -1): {pre:+.3f}%")
print(f"Announcement-day reaction (day 0):     {day0_reaction:+.3f}%")
print(f"Post-announcement drift (day +1 to +5): {post:+.3f}%")

# -- Chart: event study plot ---------------------------------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), gridspec_kw={"height_ratios": [2, 1]})

# Top panel: mean CAR with confidence band
mean_cars = [results[:, i].mean() for i in range(len(days))]
se_cars = [results[:, i].std() / np.sqrt(n_events) for i in range(len(days))]
upper = [m + 1.96 * s for m, s in zip(mean_cars, se_cars)]
lower = [m - 1.96 * s for m, s in zip(mean_cars, se_cars)]

ax1.plot(days, mean_cars, color="#2563eb", linewidth=2.5, marker="o", markersize=6, zorder=3)
ax1.fill_between(days, lower, upper, alpha=0.15, color="#2563eb", label="95% CI")
ax1.axhline(0, color="black", linewidth=0.5)
ax1.axvline(0, color="#dc2626", linewidth=1.5, linestyle="--", alpha=0.7, label="FOMC announcement")
ax1.set_ylabel("Cumulative Abnormal Return (%)")
ax1.set_title(
    f"Pre-FOMC Announcement Drift: Event Study ({n_events} meetings, 2023-2026)\n"
    f"Pre-drift: {pre:+.3f}% | Day 0: {day0_reaction:+.3f}% | Post-drift: {post:+.3f}%",
    fontsize=13,
    fontweight="bold",
)
ax1.set_xticks(days)
ax1.set_xticklabels([f"t{d:+d}" for d in days])
ax1.legend(loc="upper left")
ax1.grid(True, alpha=0.3)

# Bottom panel: t-statistics
t_stats = [mean_cars[i] / se_cars[i] if se_cars[i] > 0 else 0 for i in range(len(days))]
colors = ["#dc2626" if abs(t) >= 1.96 else "#2563eb" if abs(t) >= 1.5 else "#9ca3af" for t in t_stats]
ax2.bar(days, t_stats, color=colors, edgecolor="white", width=0.7)
ax2.axhline(1.96, color="#dc2626", linewidth=1, linestyle="--", alpha=0.5, label="p=0.05")
ax2.axhline(-1.96, color="#dc2626", linewidth=1, linestyle="--", alpha=0.5)
ax2.axhline(0, color="black", linewidth=0.5)
ax2.set_ylabel("t-statistic")
ax2.set_xlabel("Trading Days Relative to FOMC Announcement")
ax2.set_xticks(days)
ax2.set_xticklabels([f"t{d:+d}" for d in days])
ax2.legend(loc="upper left")
ax2.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("fomc-event-study-spy-python.png", dpi=150, bbox_inches="tight")
plt.show()
