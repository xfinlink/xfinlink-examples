# Full write-up: https://xfinlink.com/blog/max-drawdown-sampling-frequency-python
#
# How much of a drawdown does month-end data hide?
# Maximum drawdown of 101 S&P 500 companies and SPY over 2020-2024,
# measured on daily, weekly and month-end price grids.

import matplotlib.pyplot as plt
import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

START, END = "2020-01-01", "2024-12-31"

roster = xfl.index("sp500", as_of="2019-12-31")
ids = sorted(roster["entity_id"].tolist())[::5]

frames = [xfl.prices(entity_id=ids[i:i + 20], start=START, end=END,
                     fields=["adj_close"], max_rows=100000)
          for i in range(0, len(ids), 20)]
px = pd.concat(frames + [xfl.prices("SPY", start=START, end=END,
                                    fields=["adj_close"])], ignore_index=True)


def drawdown(s):
    return s / s.cummax() - 1


def episode(s):
    d = drawdown(s)
    trough = d.idxmin()
    return s[:trough].idxmax(), trough, d.min()


rows, series = [], {}
for eid, g in px.groupby("entity_id"):
    s = g.set_index("date")["adj_close"].sort_index().dropna()
    if len(s) < 500:
        continue
    week, month = s.resample("W-FRI").last().dropna(), s.resample("ME").last().dropna()
    series[g["ticker"].iloc[-1]] = (s, month)
    rows.append({"ticker": g["ticker"].iloc[-1], "days": len(s),
                 "daily": drawdown(s).min(), "weekly": drawdown(week).min(),
                 "monthly": drawdown(month).min()})

dd = pd.DataFrame(rows)
dd["hidden_pp"] = (dd["monthly"] - dd["daily"]) * 100
dd["captured"] = dd["monthly"] / dd["daily"]
stocks = dd[dd["ticker"] != "SPY"]
spy = dd[dd["ticker"] == "SPY"].iloc[0]

print(f"S&P 500 roster of 2019-12-31, every 5th name: {len(stocks)} companies, 2020-2024")
print(f"full five-year price history: {(stocks['days'] == 1258).sum()}")
print()
print("                     daily   weekly   month-end")
print("median max drawdown %6.1f%% %7.1f%% %10.1f%%" % (
    stocks["daily"].median() * 100, stocks["weekly"].median() * 100,
    stocks["monthly"].median() * 100))
print("median depth captured        %7.3f %10.3f" % (
    (stocks["weekly"] / stocks["daily"]).median(), stocks["captured"].median()))
print()
print("month-end understatement (pp): median %.1f  90th pct %.1f  max %.1f" % (
    stocks["hidden_pp"].median(), stocks["hidden_pp"].quantile(0.9),
    stocks["hidden_pp"].max()))
print("names hiding more than 10pp: %d of %d" % (
    (stocks["hidden_pp"] > 10).sum(), len(stocks)))
print()
s, month = series["SPY"]
p_d, t_d, v_d = episode(s)
p_m, t_m, v_m = episode(month)
covid = month.loc["2020-03-31"] / month.loc["2020-01-31"] - 1
print("SPY: daily %.1f%%  weekly %.1f%%  month-end %.1f%%" % (
    spy["daily"] * 100, spy["weekly"] * 100, spy["monthly"] * 100))
print("  worst daily episode      %s to %s  %.1f%%" % (
    p_d.date(), t_d.date(), v_d * 100))
print("  worst month-end episode  %s to %s  %.1f%%" % (
    p_m.date(), t_m.date(), v_m * 100))
print("  2020 crash on the month-end grid                  %.1f%%" % (covid * 100))
print()
print("widest month-end gaps")
for _, r in stocks.nlargest(5, "hidden_pp").iterrows():
    print("  %-5s daily %.1f%%  month-end %.1f%%  hidden %.1fpp" % (
        r["ticker"], r["daily"] * 100, r["monthly"] * 100, r["hidden_pp"]))

# ── Chart ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
    "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
    "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
    "axes.edgecolor": "#3a3a3a", "font.size": 10,
})
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))

d_daily, d_month = drawdown(s) * 100, drawdown(month) * 100
ax1.fill_between(d_daily.index, d_daily.values, 0, color="#3b82f6", alpha=0.35, linewidth=0)
ax1.plot(d_daily.index, d_daily.values, color="#3b82f6", linewidth=1.0, label="Daily prices")
ax1.step(d_month.index, d_month.values, where="post", color="#f59e0b",
         linewidth=1.6, label="Month-end prices only")
ax1.set_title("SPY: how deep the drawdown looks on each price grid", color="#e0e0e0")
ax1.set_ylabel("Fall from prior peak (%)")
ax1.legend(frameon=False, loc="lower right")
for sp in ax1.spines.values():
    sp.set_visible(False)
ax1.spines["left"].set_visible(True)
ax1.spines["bottom"].set_visible(True)

x, y = stocks["daily"] * 100, stocks["monthly"] * 100
ax2.scatter(x, y, s=26, color="#3b82f6", alpha=0.75, edgecolor="none")
lo = min(x.min(), y.min()) - 3
ax2.plot([lo, 0], [lo, 0], color="#e0e0e0", linewidth=1.0, linestyle="--", alpha=0.6)
ax2.text(-92, -18, "the vertical distance to the dashed line\nis the drawdown month-end data misses",
         color="#e0e0e0", fontsize=9)
ax2.set_title("101 S&P 500 companies, worst drawdown 2020-2024", color="#e0e0e0")
ax2.set_xlabel("Maximum drawdown on daily prices (%)")
ax2.set_ylabel("On month-end prices (%)")
for sp in ax2.spines.values():
    sp.set_visible(False)
ax2.spines["left"].set_visible(True)
ax2.spines["bottom"].set_visible(True)

plt.tight_layout()
plt.savefig("max-drawdown-sampling-frequency-python.png", dpi=150, facecolor="#0a0a0a")
