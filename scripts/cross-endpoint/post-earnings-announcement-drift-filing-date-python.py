# Full write-up: https://xfinlink.com/blog/post-earnings-announcement-drift-filing-date-python
"""
Post-earnings announcement drift measured from the SEC filing date.

Compares cumulative abnormal return by standardised-unexpected-earnings bucket
when the event clock starts at the filing date versus the fiscal period end.
Built from SEC EDGAR public filings and market data.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "JPM", "JNJ", "V",
           "PG", "XOM", "HD", "CVX", "MRK", "ABBV", "PEP", "KO", "WMT", "COST",
           "MCD", "CSCO", "ADBE", "CRM", "TXN", "QCOM", "AMD", "INTC", "NKE",
           "UNH", "LIN", "HON", "CAT", "LMT", "BA", "GE", "UPS", "MMM", "T",
           "VZ", "DIS"]
HORIZON = 60          # trading days after the event
MIN_HISTORY = 8       # prior year-over-year changes needed to standardise
MAX_LAG = 120         # plausible statutory filing window, days after period end
EVENT_START = "2018-01-01"
SLUG = "post-earnings-announcement-drift-filing-date-python"

# ----------------------------------------------------------------- fundamentals
fun = xfl.fundamentals(TICKERS, period_type="quarterly", start="2013-01-01",
                       fields=["net_income", "filing_date", "period_end"])
fun = fun.drop_duplicates(subset=["ticker", "fiscal_year", "fiscal_period"],
                          keep="first")
fun = fun.dropna(subset=["net_income", "filing_date", "period_end"])
fun["lag"] = (fun["filing_date"] - fun["period_end"]).dt.days

# filing_date is the date of the filing a figure was sourced from. For a restated
# quarter that can be a later comparative filing rather than the original
# announcement, so keep only quarters inside the statutory reporting window.
lag_all = fun["lag"].copy()
fun = fun[(fun["lag"] > 0) & (fun["lag"] <= MAX_LAG)].copy()

# ------------------------------------------------------- unexpected earnings
# Seasonal random walk: the expectation for this quarter is the same quarter one
# year ago. No analyst estimates are used anywhere in this construction.
prev = fun[["ticker", "fiscal_year", "fiscal_period", "net_income"]].copy()
prev["fiscal_year"] += 1
prev = prev.rename(columns={"net_income": "net_income_yoy"})
ev = fun.merge(prev, on=["ticker", "fiscal_year", "fiscal_period"], how="inner")
ev["ue"] = ev["net_income"] - ev["net_income_yoy"]
ev = ev.sort_values(["ticker", "period_end"])

# Standardise by the company's own past dispersion in year-over-year changes,
# shifted one quarter so nothing from the current event enters its own scale.
g = ev.groupby("ticker")["ue"]
ev["scale"] = g.transform(lambda s: s.shift(1).expanding(MIN_HISTORY).std())
ev = ev.dropna(subset=["scale"])
ev = ev[ev["scale"] > 0]
ev["sue"] = ev["ue"] / ev["scale"]

# ------------------------------------------------------------------- prices
universe = sorted(ev["ticker"].unique())
series = {}
for t in universe + ["SPY"]:
    d = xfl.prices(t, start="2017-06-01", fields=["return_daily"])
    series[t] = (d.dropna(subset=["return_daily"])
                 .set_index("date")["return_daily"])

# SPY defines the trading calendar; abnormal return is the simple excess over it
cal = np.array(sorted(series["SPY"].index.unique()))
spy = series["SPY"].reindex(cal)
excess = {t: (series[t].reindex(cal) - spy).to_numpy() for t in universe}

ev = ev[ev["filing_date"] >= pd.Timestamp(EVENT_START)]


def car_path(ticker, event_date):
    """Cumulative abnormal return over HORIZON days, starting the day after."""
    i = int(np.searchsorted(cal, np.datetime64(event_date), side="right"))
    if i + HORIZON > len(cal):
        return None
    seg = excess[ticker][i:i + HORIZON]
    if np.isnan(seg).any():
        return None
    return np.cumsum(seg)


def pre_filing_car(ticker, event_period_end, event_filing_date):
    """Excess return the period-end clock banks before the filing clock starts."""
    i = int(np.searchsorted(cal, np.datetime64(event_period_end), side="right"))
    j = int(np.searchsorted(cal, np.datetime64(event_filing_date), side="right"))
    return float(np.nansum(excess[ticker][i:j + 1]))


rows, paths = [], {"filing_date": {}, "period_end": {}}
for row in ev.itertuples():
    a = car_path(row.ticker, row.filing_date)
    b = car_path(row.ticker, row.period_end)
    if a is None or b is None:
        continue
    rows.append({"ticker": row.ticker, "period_end": row.period_end,
                 "filing_date": row.filing_date, "lag": row.lag,
                 "fiscal_period": row.fiscal_period, "sue": row.sue,
                 "car_filing": a[-1], "car_period": b[-1],
                 "car_pre": pre_filing_car(row.ticker, row.period_end,
                                           row.filing_date)})
    paths["filing_date"][len(rows) - 1] = a
    paths["period_end"][len(rows) - 1] = b

res = pd.DataFrame(rows)
res["bucket"] = pd.qcut(res["sue"], 5, labels=["Q1 low", "Q2", "Q3", "Q4",
                                               "Q5 high"])

# ------------------------------------------------------------------- output
print("Post-earnings announcement drift | seasonal random walk SUE | "
      f"{HORIZON} trading days")
print(f"universe {len(universe)} large caps | events {len(res)} | "
      f"{res['filing_date'].min().date()} to {res['filing_date'].max().date()}")
print(f"filing lag after period end: median {res['lag'].median():.0f}d  "
      f"p5 {res['lag'].quantile(0.05):.0f}d  p95 {res['lag'].quantile(0.95):.0f}d")
print(f"quarters kept by the 1-{MAX_LAG}d statutory window screen: "
      f"{int(((lag_all > 0) & (lag_all <= MAX_LAG)).sum())} of {len(lag_all)}")
print()

tab = res.groupby("bucket", observed=True).agg(
    events=("sue", "size"),
    mean_sue=("sue", "mean"),
    car_filing=("car_filing", "mean"),
    car_period=("car_period", "mean"),
    car_pre=("car_pre", "mean"))

print(f"{'SUE bucket':<11}{'events':>7}{'mean SUE':>10}"
      f"{'CAR from filing':>17}{'CAR from period end':>21}"
      f"{'pre-filing leg':>16}")
for b, r in tab.iterrows():
    print(f"{b:<11}{int(r['events']):>7}{r['mean_sue']:>10.2f}"
          f"{r['car_filing'] * 100:>16.2f}%{r['car_period'] * 100:>20.2f}%"
          f"{r['car_pre'] * 100:>15.2f}%")

hi, lo = tab.index[-1], tab.index[0]
sf = tab.loc[hi, "car_filing"] - tab.loc[lo, "car_filing"]
sp = tab.loc[hi, "car_period"] - tab.loc[lo, "car_period"]
spre = tab.loc[hi, "car_pre"] - tab.loc[lo, "car_pre"]
print()
print(f"top-minus-bottom spread from filing date:  {sf * 100:6.2f}%")
print(f"top-minus-bottom spread from period end:   {sp * 100:6.2f}%")
print(f"inflation from timing off period end:      {(sp - sf) * 100:6.2f}%"
      f"  ({sp / sf:.2f}x)")
print(f"top-minus-bottom spread, pre-filing leg:   {spre * 100:6.2f}%"
      f"  (the announcement reaction the period-end clock swallows)")

top = res.loc[res["bucket"] == hi, "car_filing"]
bot = res.loc[res["bucket"] == lo, "car_filing"]
tf, pf = ttest_ind(top, bot, equal_var=False)
tp, pp = ttest_ind(res.loc[res["bucket"] == hi, "car_period"],
                   res.loc[res["bucket"] == lo, "car_period"], equal_var=False)
print()
print(f"Welch test on the top-minus-bottom spread")
print(f"  from filing date:  t = {tf:5.2f}   p = {pf:.3f}")
print(f"  from period end:   t = {tp:5.2f}   p = {pp:.3f}")
print()
counts = res["fiscal_period"].value_counts().sort_index()
print("events by fiscal quarter: "
      + "  ".join(f"{k} {int(v)}" for k, v in counts.items()))
print(f"CAR range across all events: {res['car_filing'].min() * 100:.1f}% "
      f"to {res['car_filing'].max() * 100:.1f}%")
print(f"NaN check | sue {res['sue'].isna().sum()}  "
      f"car_filing {res['car_filing'].isna().sum()}  "
      f"car_period {res['car_period'].isna().sum()}")

# -------------------------------------------------------------------- chart
plt.rcParams.update({"figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
                     "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
                     "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
                     "axes.edgecolor": "#3a3a3a", "font.size": 10})
colours = {"Q1 low": "#ef4444", "Q2": "#f59e0b", "Q3": "#6b7280",
           "Q4": "#93c5fd", "Q5 high": "#3b82f6"}
fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
days = np.arange(1, HORIZON + 1)
titles = {"filing_date": "Clock starts at the filing date",
          "period_end": "Clock starts at the fiscal period end"}

# where the announcement lands inside the period-end window, in trading days
offset = int(np.median([
    np.searchsorted(cal, np.datetime64(f), side="right")
    - np.searchsorted(cal, np.datetime64(p), side="right")
    for p, f in zip(res["period_end"], res["filing_date"])]))

for ax, key in zip(axes, ["filing_date", "period_end"]):
    for b in tab.index:
        idx = res.index[res["bucket"] == b]
        mat = np.vstack([paths[key][i] for i in idx])
        ax.plot(days, mat.mean(axis=0) * 100, color=colours[b],
                lw=2.4 if b in (hi, lo) else 1.6, label=b)
    ax.axhline(0, color="#3a3a3a", lw=0.8)
    ax.set_title(titles[key], fontsize=10.5, color="#e0e0e0")
    ax.set_xlabel("Trading days after the event")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

axes[1].axvline(offset, color="#e0e0e0", lw=0.9, ls="--", alpha=0.6)
axes[1].annotate(f"median filing date (day {offset})", xy=(offset, 2.15),
                 xytext=(offset + 2.0, 2.15), fontsize=8.5, color="#b0b0b0",
                 va="center")

axes[0].set_ylabel("Average return in excess of SPY (%)")
axes[0].legend(frameon=False, fontsize=9, labelcolor="#e0e0e0",
               title="Earnings surprise", title_fontsize=9)
axes[0].get_legend().get_title().set_color("#e0e0e0")
fig.suptitle("Post-earnings announcement drift shrinks when measured from the "
             "filing date", fontsize=12, color="#e0e0e0")
plt.tight_layout()
plt.savefig(f"/home/user/xfinlink/worker/src/site/blog-images/{SLUG}.png",
            dpi=150, facecolor="#0a0a0a")
