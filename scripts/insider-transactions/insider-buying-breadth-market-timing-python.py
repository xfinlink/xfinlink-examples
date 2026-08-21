# Full write-up: https://xfinlink.com/blog/insider-buying-breadth-market-timing-python
"""
Do corporate insiders time the market?

Builds a monthly breadth series from Form 4 open-market purchases across the
point-in-time S&P 500 roster, then asks whether that breadth says anything
about the index's own forward return.

Signal months run January 2010 to August 2025, the last month with a complete
twelve-month forward window. Prices run to 20 August 2026.
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

FIRST_YEAR, LAST_YEAR = 2010, 2025
PRICE_END = "2026-08-20"
SLUG = "insider-buying-breadth-market-timing-python"

# ---------------------------------------------------------------- data ------

# Point-in-time roster for each year, keyed on entity id rather than ticker so
# that a symbol later reassigned to another company cannot enter the sample.
frames, roster_size = [], {}
for year in range(FIRST_YEAR, LAST_YEAR + 1):
    roster = xfl.index("sp500", as_of=f"{year}-01-01")
    ids = [int(e) for e in roster["entity_id"].dropna()]
    roster_size[year] = len(ids)
    for i in range(0, len(ids), 100):
        batch = xfl.insiders(
            entity_id=ids[i:i + 100],
            start=f"{year}-01-01", end=f"{year}-12-31",
            transaction_type=["open_market_buy", "open_market_sell"],
            fields=["entity_id", "ticker", "transaction_date", "transaction_type"],
            max_rows=200_000,
        )
        if len(batch):
            frames.append(batch)

trades = pd.concat(frames, ignore_index=True)
trades["transaction_date"] = pd.to_datetime(trades["transaction_date"]).dt.tz_localize(None)
trades["month"] = trades["transaction_date"].dt.to_period("M")

# Breadth: share of that year's roster with at least one open-market trade.
counts = (trades.groupby(["month", "transaction_type"])["entity_id"]
          .nunique().unstack(fill_value=0))
counts["roster"] = [roster_size[m.year] for m in counts.index]
counts["buy_breadth"] = 100 * counts["open_market_buy"] / counts["roster"]
counts["sell_breadth"] = 100 * counts["open_market_sell"] / counts["roster"]

# Index side: SPY total return, compounded to calendar months.
spy = xfl.prices("SPY", start=f"{FIRST_YEAR - 1}-12-01", end=PRICE_END,
                 fields=["close", "return_daily"])
spy["month"] = spy["date"].dt.to_period("M")
monthly_ret = spy.groupby("month")["return_daily"].apply(lambda s: (1 + s).prod() - 1)
month_close = spy.groupby("month")["close"].last()

# ------------------------------------------------------------ forward -------

panel = counts.join(monthly_ret.rename("spy_ret"), how="right").sort_index()
log_ret = np.log1p(panel["spy_ret"])
for h in (3, 6, 12):
    fwd = [np.expm1(log_ret.iloc[i + 1:i + 1 + h].sum()) if i + h < len(panel) else np.nan
           for i in range(len(panel))]
    panel[f"fwd{h}"] = fwd
panel = panel.loc[f"{FIRST_YEAR}-01":f"{LAST_YEAR}-12"].dropna(subset=["buy_breadth"])

signals = panel.dropna(subset=["fwd12"])
quintile = pd.qcut(signals["buy_breadth"], 5, labels=[1, 2, 3, 4, 5])
table = signals.groupby(quintile, observed=True).agg(
    months=("buy_breadth", "size"),
    lo=("buy_breadth", "min"), hi=("buy_breadth", "max"),
    f3=("fwd3", "mean"), f6=("fwd6", "mean"), f12=("fwd12", "mean"),
    hit=("fwd12", lambda s: 100 * (s > 0).mean()))


def hac(y_name, x_name, lags):
    d = signals[[x_name, y_name]].dropna()
    fit = sm.OLS(d[y_name], sm.add_constant(d[x_name])).fit(
        cov_type="HAC", cov_kwds={"maxlags": lags})
    return fit.params[x_name], fit.tvalues[x_name], fit.pvalues[x_name], fit.rsquared, int(fit.nobs)


# ------------------------------------------------------------- output -------

print("=== S&P 500 insider buying breadth vs the index's own forward return ===")
print(f"Signal months: {signals.index.min()} to {signals.index.max()}  (n={len(signals)})")
print(f"Open-market trades read: {len(trades):,}  "
      f"({int(counts['open_market_buy'].sum()):,} buyer-months, "
      f"{int(counts['open_market_sell'].sum()):,} seller-months)")
print(f"Buy breadth: mean {signals['buy_breadth'].mean():.2f}%  "
      f"median {signals['buy_breadth'].median():.2f}%  "
      f"range {signals['buy_breadth'].min():.2f}% to {signals['buy_breadth'].max():.2f}%")
print(f"Sell breadth: mean {signals['sell_breadth'].mean():.2f}%")
print(f"Unconditional forward 12-month SPY return: {signals['fwd12'].mean() * 100:.2f}%\n")

print("Forward SPY total return by buy-breadth quintile")
print(f"{'quintile':>9} {'months':>7} {'breadth range':>16} {'fwd 3m':>8} {'fwd 6m':>8} "
      f"{'fwd 12m':>9} {'12m up':>8}")
for q, r in table.iterrows():
    print(f"{int(q):>9} {int(r['months']):>7} {r['lo']:>7.2f}% -{r['hi']:>6.2f}% "
          f"{r['f3'] * 100:>7.2f}% {r['f6'] * 100:>7.2f}% {r['f12'] * 100:>8.2f}% "
          f"{r['hit']:>7.1f}%")
spread = (table.loc[5, "f12"] - table.loc[1, "f12"]) * 100
print(f"{'':>9} top fifth minus bottom fifth, 12 months: {spread:+.2f} points\n")

print("Newey-West regression of forward return on buy breadth")
for h in (3, 6, 12):
    b, t, p, r2, n = hac(f"fwd{h}", "buy_breadth", h)
    print(f"  horizon {h:>2}m  n={n}  slope={b:+.5f} per point of breadth  "
          f"t={t:+.2f}  p={p:.4f}  R2={r2:.3f}")
b, t, p, r2, n = hac("fwd12", "sell_breadth", 12)
print(f"  sell breadth, 12m  n={n}  slope={b:+.5f}  t={t:+.2f}  p={p:.4f}  R2={r2:.3f}\n")

ex2020 = signals[signals.index.year != 2020]
q_ex = pd.qcut(ex2020["buy_breadth"], 5, labels=[1, 2, 3, 4, 5])
m_ex = ex2020.groupby(q_ex, observed=True)["fwd12"].mean()
print(f"Excluding 2020 signal months: bottom fifth {m_ex.loc[1] * 100:.2f}%, "
      f"top fifth {m_ex.loc[5] * 100:.2f}%, spread {(m_ex.loc[5] - m_ex.loc[1]) * 100:+.2f} points")
print("Five busiest months for insider buying: "
      + ", ".join(f"{m} ({v:.1f}%)" for m, v in signals["buy_breadth"].nlargest(5).items()))

# -------------------------------------------------------------- chart -------

plt.rcParams.update({
    "figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
    "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
    "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
    "axes.edgecolor": "#3a3a3a", "font.size": 10,
})
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))

x = panel.index.to_timestamp()
ax1.plot(x, panel["buy_breadth"], color="#3b82f6", linewidth=1.3)
ax1.set_ylabel("S&P 500 companies with an insider\nbuying that month (% of index)",
               color="#3b82f6")
ax1.tick_params(axis="y", colors="#3b82f6")
ax1.set_ylim(0, 24)
ax1.set_title("Insider buying breadth and the S&P 500, 2010-2025", color="#fafafa", pad=10)

twin = ax1.twinx()
twin.plot(x, month_close.loc[panel.index].values, color="#9ca3af", linewidth=1.2)
twin.set_yscale("log")
twin.set_yticks([100, 150, 200, 300, 400, 600])
twin.set_yticklabels(["100", "150", "200", "300", "400", "600"])
twin.set_ylabel("SPY close, dollars (log scale)", color="#9ca3af")
twin.tick_params(colors="#9ca3af")
twin.spines["right"].set_color("#3a3a3a")
for s in ("top",):
    ax1.spines[s].set_visible(False)
    twin.spines[s].set_visible(False)

mean12 = signals["fwd12"].mean() * 100
bars = ax2.bar([1, 2, 3, 4, 5], table["f12"] * 100, color="#3b82f6", width=0.6, zorder=2)
bars[0].set_color("#1e3a5f")
ax2.axhline(mean12, color="#9ca3af", linestyle="--", linewidth=1, zorder=1)
ax2.text(5.42, mean12 + 0.55, f"all months {mean12:.1f}%", color="#9ca3af", fontsize=9)
ax2.set_xticks([1, 2, 3, 4, 5])
ax2.set_xticklabels(["quietest\nfifth", "2nd", "3rd", "4th", "busiest\nfifth"])
ax2.set_ylabel("Average SPY total return\nover the next 12 months (%)")
ax2.set_title("Forward return by insider buying breadth", color="#fafafa", pad=10)
ax2.set_xlim(0.45, 6.6)
ax2.set_ylim(0, 19.5)
for s in ("top", "right"):
    ax2.spines[s].set_visible(False)
for xi, v in zip([1, 2, 3, 4, 5], table["f12"] * 100):
    ax2.text(xi, v + 0.5, f"{v:.1f}%", ha="center", color="#e0e0e0", fontsize=9)

plt.tight_layout(h_pad=2.6)
plt.savefig(f"{SLUG}.png", dpi=150, facecolor="#0a0a0a")
print(f"\nChart written to {SLUG}.png")
