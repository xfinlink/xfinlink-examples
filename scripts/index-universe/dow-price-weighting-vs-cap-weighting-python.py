# Full write-up: https://xfinlink.com/blog/dow-price-weighting-vs-cap-weighting-python
"""How far does the Dow's price weighting pull it away from capitalisation weighting?

The same thirty companies, weighted two ways. On the snapshot date each member's
share of the index is computed from the raw as-traded close (the Dow's own rule)
and from market capitalisation. Two daily return series are then rebuilt from
that one basket over five years, differing only in the weighting rule.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

START = "2021-05-28"
SNAP = "2026-05-29"
CHART = "dow-price-weighting-vs-cap-weighting-python.png"

members = xfl.index("djia", as_of=SNAP)
tickers = sorted(members["ticker"].tolist())
names = dict(zip(members["ticker"], members["entity_name"]))

px = xfl.prices(tickers, start=START, end=SNAP,
                fields=["close", "adj_close"], max_rows=250000)
px["date"] = pd.to_datetime(px["date"])
close = px.pivot(index="date", columns="ticker", values="close").sort_index().ffill()
adj = px.pivot(index="date", columns="ticker", values="adj_close").sort_index().ffill()


def market_cap(on):
    """Market capitalisation for every member on one date, in $m."""
    df = xfl.metrics(tickers, period_type="daily", fields=["market_cap"],
                     start=on, end=on)
    return df.set_index("ticker")["market_cap"].reindex(close.columns)


cap_start, cap_snap = market_cap(START), market_cap(SNAP)

# --- the two weighting rules on the snapshot date --------------------------
snap = pd.DataFrame({"price": close.iloc[-1], "mcap": cap_snap})
snap["price_w"] = 100 * snap["price"] / snap["price"].sum()
snap["cap_w"] = 100 * snap["mcap"] / snap["mcap"].sum()
snap["gap"] = snap["price_w"] - snap["cap_w"]
snap["ratio"] = snap["price_w"] / snap["cap_w"]
snap = snap.sort_values("price_w", ascending=False)
reallocation = snap["gap"].abs().sum() / 2

# --- five years of returns from one basket, two weighting rules ------------
ret = adj.pct_change().iloc[1:]

# price weights are the previous close restated onto today's share basis, so a
# split cuts a member's weight on the day it happens, exactly as the divisor does
w_price = close.reindex(ret.index).div(1 + ret)

# capitalisation weights start from market cap and are carried by price
growth = (1 + ret).cumprod()
growth.loc[close.index[0]] = 1.0
w_cap = growth.sort_index().mul(cap_start, axis=1).shift(1).reindex(ret.index)


def index_return(weights):
    """Previous day's weights applied to today's returns, renormalised."""
    w = weights.where(ret.notna())
    w = w.div(w.sum(axis=1), axis=0)
    return (w * ret).sum(axis=1), w


r_price, wp = index_return(w_price)
r_cap, wc = index_return(w_cap)


def stats(r):
    total = (1 + r).prod() - 1
    curve = (1 + r).cumprod()
    return {"total": 100 * total,
            "cagr": 100 * ((1 + total) ** (252 / len(r)) - 1),
            "vol": 100 * r.std() * np.sqrt(252),
            "maxdd": 100 * (curve / curve.cummax() - 1).min()}


sp, sc = stats(r_price), stats(r_cap)
diff = r_price - r_cap
tracking_error = 100 * diff.std() * np.sqrt(252)

top5 = snap.index[:5].tolist()
contrib_p = 100 * (wp[top5] * ret[top5]).sum().sum()
contrib_c = 100 * (wc[top5] * ret[top5]).sum().sum()
sum_p, sum_c = 100 * r_price.sum(), 100 * r_cap.sum()

# --- output ---------------------------------------------------------------
print(f"Dow Jones Industrial Average, {len(snap)} members at {SNAP}")
print(f"{'ticker':<7}{'company':<30}{'close $':>9}{'price wt %':>12}"
      f"{'cap wt %':>10}{'gap pp':>9}{'ratio':>7}")
for t, r in snap.iterrows():
    print(f"{t:<7}{names[t][:29]:<30}{r['price']:>9,.2f}{r['price_w']:>12.2f}"
          f"{r['cap_w']:>10.2f}{r['gap']:>+9.2f}{r['ratio']:>7.2f}")

print(f"\nWeight that would have to change hands to turn price weighting into "
      f"cap weighting: {reallocation:.1f}%")
print(f"  heaviest by price  {snap.index[0]} {snap['price_w'].iloc[0]:.2f}% "
      f"(cap weight {snap['cap_w'].iloc[0]:.2f}%)")
print(f"  heaviest by cap    {snap['cap_w'].idxmax()} {snap['cap_w'].max():.2f}% "
      f"(price weight {snap.loc[snap['cap_w'].idxmax(), 'price_w']:.2f}%)")

sector = px.drop_duplicates("ticker").set_index("ticker")["gics_sector"]
by_sector = (snap.join(sector).groupby("gics_sector")[["price_w", "cap_w"]].sum()
             .sort_values("price_w", ascending=False))
print("\nSector weight under each rule")
print(f"  {'':<24}{'price %':>9}{'cap %':>9}")
for s, r in by_sector.iterrows():
    print(f"  {s:<24}{r['price_w']:>9.2f}{r['cap_w']:>9.2f}")

print(f"\nSame 30 companies, two weighting rules, {ret.index[0].date()} to "
      f"{ret.index[-1].date()} ({len(ret)} sessions)")
print(f"{'':<24}{'price-weighted':>16}{'cap-weighted':>15}{'difference':>13}")
for label, key in [("cumulative return %", "total"), ("annualised return %", "cagr"),
                   ("annualised volatility %", "vol"), ("maximum drawdown %", "maxdd")]:
    print(f"{label:<24}{sp[key]:>16.2f}{sc[key]:>15.2f}{sp[key] - sc[key]:>+13.2f}")
print(f"{'daily correlation':<24}{r_price.corr(r_cap):>16.4f}")
print(f"{'tracking error % p.a.':<24}{tracking_error:>16.2f}")

print(f"\nWhat the five highest-priced members contributed ({', '.join(top5)})")
print(f"  under price weighting  {contrib_p:>7.1f} pp of {sum_p:.1f} pp "
      f"({100 * contrib_p / sum_p:.0f}% of the total)")
print(f"  under cap weighting    {contrib_c:>7.1f} pp of {sum_c:.1f} pp "
      f"({100 * contrib_c / sum_c:.0f}% of the total)")

# every share split in the window, and what it did to each weight
step = (close / adj).pipe(lambda f: f / f.shift(1)).stack()
splits = step[(step - 1).abs() > 0.01]
print("\nShare splits inside the window: weight before and on the day, then the "
      "member's contribution from that day to the end")
print(f"{'date':<12}{'':<6}{'split':>8}{'price wt before':>17}{'price wt on':>13}"
      f"{'cap wt on':>11}{'price contrib':>15}{'cap contrib':>13}")
for (day, t), f in splits.items():
    prev = ret.index[ret.index.get_loc(day) - 1]
    after = ret.index >= day
    print(f"{str(day.date()):<12}{t:<6}{1 / f:>7.0f}:1{100 * wp.loc[prev, t]:>16.2f}%"
          f"{100 * wp.loc[day, t]:>12.2f}%{100 * wc.loc[day, t]:>10.2f}%"
          f"{100 * (wp[t][after] * ret[t][after]).sum():>13.2f}pp"
          f"{100 * (wc[t][after] * ret[t][after]).sum():>11.2f}pp")

# --- chart ----------------------------------------------------------------
plt.rcParams.update({"figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
                     "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
                     "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
                     "axes.edgecolor": "#3a3a3a", "font.size": 9})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 7))

y = np.arange(len(snap))[::-1]
ax1.hlines(y, snap["cap_w"], snap["price_w"], color="#3a3a3a", lw=1.2, zorder=1)
ax1.scatter(snap["cap_w"], y, s=24, color="#9ca3af", zorder=2, label="Market-cap weight")
ax1.scatter(snap["price_w"], y, s=24, color="#3b82f6", zorder=3, label="Price weight")
ax1.set_yticks(y)
ax1.set_yticklabels(snap.index, fontsize=7.5)
ax1.set_ylim(-1, len(snap))
ax1.set_xlabel("Share of the index (%)")
ax1.set_title(f"Weight under each rule, {SNAP}", fontsize=10)
ax1.legend(frameon=False, fontsize=8, loc="lower right")
ax1.spines[["top", "right"]].set_visible(False)

ax2.plot((1 + r_price).cumprod(), color="#3b82f6", lw=1.4, label="Price-weighted")
ax2.plot((1 + r_cap).cumprod(), color="#f59e0b", lw=1.4, label="Market-cap-weighted")
ax2.set_ylabel("Growth of $1")
ax2.set_title("The same 30 companies over five years", fontsize=10)
ax2.legend(frameon=False, fontsize=8, loc="upper left")
ax2.spines[["top", "right"]].set_visible(False)
ax2.tick_params(axis="x", labelsize=8)

fig.suptitle("Price weighting against capitalisation weighting in the Dow",
             fontsize=12, color="#e0e0e0")
plt.tight_layout()
plt.savefig(CHART, dpi=150, facecolor="#0a0a0a")
