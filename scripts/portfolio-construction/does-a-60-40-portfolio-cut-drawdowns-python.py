# Full write-up: https://xfinlink.com/blog/does-a-60-40-portfolio-cut-drawdowns-python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

px = xfl.prices(["SPY", "AGG"], start="2008-01-01", end="2025-12-31",
                fields=["return_daily"], max_rows=200000)
r = px.pivot_table(index="date", columns="ticker", values="return_daily").dropna()

# monthly-rebalanced 60/40: reset to fixed weights each month-end, let them drift within
w = pd.Series({"SPY": 0.60, "AGG": 0.40})
port = pd.Series(index=r.index, dtype=float)
for _, block in r.groupby(r.index.to_period("M")):
    val = ((1 + block).cumprod() * w).sum(axis=1)
    port.loc[block.index] = (val / val.shift(1).fillna(1.0)).values - 1
port = port.dropna()
eq = r["SPY"].loc[port.index]


def summarise(returns, label):
    cum = (1 + returns).cumprod()
    yrs = len(returns) / 252
    cagr = cum.iloc[-1] ** (1 / yrs) - 1
    vol = returns.std() * np.sqrt(252)
    dd = cum / cum.cummax() - 1
    worst12 = (cum / cum.shift(252) - 1).min()
    print(f"{label:14s}  CAGR {cagr*100:5.2f}%   vol {vol*100:5.2f}%   "
          f"max drawdown {dd.min()*100:6.1f}%   worst 12m {worst12*100:6.1f}%   "
          f"return/vol {cagr/vol:.2f}")
    return dd


print("SPY (100% equity) vs a monthly-rebalanced 60/40 SPY/AGG, total return")
print(f"daily data {port.index.min().date()} to {port.index.max().date()}, "
      f"{len(port)} trading days")
print()
dd_eq = summarise(eq, "100% equity")
dd_60 = summarise(port, "60/40")
print()
for yr in [2020, 2022]:
    m = port.index.year == yr
    e = (1 + eq[m]).prod() - 1
    p = (1 + port[m]).prod() - 1
    b = (1 + r["AGG"].loc[port.index][m]).prod() - 1
    print(f"{yr} total return  equity {e*100:6.1f}%   bonds {b*100:6.1f}%   60/40 {p*100:6.1f}%")

# chart: drawdown through time, equity vs 60/40
BG, FG, ACC, EQ = "#0a0a0a", "#e0e0e0", "#3b82f6", "#ef4444"
fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.fill_between(dd_eq.index, dd_eq * 100, 0, color=EQ, alpha=0.35)
ax.fill_between(dd_60.index, dd_60 * 100, 0, color=ACC, alpha=0.55)
ax.plot(dd_eq.index, dd_eq * 100, color=EQ, lw=1.0, label="100% equity (SPY)")
ax.plot(dd_60.index, dd_60 * 100, color=ACC, lw=1.0, label="60/40 SPY/AGG")
ax.set_ylabel("Drawdown from prior peak (%)", color=FG)
ax.set_title("The 60/40 halves the depth of a crash, but does not remove it",
             color=FG, fontsize=13)
ax.set_ylim(-55, 2)
for spine in ax.spines.values():
    spine.set_color("#3f3f46")
ax.tick_params(colors=FG)
ax.axhline(0, color="#3f3f46", lw=0.8)
ax.legend(facecolor=BG, edgecolor="#3f3f46", labelcolor=FG, loc="lower right")
plt.tight_layout()
plt.savefig("does-a-60-40-portfolio-cut-drawdowns-python.png", dpi=150, facecolor=BG)
