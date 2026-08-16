# Full write-up: https://xfinlink.com/blog/sector-correlations-calm-vs-stress-python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

sectors = ["XLK", "XLE", "XLF", "XLV", "XLP", "XLU", "XLI", "XLY", "XLB", "XLRE"]
tickers = ["SPY"] + sectors

px = xfl.prices(tickers, start="2016-01-01", end="2025-12-31",
                fields=["close", "return_daily"], max_rows=200000)

# wide matrix of daily returns, common dates only
rets = px.pivot_table(index="date", columns="ticker", values="return_daily").dropna()

# regime: SPY 20-day realized volatility, top third = stress
spy_vol = rets["SPY"].rolling(20).std() * np.sqrt(252)
cut = spy_vol.quantile(2 / 3)
regime = pd.Series(np.where(spy_vol >= cut, "stress", "calm"), index=rets.index)
regime = regime[spy_vol.notna()]

sec = rets[sectors].loc[regime.index]
calm = sec[regime == "calm"]
strs = sec[regime == "stress"]


def avg_pairwise_corr(frame):
    c = frame.corr().values
    iu = np.triu_indices_from(c, k=1)
    return c[iu].mean()


def effective_n(frame):
    ev = np.linalg.eigvalsh(frame.corr().values)
    ev = ev[ev > 0]
    p = ev / ev.sum()
    return np.exp(-(p * np.log(p)).sum())


print("SPY + 10 sector SPDRs, daily returns "
      f"{sec.index.min().date()} to {sec.index.max().date()}")
print(f"{len(sec)} trading days with a defined regime "
      f"({(regime=='calm').sum()} calm, {(regime=='stress').sum()} stress)")
print(f"stress = SPY 20-day annualised volatility in the top third (>= {cut*100:.1f}%)")
print()
print(f"average pairwise sector correlation, calm   : {avg_pairwise_corr(calm):.3f}")
print(f"average pairwise sector correlation, stress : {avg_pairwise_corr(strs):.3f}")
print()

rows = []
for s in sectors:
    others = [x for x in sectors if x != s]
    c_calm = calm[[s] + others].corr().loc[s, others].mean()
    c_str = strs[[s] + others].corr().loc[s, others].mean()
    rows.append((s, c_calm, c_str, c_str - c_calm))

tab = pd.DataFrame(rows, columns=["sector", "calm", "stress", "rise"]).sort_values("rise")
print("average correlation of each sector to the other nine")
print("sector    calm   stress   rise")
for _, r in tab.iterrows():
    print(f"{r['sector']:5s}   {r['calm']:.3f}   {r['stress']:.3f}   {r['rise']:+.3f}")

print()
print(f"effective number of independent sectors, calm   : {effective_n(calm):.2f} of 10")
print(f"effective number of independent sectors, stress : {effective_n(strs):.2f} of 10")

# chart: dumbbell of each sector's average correlation, calm vs stress
BG, FG, ACC, STRESS = "#0a0a0a", "#e0e0e0", "#3b82f6", "#ef4444"
plot = tab.sort_values("calm")
fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
y = np.arange(len(plot))
ax.hlines(y, plot["calm"], plot["stress"], color="#3f3f46", lw=2, zorder=1)
ax.scatter(plot["calm"], y, color=ACC, s=70, zorder=2, label="Calm markets")
ax.scatter(plot["stress"], y, color=STRESS, s=70, zorder=2, label="Stressed markets")
ax.set_yticks(y)
ax.set_yticklabels(plot["sector"], color=FG)
ax.set_xlabel("Average correlation to the other nine sectors", color=FG)
ax.set_title("Sector correlations rise across the board when markets are stressed",
             color=FG, fontsize=13)
ax.set_xlim(0.15, 0.90)
for spine in ax.spines.values():
    spine.set_color("#3f3f46")
ax.tick_params(colors=FG)
ax.legend(facecolor=BG, edgecolor="#3f3f46", labelcolor=FG, loc="lower right")
plt.tight_layout()
plt.savefig("sector-correlations-calm-vs-stress-python.png", dpi=150, facecolor=BG)
