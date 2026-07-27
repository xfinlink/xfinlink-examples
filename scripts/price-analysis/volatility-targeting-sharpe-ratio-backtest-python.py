# Full write-up: https://xfinlink.com/blog/volatility-targeting-sharpe-ratio-backtest-python
import numpy as np
import pandas as pd
import xfinlink as xfl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

TICKERS = ["SPY", "IWM", "EFA", "EEM", "VNQ", "TLT", "GLD"]
START = "2006-11-01"        # warm-up so the first test day has a full window
TEST_START = "2007-01-01"
TARGET_VOL = 0.15           # annualised
LOOKBACK = 20               # trading days
MAX_LEVERAGE = 2.0
COST = 0.0001               # 1 basis point per unit of notional traded

# ── Data retrieval ───────────────────────────────────────────────────
px = pd.concat([xfl.prices(t, start=START, fields=["close", "return_daily"])
                for t in TICKERS], ignore_index=True)

# ── Helpers ──────────────────────────────────────────────────────────
def sharpe(r):
    return r.mean() / r.std() * np.sqrt(252)

def max_drawdown(r):
    curve = (1 + r).cumprod()
    return (curve / curve.cummax() - 1).min()

def cagr(r):
    return (1 + r).cumprod().iloc[-1] ** (252 / len(r)) - 1

def backtest(returns, lag):
    """Scale exposure by inverse realised volatility. lag=1 uses only
    information available before the trading day; lag=0 peeks at it."""
    realised = returns.rolling(LOOKBACK).std() * np.sqrt(252)
    weight = (TARGET_VOL / realised.shift(lag)).clip(upper=MAX_LEVERAGE)
    frame = pd.DataFrame({"r": returns, "w": weight}).dropna()
    frame = frame[frame.index >= TEST_START]
    turnover = frame["w"].diff().abs().fillna(0.0)
    net = frame["w"] * frame["r"] - COST * turnover
    return frame, net, turnover

rows, curves = [], {}

for t in TICKERS:
    sub = px[px["ticker"] == t].sort_values("date").set_index("date")
    r = sub["return_daily"].dropna()

    frame, net, turnover = backtest(r, lag=1)
    _, peek, _ = backtest(r, lag=0)          # look-ahead control
    bh = frame["r"]

    # constant rescaling to equal full-sample volatility, so that CAGR and
    # drawdown compare like with like
    matched = net * (bh.std() / net.std())

    rows.append({
        "ticker": t,
        "n": len(frame),
        "bh_sharpe": sharpe(bh),
        "vt_sharpe": sharpe(net),
        "peek_sharpe": sharpe(peek),
        "bh_vol": bh.std() * np.sqrt(252),
        "bh_cagr": cagr(bh),
        "vt_cagr": cagr(matched),
        "bh_mdd": max_drawdown(bh),
        "vt_mdd": max_drawdown(matched),
        "avg_exposure": frame["w"].mean(),
        "turnover": turnover.mean() * 252,
    })
    curves[t] = {"bh": (1 + bh).cumprod(), "vt": (1 + matched).cumprod()}

res = pd.DataFrame(rows).set_index("ticker")

print(f"Volatility targeting versus buy and hold, {TEST_START} to "
      f"{px['date'].max().date()}")
print(f"{LOOKBACK}-day realised volatility, {TARGET_VOL:.0%} annualised target, "
      f"leverage capped at {MAX_LEVERAGE:.1f}x, {COST * 1e4:.0f}bp per unit traded")
print("Sharpe ratio here is mean return over volatility, no risk-free deduction\n")

print(f"{'Ticker':<8}{'Days':>7}{'B&H Sharpe':>12}{'Targeted':>10}"
      f"{'Avg exposure':>14}{'Turnover':>10}")
print("-" * 61)
for t, row in res.iterrows():
    print(f"{t:<8}{row['n']:>7.0f}{row['bh_sharpe']:>12.2f}{row['vt_sharpe']:>10.2f}"
          f"{row['avg_exposure']:>14.2f}{row['turnover']:>9.1f}x")

print("\nVolatility-matched comparison (both series rescaled to equal volatility)")
print(f"{'Ticker':<8}{'B&H CAGR':>11}{'Targeted':>10}{'B&H MaxDD':>12}"
      f"{'Targeted':>10}{'Ann vol':>10}")
print("-" * 61)
for t, row in res.iterrows():
    print(f"{t:<8}{row['bh_cagr']:>10.1%}{row['vt_cagr']:>10.1%}"
          f"{row['bh_mdd']:>12.1%}{row['vt_mdd']:>10.1%}{row['bh_vol']:>10.1%}")

print("\nLook-ahead control: same rule, volatility measured through the "
      "trading day itself")
print(f"{'Ticker':<8}{'Correct lag':>13}{'Same-day peek':>15}{'Inflation':>11}")
print("-" * 47)
for t, row in res.iterrows():
    print(f"{t:<8}{row['vt_sharpe']:>13.2f}{row['peek_sharpe']:>15.2f}"
          f"{row['peek_sharpe'] - row['vt_sharpe']:>11.2f}")

print(f"\nMedian Sharpe: buy and hold {res['bh_sharpe'].median():.2f}, "
      f"targeted {res['vt_sharpe'].median():.2f}, "
      f"targeted with same-day peek {res['peek_sharpe'].median():.2f}")
print(f"Median max drawdown: buy and hold {res['bh_mdd'].median():.1%}, "
      f"targeted {res['vt_mdd'].median():.1%}")
print(f"Sharpe improved on {(res['vt_sharpe'] > res['bh_sharpe']).sum()} of "
      f"{len(res)} assets; drawdown shrank on "
      f"{(res['vt_mdd'] > res['bh_mdd']).sum()} of {len(res)}")

# ── Chart ────────────────────────────────────────────────────────────
plt.style.use("dark_background")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7),
                               gridspec_kw={"height_ratios": [3, 2]})
fig.patch.set_facecolor("#0a0a0a")
for ax in (ax1, ax2):
    ax.set_facecolor("#0a0a0a")
    ax.tick_params(colors="#e0e0e0")
    for spine in ax.spines.values():
        spine.set_color("#333333")

ax1.plot(curves["SPY"]["bh"].index, curves["SPY"]["bh"].values,
         color="#9ca3af", linewidth=1.4, label="SPY, buy and hold")
ax1.plot(curves["SPY"]["vt"].index, curves["SPY"]["vt"].values,
         color="#3b82f6", linewidth=1.6,
         label="SPY, volatility targeted (matched to the same volatility)")
ax1.set_yscale("log")
ax1.set_ylabel("Growth of one dollar, log scale", color="#e0e0e0")
ax1.set_title("Does volatility targeting improve Sharpe ratios?",
              color="#e0e0e0", fontsize=13)
ax1.legend(facecolor="#0a0a0a", edgecolor="#333333", labelcolor="#e0e0e0",
           loc="upper left")

x = np.arange(len(res))
ax2.bar(x - 0.26, res["bh_sharpe"], width=0.26, color="#9ca3af",
        label="Buy and hold")
ax2.bar(x, res["vt_sharpe"], width=0.26, color="#3b82f6",
        label="Volatility targeted, after costs")
ax2.bar(x + 0.26, res["peek_sharpe"], width=0.26, color="#3b82f6", alpha=0.35,
        label="Same rule with a one-day look-ahead")
ax2.set_xticks(x)
ax2.set_xticklabels(res.index, color="#e0e0e0")
ax2.set_ylabel("Sharpe ratio", color="#e0e0e0")
ax2.axhline(0, color="#333333", linewidth=0.8)
ax2.legend(facecolor="#0a0a0a", edgecolor="#333333", labelcolor="#e0e0e0",
           fontsize=9)

plt.tight_layout()
plt.savefig("volatility-targeting-sharpe-ratio-backtest-python.png", dpi=150,
            facecolor="#0a0a0a")
