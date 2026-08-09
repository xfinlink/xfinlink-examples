# Full write-up: https://xfinlink.com/blog/out-of-sample-return-predictability-python
"""Can any signal beat the historical average at forecasting next month's return?

Five lagged predictors, each fitted in an expanding window and scored against
the prevailing-mean benchmark with the Campbell-Thompson out-of-sample R.
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

TICKERS = ["SPY", "MDY", "TLT", "SHY", "LQD", "IEF"]
START, END = "2002-07-01", "2026-07-31"
MIN_TRAIN = 60          # months of history before the first live forecast
CHART = "out-of-sample-return-predictability-python.png"

# ── 1. Daily total returns → monthly ──────────────────────────────────

px = xfl.prices(TICKERS, start=START, end=END,
                fields=["return_daily"], max_rows=200000)
px = px.dropna(subset=["return_daily"])
px["month"] = px["date"].dt.to_period("M")

monthly = (px.groupby(["ticker", "month"])["return_daily"]
             .apply(lambda r: (1.0 + r).prod() - 1.0)
             .unstack(0)
             .sort_index())
monthly = monthly.iloc[1:]  # drop the partial first month


def trailing_12m(col):
    return (1.0 + monthly[col]).rolling(12).apply(np.prod, raw=True) - 1.0


# ── 2. Five predictors, all known at the end of month t ───────────────

X = pd.DataFrame({
    "momentum": trailing_12m("SPY"),
    "volatility": monthly["SPY"].rolling(12).std() * np.sqrt(12),
    "term spread": trailing_12m("TLT") - trailing_12m("SHY"),
    "credit spread": trailing_12m("LQD") - trailing_12m("IEF"),
    "size spread": trailing_12m("MDY") - trailing_12m("SPY"),
})
X["y_next"] = monthly["SPY"].shift(-1)          # the month being forecast
data = X.dropna()
PREDICTORS = [c for c in data.columns if c != "y_next"]
y = data["y_next"].to_numpy()
n = len(data)


def ols_predict(x_train, y_train, x_new):
    """Fit y = a + b'x on the training window, return the fitted value at x_new."""
    A = np.column_stack([np.ones(len(x_train)), x_train])
    beta, *_ = np.linalg.lstsq(A, y_train, rcond=None)
    return float(beta[0] + np.dot(beta[1:], x_new))


# ── 3. Expanding-window forecasts against the prevailing mean ─────────

models = {p: [[p]] for p in PREDICTORS}
models["all five"] = [PREDICTORS]

fcst = {name: np.full(n, np.nan) for name in models}
mean_fcst = np.full(n, np.nan)

for i in range(MIN_TRAIN, n):
    y_train = y[:i]
    mean_fcst[i] = y_train.mean()
    for name, (cols,) in models.items():
        x_train = data[cols].to_numpy()[:i]
        fcst[name][i] = ols_predict(x_train, y_train, data[cols].to_numpy()[i])

live = slice(MIN_TRAIN, n)
err_mean = y[live] - mean_fcst[live]
sse_mean = np.sum(err_mean ** 2)

rows, cssed = [], {}
for name in models:
    f = fcst[name][live]
    err = y[live] - f
    err_ct = y[live] - np.maximum(f, 0.0)      # Campbell-Thompson restriction
    cols = models[name][0]
    A = np.column_stack([np.ones(n), data[cols].to_numpy()])
    resid = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
    rows.append({
        "model": name,
        "is_r2": (1.0 - resid.var() / y.var()) * 100,
        "oos_r2": (1.0 - np.sum(err ** 2) / sse_mean) * 100,
        "oos_r2_ct": (1.0 - np.sum(err_ct ** 2) / sse_mean) * 100,
        "hit": np.mean(np.sign(f) == np.sign(y[live])) * 100,
    })
    cssed[name] = np.cumsum(err_mean ** 2 - err ** 2)

table = pd.DataFrame(rows)
months = data.index[live] + 1   # the month being forecast

# ── 4. Report ─────────────────────────────────────────────────────────

print(f"Forecasting SPY monthly total return, one month ahead")
print(f"Training starts {data.index[0]}, {len(months)} live forecasts "
      f"{months[0]} to {months[-1]}\n")
print(f"{'model':16}{'in-sample R2':>14}{'OOS R2':>9}{'OOS R2 (CT)':>13}{'sign hit%':>11}")
for _, r in table.iterrows():
    print(f"{r['model']:16}{r['is_r2']:>13.2f}%{r['oos_r2']:>8.2f}%"
          f"{r['oos_r2_ct']:>12.2f}%{r['hit']:>10.1f}%")

print(f"\nBenchmark: prevailing mean of {y[:MIN_TRAIN].mean() * 100:.2f}% at the first "
      f"forecast, {y[:n - 1].mean() * 100:.2f}% at the last")
print(f"Realised mean over the live window: {y[live].mean() * 100:.2f}% per month, "
      f"standard deviation {y[live].std(ddof=1) * 100:.2f}%")
print(f"Positive months in the live window: {np.mean(y[live] > 0) * 100:.1f}% "
      f"(the base rate the sign hit rate has to beat)")
best = table.loc[table["oos_r2"].idxmax()]
print(f"Best out-of-sample model: {best['model']} at {best['oos_r2']:.2f}%")

# ── 5. Chart ──────────────────────────────────────────────────────────

plt.rcParams.update({
    "figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
    "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
    "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
    "axes.edgecolor": "#333333", "font.size": 10,
})
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))

pos = np.arange(len(table))
ax1.bar(pos - 0.2, table["is_r2"], 0.4, color="#3b82f6", label="Fitted on all the data")
ax1.bar(pos + 0.2, table["oos_r2"], 0.4, color="#6b7280", label="Live forecasts only")
ax1.axhline(0, color="#888888", lw=0.9)
ax1.set_xticks(pos)
ax1.set_xticklabels(table["model"])
ax1.set_ylabel("Variance explained (%)")
ax1.set_title("Predicting next month's return: in-sample fit versus live forecasts")
ax1.legend(frameon=False, loc="lower left")
for spine in ("top", "right"):
    ax1.spines[spine].set_visible(False)

dates = months.to_timestamp()
colors = ["#3b82f6", "#f59e0b", "#10b981", "#ef4444", "#a78bfa", "#e0e0e0"]
for (name, series), color in zip(cssed.items(), colors):
    ax2.plot(dates, series * 1e4, color=color, lw=1.4, label=name)
ax2.axhline(0, color="#888888", lw=0.9)
ax2.set_ylabel("Cumulative squared error saved (%-squared)")
ax2.set_xlabel("Year")
ax2.set_title("Above zero means the signal beat the historical average so far")
ax2.legend(frameon=False, ncol=3, loc="lower left", fontsize=9)
for spine in ("top", "right"):
    ax2.spines[spine].set_visible(False)

plt.tight_layout(h_pad=2.0)
plt.savefig(CHART, dpi=150, facecolor="#0a0a0a")
print(f"\nchart saved to {CHART}")
