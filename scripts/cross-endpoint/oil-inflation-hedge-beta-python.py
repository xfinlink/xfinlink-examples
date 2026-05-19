# Full write-up: https://xfinlink.com/blog/oil-inflation-hedge-beta-python

import xfinlink as xfl
import pandas as pd
import numpy as np
from fredapi import Fred
from scipy import stats
import matplotlib.pyplot as plt

xfl.api_key = "YOUR_API_KEY"  # free at https://xfinlink.com/signup
fred = Fred(api_key="YOUR_FRED_KEY")

# -- Configuration ----------------------------------------------------------
tickers = ["XOM", "CVX", "COP"]

# -- Monthly CPI from FRED --------------------------------------------------
cpi = fred.get_series("CPIAUCSL", observation_start="2018-01-01")
cpi_mom = cpi.pct_change().dropna().rename("cpi_change")
cpi_mom.index = cpi_mom.index.to_period("M")

# -- Monthly stock returns from xfinlink ------------------------------------
df = xfl.prices(tickers, start="2018-01-01", fields=["close"])
df["date"] = pd.to_datetime(df["date"])
monthly = df.groupby(["ticker", df["date"].dt.to_period("M")])["close"].last()
returns = monthly.groupby("ticker").pct_change().dropna().rename("ret")
returns = returns.reset_index()

# -- Full-sample OLS regression per ticker -----------------------------------
results = {}
for t in tickers:
    tr = returns[returns["ticker"] == t].set_index("date")["ret"]
    merged = pd.concat([tr, cpi_mom], axis=1, join="inner").dropna()

    slope, intercept, r, p, se = stats.linregress(
        merged["cpi_change"], merged["ret"]
    )

    # Rolling 24-month beta
    rolling_betas = []
    rolling_pvals = []
    dates = []
    for i in range(24, len(merged)):
        window = merged.iloc[i - 24 : i]
        s, _, rv, pv, _ = stats.linregress(
            window["cpi_change"], window["ret"]
        )
        rolling_betas.append(s)
        rolling_pvals.append(pv)
        dates.append(window.index[-1])

    rb = pd.Series(rolling_betas, index=dates)
    rp = pd.Series(rolling_pvals, index=dates)
    sig_pct = (rp < 0.10).mean() * 100
    median_beta = rb.median()

    results[t] = {
        "full_beta": slope,
        "p": p,
        "r2": r ** 2,
        "median_beta": median_beta,
        "sig_pct": sig_pct,
        "rolling_betas": rb,
        "rolling_pvals": rp,
    }

# -- Print results -----------------------------------------------------------
for t in tickers:
    r = results[t]
    print(
        f"{t}: full_beta={r['full_beta']:+.2f} (p={r['p']:.3f}, "
        f"R\u00b2={r['r2']:.3f}) | rolling median={r['median_beta']:+.2f}, "
        f"sig at 10%: {r['sig_pct']:.0f}%"
    )

print("\nConclusion: inflation betas are UNSTABLE")

# -- Chart: rolling 24-month inflation beta ----------------------------------
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
fig.suptitle(
    "Rolling 24-Month Inflation Beta: Oil Stocks vs CPI",
    fontsize=14,
    fontweight="bold",
)

for ax, t in zip(axes, tickers):
    rb = results[t]["rolling_betas"]
    rp = results[t]["rolling_pvals"]

    # Convert PeriodIndex to timestamps for plotting
    x = rb.index.to_timestamp()
    ax.plot(x, rb.values, color="#2563eb", linewidth=1.5, label="Beta")
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")

    # Shade significant windows
    sig_mask = rp < 0.10
    for i in range(len(x)):
        if sig_mask.iloc[i]:
            ax.axvspan(
                x[i] - pd.Timedelta(days=15),
                x[i] + pd.Timedelta(days=15),
                alpha=0.15,
                color="green",
            )

    ax.set_ylabel("Inflation Beta")
    ax.set_title(
        f"{t}  |  Full-sample beta: {results[t]['full_beta']:+.2f}  "
        f"p={results[t]['p']:.3f}  |  Sig windows: {results[t]['sig_pct']:.0f}%"
    )
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel("Date")
plt.tight_layout()
plt.savefig("oil-inflation-hedge-beta-python.png", dpi=150, bbox_inches="tight")
plt.show()
