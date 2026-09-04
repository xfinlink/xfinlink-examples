# Full write-up: https://xfinlink.com/blog/index-fund-ownership-stock-beta-python
"""Does heavy index-fund ownership make a stock move with the index?

Big Three (BlackRock, Vanguard, State Street) Form 13F ownership of every
S&P 500 member at 31 December 2024, against each stock's market beta,
R-squared and idiosyncratic volatility over calendar 2025.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

Q = "2024-12-31"
BIG3 = {2432: "BlackRock", 10139: "Vanguard", 486: "State Street"}

# ── 1. universe and the three books ──────────────────────────────────
roster = xfl.index("sp500", as_of=Q).drop_duplicates("entity_id")

books = pd.concat([
    xfl.manager_holdings(mid, quarter=Q, max_rows=20000,
                         fields=["entity_id", "entity_name", "value_usd", "put_call"])
    for mid in BIG3
])
books = books[books["put_call"].isna()]

# a second share class files under its own entity; fold it into the issuer
root = roster[["entity_id", "entity_name"]].rename(
    columns={"entity_id": "issuer_id", "entity_name": "root"})
extra = books.loc[~books["entity_id"].isin(set(roster["entity_id"])),
                  ["entity_id", "entity_name"]].drop_duplicates()
pairs = extra.merge(root, how="cross")
pairs = pairs[[c.upper().startswith(r.upper() + " ")
               for c, r in zip(pairs["entity_name"], pairs["root"])]]
books["entity_id"] = books["entity_id"].replace(
    dict(zip(pairs["entity_id"], pairs["issuer_id"])))

big3 = books.groupby("entity_id")["value_usd"].sum().rename("big3_value").reset_index()

# ── 2. ownership as a share of market value ──────────────────────────
tick = roster["ticker"].tolist()
caps = pd.concat([
    xfl.metrics(tick[i:i + 100], period_type="daily", fields=["market_cap"],
                start="2024-12-24", end=Q, max_rows=20000)
    for i in range(0, len(tick), 100)
])
caps = caps.sort_values("period_end").groupby("entity_id").tail(1)
caps["mcap"] = caps["market_cap"] * 1e6

df = roster.merge(big3, on="entity_id", how="left").merge(
    caps[["entity_id", "mcap"]], on="entity_id")
df["own"] = df["big3_value"] / df["mcap"]
n_priced, n_nomatch = len(df), int(df["big3_value"].isna().sum())
df = df[df["own"].between(0.02, 0.40)]

# ── 3. beta, R-squared and residual volatility over 2025 ─────────────
ids = sorted(int(e) for e in df["entity_id"])
px = pd.concat([
    xfl.prices(entity_id=ids[i:i + 40], start="2024-12-24", end="2025-12-31",
               fields=["adj_close"], max_rows=60000)
    for i in range(0, len(ids), 40)
])
spy = xfl.prices("SPY", start="2024-12-24", end="2025-12-31", fields=["adj_close"])
mkt = spy.sort_values("date").set_index("date")["adj_close"].pct_change().rename("mkt")

rows = []
for eid, g in px.groupby("entity_id"):
    r = g.sort_values("date").set_index("date")["adj_close"].pct_change().rename("r")
    j = pd.concat([r, mkt], axis=1, sort=False).dropna()
    if len(j) < 200:
        continue
    beta = np.cov(j["r"], j["mkt"])[0, 1] / np.var(j["mkt"], ddof=1)
    resid = j["r"] - beta * j["mkt"]
    rows.append({"entity_id": eid, "days": len(j), "beta": beta,
                 "r2": j["r"].corr(j["mkt"]) ** 2,
                 "total_vol": j["r"].std() * np.sqrt(252),
                 "idio_vol": resid.std() * np.sqrt(252)})

res = df.merge(pd.DataFrame(rows), on="entity_id")
res["sector"] = res["entity_id"].map(px.drop_duplicates("entity_id")
                                     .set_index("entity_id")["gics_sector"])
res["logcap"] = np.log(res["mcap"])
res["q"] = pd.qcut(res["own"], 5, labels=[1, 2, 3, 4, 5])

# ── 4. quintiles and the cross-sectional regression ──────────────────
tab = res.groupby("q", observed=True).agg(
    n=("ticker", "size"), own=("own", "mean"), beta=("beta", "mean"),
    r2=("r2", "mean"), total_vol=("total_vol", "mean"),
    idio_vol=("idio_vol", "mean"), mcap=("mcap", "median"))

dummies = pd.get_dummies(res["sector"], drop_first=True).astype(float).reset_index(drop=True)
fits = {}
for y in ("beta", "r2", "idio_vol"):
    plain = sm.OLS(res[y], sm.add_constant(res[["own"]])).fit()
    X = pd.concat([sm.add_constant(res[["own", "logcap"]]).reset_index(drop=True),
                   dummies], axis=1)
    full = sm.OLS(res[y].reset_index(drop=True), X).fit()
    fits[y] = (plain, full)

# ── 5. report ────────────────────────────────────────────────────────
print("Big Three ownership and market sensitivity, S&P 500")
print(f"  index members priced at {Q}                {n_priced:>6}")
print(f"  no matched Form 13F position               {n_nomatch:>6}")
print(f"  stake outside 2-40% of market value        {n_priced - n_nomatch - len(df):>6}")
print(f"  fewer than 200 trading days in 2025        {len(df) - len(res):>6}")
print(f"  companies in the sample                    {len(res):>6}")
print(f"\nCombined stake: mean {res['own'].mean():.1%}, median {res['own'].median():.1%}, "
      f"range {res['own'].min():.1%} ({res.nsmallest(1, 'own')['ticker'].iloc[0]}) to "
      f"{res['own'].max():.1%} ({res.nlargest(1, 'own')['ticker'].iloc[0]})")
print(f"Correlation with log market cap: {res['own'].corr(res['logcap']):+.3f}\n")

print("Quintiles of Big Three ownership, sorted low to high")
print(f"{'Quintile':<9}{'n':>5}{'Owned':>8}{'Beta':>8}{'R-sq':>8}"
      f"{'Vol':>8}{'Idio vol':>10}{'Median cap':>13}")
print("-" * 69)
for q, row in tab.iterrows():
    print(f"{q:<9}{row['n']:>5.0f}{row['own']:>8.1%}{row['beta']:>8.2f}{row['r2']:>8.2f}"
          f"{row['total_vol']:>8.1%}{row['idio_vol']:>10.1%}"
          f"{row['mcap'] / 1e9:>11.1f}bn")

print("\nCross-sectional regression, coefficient on ownership share")
print(f"{'Dependent variable':<22}{'Ownership only':>22}{'+ size + sector':>22}")
print("-" * 66)
for y, label in [("beta", "Market beta"), ("r2", "R-squared"),
                 ("idio_vol", "Idiosyncratic vol")]:
    plain, full = fits[y]
    print(f"{label:<22}"
          f"{plain.params['own']:>+11.3f} (t{plain.tvalues['own']:>6.2f})"
          f"{full.params['own']:>+11.3f} (t{full.tvalues['own']:>6.2f})")

print("\nSector means")
sec = res.groupby("sector").agg(n=("ticker", "size"), own=("own", "mean"),
                                beta=("beta", "mean")).sort_values("own", ascending=False)
for s, row in sec.iterrows():
    print(f"  {s:<24}{row['n']:>4.0f}{row['own']:>8.1%}{row['beta']:>8.2f}")

# ── 6. chart ─────────────────────────────────────────────────────────
plt.rcParams.update({"figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
                     "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
                     "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
                     "axes.edgecolor": "#3a3a3a", "font.size": 10})
fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(res["own"] * 100, res["beta"], s=14, alpha=0.45, color="#3b82f6",
           edgecolors="none")
grid = np.linspace(res["own"].min(), res["own"].max(), 50)
plain = fits["beta"][0]
ax.plot(grid * 100, plain.params["const"] + plain.params["own"] * grid,
        color="#f59e0b", lw=2, label="Fitted line")
ax.plot(tab["own"] * 100, tab["beta"], "o-", color="#e0e0e0", lw=1.6, ms=7,
        label="Quintile average")
ax.set_xlabel("Share of company owned by BlackRock, Vanguard and State Street (%)")
ax.set_ylabel("Beta to the S&P 500 during 2025")
ax.set_title("Index-fund ownership and market sensitivity, 484 S&P 500 companies")
ax.legend(frameon=False, loc="upper right")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("index-fund-ownership-stock-beta-python.png", dpi=150,
            facecolor="#0a0a0a")
