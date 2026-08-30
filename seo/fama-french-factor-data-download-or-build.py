# Full write-up: https://xfinlink.com/blog/fama-french-factor-data-download-or-build
import numpy as np
import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

FORM = pd.date_range("2020-12-31", "2024-11-30", freq="ME")  # 48 formation dates

# 1. The universe as it stood on each formation date, carried by entity_id
rosters = {d: xfl.index("sp500", as_of=d.strftime("%Y-%m-%d")) for d in FORM}
ids = sorted({int(i) for r in rosters.values() for i in r["entity_id"]})

# 2. Daily total return and market cap for every company that was ever a member
frames = []
for i in range(0, len(ids), 50):
    frames.append(xfl.prices(entity_id=ids[i:i + 50], start="2019-12-01", end="2024-12-31",
                             fields=["return_daily", "market_cap"], max_rows=200000))
px = pd.concat(frames, ignore_index=True)
px["month"] = px["date"].dt.to_period("M")

# 3. Monthly total return and month-end market cap, one column per company
ret = (px.dropna(subset=["return_daily"]).groupby(["entity_id", "month"])["return_daily"]
         .apply(lambda s: (1 + s).prod() - 1).unstack(0))
cap = (px.dropna(subset=["market_cap"]).sort_values("date")
         .groupby(["entity_id", "month"])["market_cap"].last().unstack(0)
         .reindex(columns=ret.columns))

# 4. Momentum signal: cumulative return over months t-12 to t-2
signal = ((1 + ret).rolling(11).apply(np.prod, raw=True) - 1).shift(1)


def spread(sig, fwd):
    """Next month's equal-weighted return, top 30% by signal minus bottom 30%."""
    names = sig.dropna().index.intersection(fwd.dropna().index)
    pct = sig[names].rank(pct=True)
    return fwd[pct[pct >= 0.7].index].mean() - fwd[pct[pct <= 0.3].index].mean()


rows = []
for d, roster in rosters.items():
    formation = pd.Period(d, freq="M")
    members = [int(i) for i in roster["entity_id"] if int(i) in ret.columns]
    fwd = ret.loc[formation + 1, members]
    rows.append({"month": formation + 1,
                 "size": -spread(cap.loc[formation, members], fwd),  # small minus big
                 "momentum": spread(signal.loc[formation, members], fwd)})

built = pd.DataFrame(rows).set_index("month")

# 5. The published series for the same months, straight from the library
URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"


def library(zip_name, column, skiprows):
    df = pd.read_csv(URL + zip_name, skiprows=skiprows)
    df = df.rename(columns={df.columns[0]: "ym"})
    df = df[df["ym"].astype(str).str.strip().str.fullmatch(r"\d{6}")]
    df.index = pd.PeriodIndex(df["ym"].astype(str).str.strip(), freq="M")
    return df[column].astype(float) / 100


both = built.join(library("F-F_Research_Data_Factors_CSV.zip", "SMB", 3)) \
            .join(library("F-F_Momentum_Factor_CSV.zip", "Mom", 13)).dropna()

print(f"companies ever in the index: {len(ids)}   months: {len(both)}")
for mine, published in [("size", "SMB"), ("momentum", "Mom")]:
    for label, s in [(f"S&P 500 {mine}", both[mine]), (f"library {published}", both[published])]:
        print(f"{label:20s} annualised {s.mean() * 12:7.2%}   volatility {s.std() * np.sqrt(12):6.2%}"
              f"   cumulative {(1 + s).prod() - 1:7.2%}")
    print(f"{'correlation':20s} {both[mine].corr(both[published]):.2f}\n")
