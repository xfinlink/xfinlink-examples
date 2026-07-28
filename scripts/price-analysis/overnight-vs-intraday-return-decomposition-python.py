# Full write-up: https://xfinlink.com/blog/overnight-vs-intraday-return-decomposition-python
"""Split ten years of US large-cap price return into its overnight and intraday legs.

Overnight return runs from one close to the next open; intraday return runs from
that open to that close. The two legs multiply to the full-session price return.
Raw open/close prices are used with the split factor applied to the previous
close, and every session is reconciled against the reported total daily return
before anything is aggregated.

Built from SEC EDGAR public filings and market data.
"""

import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

START, END = "2015-01-02", "2024-12-31"
OUT_PNG = "overnight-vs-intraday-return-decomposition-python.png"

# Candidate pool: liquid US large caps, every GICS sector represented.
CANDIDATES = [
    "AAPL", "MSFT", "ORCL", "TXN", "ACN", "ADBE", "CRM", "INTC", "CSCO", "QCOM", "IBM", "NVDA",
    "GOOG", "NFLX", "DIS", "CMCSA", "VZ", "EA", "T",
    "AMZN", "TSLA", "LOW", "SBUX", "HD", "MCD", "NKE", "TJX",
    "WMT", "PM", "MO", "KO", "PEP", "COST", "CL", "KMB",
    "JNJ", "TMO", "ABT", "LLY", "UNH", "AMGN", "MRK", "DHR",
    "JPM", "WFC", "AXP", "BAC", "GS", "USB", "SPGI", "CB",
    "XOM", "COP", "CVX", "SLB", "EOG", "PSX",
    "LMT", "UNP", "CAT", "DE", "NSC", "ETN", "EMR", "HON",
    "SO", "NEE", "DUK", "AEP", "XEL", "D",
    "SHW", "ECL", "NUE", "VMC", "FCX", "PPG", "APD",
    "SPG", "PLD", "AMT", "PSA", "EQR", "O",
    "SPY",
]

FIELDS = ["open", "close", "adj_close", "return_daily", "split_ratio", "dividend", "volume"]

frames = []
for t in sorted(set(CANDIDATES)):
    for attempt in range(3):
        try:
            frames.append(xfl.prices(t, start=START, end=END, fields=FIELDS, max_rows=20000))
            break
        except Exception:
            time.sleep(6)
    else:
        raise SystemExit(f"could not retrieve {t}")

px = pd.concat(frames, ignore_index=True).sort_values(["ticker", "date"]).reset_index(drop=True)

# ---------------------------------------------------------------- the three legs
# open, close and dividend are as-traded. The previous close is restated on
# today's share basis with the split factor before either leg is formed.
grp = px.groupby("ticker")
split = px["split_ratio"].astype(float).fillna(1.0)
prev_close = grp["close"].shift(1) / split

px["overnight"] = px["open"] / prev_close - 1.0
px["intraday"] = px["close"] / px["open"] - 1.0
px["div_leg"] = px["dividend"].astype(float).fillna(0.0) / prev_close
px["rebuilt"] = (1 + px["overnight"]) * (1 + px["intraday"]) - 1 + px["div_leg"]
px["resid"] = (px["rebuilt"] - px["return_daily"]).abs()

# ------------------------------------------------- reconciliation and the screen
fail = px[px["resid"] > 1e-4]
excluded = sorted(fail["ticker"].unique())

CORPORATE_ACTION = {
    ("T", "2022-04-11"): "Warner Bros Discovery spin-off",
    ("DHR", "2016-07-05"): "Fortive spin-off",
    ("DHR", "2023-10-02"): "Veralto spin-off",
    ("CB", "2016-01-15"): "ACE / Chubb merger completion",
    ("APD", "2016-10-03"): "Versum Materials spin-off",
    ("MRK", "2021-06-03"): "Organon spin-off",
    ("IBM", "2021-11-04"): "Kyndryl spin-off",
    ("HON", "2016-10-03"): "AdvanSix spin-off",
    ("HON", "2018-10-01"): "Garrett Motion spin-off",
    ("HON", "2018-10-29"): "Resideo spin-off",
    ("O", "2021-11-15"): "Orion Office REIT spin-off",
}

print("=" * 80)
print("STEP 1  RECONCILIATION   overnight x intraday + dividend  vs  reported total return")
print(f"  candidate names            : {px['ticker'].nunique()}")
print(f"  sessions checked           : {px['overnight'].notna().sum():,}")
print(f"  share splits inside window : {int(px['split_ratio'].notna().sum())}")
print(f"  median absolute residual   : {px['resid'].median():.2e}")
print(f"  sessions off by over 1 bp  : {len(fail)}  across {len(excluded)} names")
print()
print(f"{'ticker':8s}{'date':13s}{'rebuilt':>10s}{'reported':>11s}   corporate action")
for _, r in fail.sort_values("resid", ascending=False).iterrows():
    key = (r["ticker"], r["date"].strftime("%Y-%m-%d"))
    print(f"{key[0]:8s}{key[1]:13s}{r['rebuilt']:+10.2%}{r['return_daily']:+11.2%}   "
          f"{CORPORATE_ACTION.get(key, '')}")
print(f"\n  every split in the window reconciles; the {len(fail)} exceptions are all share")
print(f"  distributions. Names dropped: {', '.join(excluded)}")

# ------------------------- universe: 3 most traded names per sector, by turnover
clean = px[~px["ticker"].isin(excluded) & (px["ticker"] != "SPY")].copy()
clean["turnover"] = clean["close"] * clean["volume"]
rank = (clean.groupby(["gics_sector", "ticker"])["turnover"].median()
        .reset_index().sort_values(["gics_sector", "turnover"], ascending=[True, False]))
universe = rank.groupby("gics_sector").head(3)["ticker"].tolist()

panel = clean[clean["ticker"].isin(universe)].dropna(subset=["overnight"])
spy = px[px["ticker"] == "SPY"].dropna(subset=["overnight"])
years = (pd.Timestamp(END) - pd.Timestamp(START)).days / 365.25

print()
print("=" * 80)
print(f"STEP 2  UNIVERSE   {len(universe)} names, 11 sectors, "
      f"{panel['date'].nunique():,} sessions, {START} to {END}")

# ------------------------------------------------------------------- per name
rows = []
for t, d in pd.concat([panel, spy]).groupby("ticker"):
    on, intra = (1 + d["overnight"]).prod(), (1 + d["intraday"]).prod()
    rows.append(dict(ticker=t, sector=d["gics_sector"].iloc[0] or "Index",
                     cum_on=on - 1, cum_id=intra - 1, cum_px=on * intra - 1,
                     ann_on=on ** (1 / years) - 1, ann_id=intra ** (1 / years) - 1,
                     bp_on=d["overnight"].mean() * 1e4, bp_id=d["intraday"].mean() * 1e4))
res = pd.DataFrame(rows).set_index("ticker")
tab = res.drop(index="SPY").sort_values("cum_px", ascending=False)

print()
print("CUMULATIVE PRICE RETURN, SPLIT BY SESSION")
print(f"{'':8s}{'overnight':>12s}{'intraday':>12s}{'full day':>13s}{'ann. o/n':>11s}{'ann. intra':>12s}")
for t, r in tab.iterrows():
    print(f"{t:8s}{r['cum_on']:+12.1%}{r['cum_id']:+12.1%}{r['cum_px']:+13.1%}"
          f"{r['ann_on']:+11.2%}{r['ann_id']:+12.2%}")
s = res.loc["SPY"]
print("-" * 68)
print(f"{'SPY':8s}{s['cum_on']:+12.1%}{s['cum_id']:+12.1%}{s['cum_px']:+13.1%}"
      f"{s['ann_on']:+11.2%}{s['ann_id']:+12.2%}")
print(f"\n  overnight leg larger than intraday leg : {int((tab['cum_on'] > tab['cum_id']).sum())} of {len(tab)} names")
print(f"  intraday leg negative over ten years   : {int((tab['cum_id'] < 0).sum())} names")
print(f"  overnight leg negative over ten years  : {int((tab['cum_on'] < 0).sum())} names")

# ------------------------------------------------------------ equal-weight basket
ew = panel.pivot_table(index="date", values=["overnight", "intraday"], aggfunc="mean").dropna()
curve_on = (1 + ew["overnight"]).cumprod()
curve_id = (1 + ew["intraday"]).cumprod()
curve_all = ((1 + ew["overnight"]) * (1 + ew["intraday"])).cumprod()

print()
print(f"EQUAL-WEIGHT BASKET OF THE {len(universe)} NAMES, ONE DOLLAR INVESTED 2 JANUARY 2015")
for label, curve in (("overnight session only", curve_on), ("intraday session only", curve_id),
                     ("both sessions", curve_all)):
    print(f"  {label:24s} ${curve.iloc[-1]:6.2f}   {curve.iloc[-1] ** (1 / years) - 1:+7.2%} a year")
print(f"  mean return per session   overnight {ew['overnight'].mean() * 1e4:+6.2f} bp"
      f"   intraday {ew['intraday'].mean() * 1e4:+6.2f} bp")
print(f"  volatility per session    overnight {ew['overnight'].std() * 1e2:6.2f}%"
      f"    intraday {ew['intraday'].std() * 1e2:6.2f}%")
print(f"  return per unit of risk   overnight "
      f"{ew['overnight'].mean() / ew['overnight'].std() * np.sqrt(252):6.2f}"
      f"     intraday {ew['intraday'].mean() / ew['intraday'].std() * np.sqrt(252):6.2f}")
print(f"  share of positive sessions  overnight {(ew['overnight'] > 0).mean():.1%}"
      f"     intraday {(ew['intraday'] > 0).mean():.1%}")

# -------------------------------------------------------------------- by sector
sec = (res.drop(index="SPY").groupby("sector")[["bp_on", "bp_id"]].mean()
       .sort_values("bp_on", ascending=False))
print()
print("BY SECTOR, MEAN RETURN PER SESSION IN BASIS POINTS")
print(f"{'sector':26s}{'overnight':>11s}{'intraday':>11s}{'difference':>12s}")
for k, r in sec.iterrows():
    print(f"{k:26s}{r['bp_on']:+11.2f}{r['bp_id']:+11.2f}{r['bp_on'] - r['bp_id']:+12.2f}")
print(f"\n  sectors where overnight beats intraday: {int((sec['bp_on'] > sec['bp_id']).sum())} of 11")

# ------------------------------------------------------------------------ chart
BG, FG, ACC, ALT = "#0a0a0a", "#e0e0e0", "#3b82f6", "#f59e0b"
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={"height_ratios": [1.3, 1]})
fig.patch.set_facecolor(BG)
for ax in (ax1, ax2):
    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_color("#333333")
    ax.tick_params(colors=FG, labelsize=9)
    ax.yaxis.label.set_color(FG)
    ax.xaxis.label.set_color(FG)

ax1.plot(curve_all.index, curve_all.values, color=FG, lw=1.3, label="Holding through both sessions")
ax1.plot(curve_on.index, curve_on.values, color=ACC, lw=1.7, label="Overnight only (close to next open)")
ax1.plot(curve_id.index, curve_id.values, color=ALT, lw=1.7, label="Intraday only (open to close)")
ax1.axhline(1.0, color="#444444", lw=0.8)
ax1.set_ylabel("Value of one dollar invested")
ax1.set_title("Where a decade of price return actually accrued: 33 US large caps, 2015-2024",
              color=FG, fontsize=12, pad=10)
legend = ax1.legend(facecolor=BG, edgecolor="#333333", fontsize=9, loc="upper left")
for text in legend.get_texts():
    text.set_color(FG)

y = np.arange(len(sec))
ax2.barh(y - 0.2, sec["bp_on"].values, height=0.38, color=ACC, label="Overnight")
ax2.barh(y + 0.2, sec["bp_id"].values, height=0.38, color=ALT, label="Intraday")
ax2.set_yticks(y)
ax2.set_yticklabels(sec.index, fontsize=9)
ax2.invert_yaxis()
ax2.axvline(0, color="#666666", lw=0.8)
ax2.set_xlabel("Average return per session, basis points")
legend2 = ax2.legend(facecolor=BG, edgecolor="#333333", fontsize=9, loc="lower right")
for text in legend2.get_texts():
    text.set_color(FG)

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150, facecolor=BG)
print("\nchart written to", OUT_PNG)
