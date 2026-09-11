# Full write-up: https://xfinlink.com/blog/how-to-run-an-event-study-in-python
#
# Event study of S&P 500 additions, 2023-2025.
# Market model estimated on sessions -120..-21, abnormal returns over -5..+20.

import numpy as np
import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

ev = xfl.index_events("sp500", event_type="added", start="2023-01-01", end="2025-12-31")
mkt = xfl.prices("SPY", start="2022-01-01", end="2026-03-31", fields=["close"])
mkt = mkt[["date", "close"]].rename(columns={"close": "mkt"}).set_index("date")
mkt["rm"] = mkt["mkt"].pct_change()

rows = []
for _, e in ev.iterrows():
    d = pd.Timestamp(e["effective_date"])
    px = xfl.prices(
        entity_id=int(e["entity_id"]),
        start=(d - pd.Timedelta(days=260)).strftime("%Y-%m-%d"),
        end=(d + pd.Timedelta(days=50)).strftime("%Y-%m-%d"),
        fields=["close"],
    )
    if px.empty:
        continue
    df = px[["date", "close"]].set_index("date").join(mkt[["rm"]], how="inner")
    df["ri"] = df["close"].pct_change()
    df = df.dropna()

    pos = df.index.searchsorted(d)
    est = df.iloc[max(0, pos - 120):pos - 20]
    win = df.iloc[pos - 5:pos + 21]
    if len(est) < 60 or len(win) < 26:
        continue

    beta, alpha = np.polyfit(est["rm"], est["ri"], 1)
    ar = win["ri"] - (alpha + beta * win["rm"])
    rows.append({
        "ticker": e["ticker"], "date": d.date(), "beta": beta,
        "car_incl": ar.iloc[0:7].sum() * 100,
        "ar_event": ar.iloc[5] * 100,
        "car_post": ar.iloc[7:26].sum() * 100,
    })

r = pd.DataFrame(rows).round(2)
print(f"events={len(ev)}  usable={len(r)}  no usable estimation window={len(ev) - len(r)}")
print()
print(r.head(6).to_string(index=False))
print()

labels = {"car_incl": "CAR(-5,+1)", "ar_event": "AR(0)", "car_post": "CAR(+2,+20)"}
for col, lab in labels.items():
    x = r[col]
    t = x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
    print(f"{lab:<12} mean={x.mean():+6.2f}%  median={x.median():+6.2f}%  "
          f"share>0={(x > 0).mean() * 100:4.1f}%  t={t:+5.2f}")
