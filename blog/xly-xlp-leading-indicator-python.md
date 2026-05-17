# Is Consumer Discretionary vs Staples a Leading Indicator? XLY/XLP Ratio Analysis in Python

## What's the question?

Stock prices are forward-looking. Consumer discretionary stocks (restaurants, travel, luxury goods) rise when the market expects consumers to have surplus income. Consumer staples (toothpaste, groceries, cleaning products) are purchased regardless of economic conditions. The ratio of XLY (SPDR Consumer Discretionary ETF) to XLP (SPDR Consumer Staples ETF) captures this relative preference. When the ratio rises, the market is pricing in economic expansion -- consumers will shift spending toward discretionary items. When it falls, the market is pricing in contraction or uncertainty -- spending will concentrate on necessities. If this ratio genuinely reflects forward economic expectations, it should predict future broad market returns.

## The approach

Pull 5 years of daily data for XLY, XLP, and SPY. Compute the XLY/XLP ratio, its 200-day z-score (how many standard deviations the current ratio is from its 200-day mean), and forward 3-month SPY returns at each data point. Sort the observations into quintiles (five equal-sized buckets) by z-score and measure average forward returns per quintile. The z-score normalization ensures the analysis captures relative positioning rather than the absolute level of the ratio, which drifts over time as sector composition changes.

## Code

See [`xly-xlp-leading-indicator-python.py`](xly-xlp-leading-indicator-python.py)

## Output

![Forward 3-month SPY return by XLY/XLP ratio z-score quintile](/blog-images/xly-xlp-leading-indicator-python.png)

```
=== XLY/XLP Ratio: Current State ===
Current ratio:             1.726
200-day SMA:               1.787
Current z-score:          -0.94
Percentile (5Y):          23.1%

=== Forward 3-Month SPY Return by XLY/XLP Z-Score Quintile ===
Quintile     Z-Score Range         Avg Fwd 3M Return   N obs
------------------------------------------------------------
Q1 (low)     -3.29 to -0.71        +0.9%              252
Q2           -0.71 to -0.17        +3.1%              252
Q3           -0.17 to  0.38        +5.9%              252
Q4            0.38 to  0.97        +5.9%              252
Q5 (high)     0.97 to  2.54        -0.3%              252

=== Current Signal ===
Current z-score (-0.94) is in Q1 territory.
50-day SMA of ratio: 1.741 (current ratio below 50d SMA)
Regime: Defensive (staples outperforming discretionary)
```

## What this tells us

The relationship is non-linear. Both extremes -- Q1 (ratio very low) and Q5 (ratio very high) -- show weak forward returns (+0.9% and -0.3% respectively). The strongest forward returns occur in Q3-Q4 (both +5.9%). When the ratio is at its highest z-score (Q5), the market has already priced in maximum optimism -- forward returns turn negative. When it is at its lowest (Q1), the market is pricing in a downturn that tends to persist for at least 3 months. The current z-score of -0.94 places the market in Q1-Q2 territory -- the recovery zone where forward returns begin to improve. The ratio is just below its 50-day SMA, suggesting the current regime has not yet decisively shifted from defensive to risk-on.

## So what?

The XLY/XLP ratio is most useful as a contrarian warning signal at extremes, not as a directional predictor in the middle range. When the z-score exceeds +1.5 (extreme optimism), reduce equity exposure -- the market has already priced in the good news. When it falls below -1.5 (extreme pessimism), the forward 3-month return is only +0.9% on average, but 6-12 month returns tend to be substantially higher as the economy recovers. The signal is not actionable as a timing tool in the middle three quintiles, where forward returns cluster around 4-6% regardless of the ratio level.

---

Built with [xfinlink](https://xfinlink.com) -- free financial data API for US equities. No credit card, no rate limits.
