# Quantitative Factor Research Study

A rigorous, end-to-end quantitative factor research framework built in Python.
Demonstrates the full research workflow: hypothesis formation, data engineering,
factor construction, backtesting, and honest interpretation of results.

---

## Project Structure

```
quant_factor_research/
├── main.py              # Orchestrator — run this
├── requirements.txt
├── src/
│   ├── pipeline.py      # Section 1: Data download and cleaning
│   ├── factors.py       # Section 2: Factor signal construction
│   ├── backtest.py      # Section 3: Portfolio formation and performance metrics
│   └── analysis.py      # Section 4: Charts and summary tables
├── data/                # Auto-created — cached parquet files
└── output/              # Auto-created — all charts saved here
```

---

## Setup

```bash
pip install -r requirements.txt
python main.py
```

**Options:**
```bash
python main.py --start 2010-01-01 --end 2023-12-31   # custom date range
python main.py --refresh                               # force re-download data
python main.py --no-cost-adjust                        # skip transaction cost netting
```

The first run downloads ~500 stocks × 20 years of data. Expect 5–10 minutes.
Subsequent runs load from the parquet cache and finish in under a minute.

---

## Factors

### 1. Momentum (MOM)
- **Construction:** Compound return from month *t−12* to *t−2* (12-1 convention).
  The most recent month is skipped to avoid short-term reversal contamination.
- **Academic basis:** Jegadeesh & Titman (1993) — *Returns to Buying Winners and
  Selling Losers.* One of the most replicated findings in empirical finance.
- **Hypothesis:** Stocks with strong past performance continue to outperform
  over the subsequent 3–12 month horizon.
- **Portfolio:** Long Q5 (highest past return), short Q1 (lowest).

### 2. Low Volatility (LVOL)
- **Construction:** Trailing 252-day realized standard deviation of daily returns,
  annualized. Signal is negated so high signal = low volatility.
- **Academic basis:** Ang et al. (2006) — *The Cross-Section of Volatility and
  Expected Returns.* The "low vol anomaly": less volatile stocks earn higher
  risk-adjusted returns than theory predicts.
- **Hypothesis:** Low-volatility stocks are systematically underpriced because
  institutional investors with leverage constraints prefer high-vol names.
- **Portfolio:** Long Q5 (lowest volatility), short Q1 (highest volatility).

### 3. Short-term Reversal (REV)
- **Construction:** Prior 1-month return, negated.
- **Academic basis:** Jegadeesh (1990) — *Evidence of Predictable Behavior of
  Security Returns.* Strong winners over one month tend to give back gains.
- **Hypothesis:** Market microstructure effects and liquidity provision dynamics
  cause mean reversion at the 1-month horizon.
- **Portfolio:** Long Q5 (prior month's losers), short Q1 (prior month's winners).

---

## Methodology

### Portfolio Formation
- Universe: S&P 500 current constituents (see Limitations).
- Rebalancing: monthly, at end-of-month close.
- Portfolio construction: equal-weighted within each quintile.
- Signal → return shift: signal computed at *t*, return measured at *t+1*.
  This is the core **lookahead bias prevention** mechanism.

### Performance Metrics
| Metric | Description |
|---|---|
| Annualized Return | CAGR from monthly returns |
| Sharpe Ratio | Ann. return / ann. vol (RF = 0%) |
| Max Drawdown | Largest peak-to-trough loss |
| Hit Rate | % of months with positive return |
| Avg Turnover | Avg % of portfolio replaced each month |

### Transaction Costs
- Default: **10 bps one-way** (20 bps round-trip) per rebalance.
- Applied proportionally to monthly turnover of each quintile.
- Conservative for liquid large-cap stocks; adjust for real-world conditions.

---

## Output Charts

For each factor:
- `{Factor}_cumulative_returns.png` — All quintiles + L/S gross vs net
- `{Factor}_quintile_bar.png` — Annualized return per quintile (monotonicity test)
- `{Factor}_drawdown.png` — Underwater equity curve
- `{Factor}_rolling_sharpe.png` — 36-month rolling Sharpe ratio

Combined:
- `factor_correlation_heatmap.png` — Pairwise L/S return correlations
- `all_factors_combined.png` — All factors' L/S returns on one chart

---

## Known Limitations

These are documented deliberately — understanding what a backtest *cannot* tell
you is as important as the results themselves.

| Bias | Description | Impact |
|---|---|---|
| **Survivorship bias** | We use today's S&P 500 list, not point-in-time historical constituents. Companies that were removed due to bankruptcy or poor performance are excluded. | Upward bias on returns, especially for value and momentum strategies. |
| **Lookahead in universe** | The constituent list itself is known only in hindsight. | Universe selection bias compounds survivorship bias. |
| **Price impact / market impact** | We assume trades at closing prices with no slippage. Real rebalances move the market. | Overstated returns for high-turnover strategies. |
| **Risk-free rate = 0** | Sharpe ratios don't subtract T-bill returns. | Overstated Sharpe ratios in high-rate environments. |
| **No fundamental data** | Value factor excluded because yfinance book values are not point-in-time safe. Using them would introduce lookahead bias via accounting restatements. | Limits the factor set. |

---

## Skills Demonstrated

- **Data engineering:** batched downloads, caching, missing data handling
- **Factor construction:** academically grounded, documented assumptions
- **Lookahead bias prevention:** explicit forward-shift of signals
- **Transaction cost modeling:** turnover-proportional cost adjustment
- **Statistical rigor:** Sharpe, drawdown, rolling metrics, monotonicity testing
- **Honest reporting:** documented limitations, survivorship bias warning

---

## References

1. Jegadeesh, N. & Titman, S. (1993). *Returns to Buying Winners and Selling Losers.*
   Journal of Finance, 48(1), 65–91.
2. Ang, A., Hodrick, R., Xing, Y., & Zhang, X. (2006). *The Cross-Section of
   Volatility and Expected Returns.* Journal of Finance, 61(1), 259–299.
3. Jegadeesh, N. (1990). *Evidence of Predictable Behavior of Security Returns.*
   Journal of Finance, 45(3), 881–898.
4. Fama, E. & French, K. (1993). *Common Risk Factors in the Returns on Stocks
   and Bonds.* Journal of Financial Economics, 33(1), 3–56.
