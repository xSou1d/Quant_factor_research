"""
================================================================================
MODULE 5: VALUE FACTOR CONSTRUCTION
================================================================================
Factor: Earnings Yield (EY)
  Earnings Yield = Trailing Twelve Month Net Income / Market Capitalization

Academic basis:
  Fama & French (1992) showed that earnings yield (E/P) is one of the strongest
  cross-sectional predictors of stock returns. High earnings yield ("cheap" stocks)
  outperform low earnings yield ("expensive" stocks) over long horizons.

Construction:
  1. Download annual income statements for each ticker via yfinance.
  2. Assign an availability date of fiscal year end + 60 days (reporting lag).
  3. For each month t, use the most recently available Net Income figure.
  4. Approximate market cap = current shares outstanding × end-of-month price.
  5. Earnings Yield = Net Income / Market Cap.
  6. Cross-sectionally winsorize at 1st/99th percentile each month.

Limitations (documented explicitly):
  - Shares outstanding: yfinance provides current share count only, not historical.
    This is approximately correct for stable large-cap S&P 500 stocks but
    introduces modest error for companies with aggressive buyback programs.
  - Reporting lag: fiscal year end + 60 days is a proxy. Actual announcement
    dates vary (30–90 days after fiscal year end) and are unavailable from yfinance.
  - Restatements: yfinance returns restated financials, not originally reported
    figures. Restatements introduce a small amount of lookahead bias.
  - Coverage: tickers with missing or incomplete fundamental data receive NaN
    signals and are excluded from rankings for affected months.
================================================================================
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm

REPORTING_LAG_DAYS = 60   # days after fiscal year end before earnings are "known"


# ── Step 1: Download Fundamental Data ────────────────────────────────────────

def fetch_fundamental_data(tickers: list[str], data_dir: str) -> pd.DataFrame:
    """
    Download annual Net Income and shares outstanding for each ticker.
    Results are cached to disk — first run takes ~5 minutes for 400+ tickers.

    Returns
    -------
    DataFrame with columns: ticker, fiscal_end, available_date, net_income, shares
    """
    cache_file = os.path.join(data_dir, "fundamentals.parquet")

    if os.path.exists(cache_file):
        print("[Value] Loading cached fundamental data...")
        df = pd.read_parquet(cache_file)
        print(f"[Value] Loaded fundamentals for {df['ticker'].nunique()} tickers.")
        return df

    print(f"[Value] Downloading fundamental data for {len(tickers)} tickers...")
    print("[Value] This takes ~5 minutes. Results will be cached for future runs.")
    records = []
    failed  = 0

    for ticker in tqdm(tickers, desc="Fetching fundamentals"):
        try:
            t          = yf.Ticker(ticker)
            financials = t.financials  # annual income statement

            if financials is None or financials.empty:
                failed += 1
                continue

            if "Net Income" not in financials.index:
                failed += 1
                continue

            net_income_series = financials.loc["Net Income"]

            # Get current shares outstanding
            shares = None
            try:
                shares = t.fast_info.get("shares", None)
            except Exception:
                pass
            if shares is None:
                try:
                    shares = t.info.get("sharesOutstanding", None)
                except Exception:
                    pass
            if not shares or shares <= 0:
                failed += 1
                continue

            for fiscal_end_date, net_income in net_income_series.items():
                if pd.isna(net_income):
                    continue
                records.append({
                    "ticker":         ticker,
                    "fiscal_end":     pd.Timestamp(fiscal_end_date),
                    "available_date": pd.Timestamp(fiscal_end_date)
                                      + pd.Timedelta(days=REPORTING_LAG_DAYS),
                    "net_income":     float(net_income),
                    "shares":         float(shares),
                })

        except Exception:
            failed += 1
            continue

    df = pd.DataFrame(records)

    if df.empty:
        print("[Value] Warning: no fundamental data retrieved.")
        return df

    df.to_parquet(cache_file)
    success = df["ticker"].nunique()
    print(f"[Value] Downloaded: {success} tickers succeeded, {failed} failed.")
    return df


# ── Step 2: Build Monthly Earnings Yield Panel ────────────────────────────────

def build_earnings_yield(
    fundamentals:   pd.DataFrame,
    monthly_prices: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each month end t and each stock, compute:
        Earnings Yield = Net Income (most recent available at t) / Market Cap (at t)

    Uses merge_asof so that only earnings with available_date <= month end are used.
    This ensures no future earnings information contaminates the signal.

    Parameters
    ----------
    fundamentals    : output of fetch_fundamental_data()
    monthly_prices  : (months × tickers) end-of-month adjusted close prices

    Returns
    -------
    earnings_yield  : (months × tickers) earnings yield signal
    """
    monthly_index = monthly_prices.index
    result        = pd.DataFrame(
        np.nan,
        index=monthly_index,
        columns=monthly_prices.columns,
    )

    month_df = pd.DataFrame({"date": monthly_index})

    for ticker in tqdm(monthly_prices.columns, desc="Building earnings yield"):
        ticker_data = fundamentals[fundamentals["ticker"] == ticker].copy()
        if ticker_data.empty:
            continue

        ticker_data = (
            ticker_data
            .sort_values("available_date")
            [["available_date", "net_income", "shares"]]
            .rename(columns={"available_date": "date"})
        )

        # As-of merge: each month gets the most recently available earnings
        merged = pd.merge_asof(
            month_df,
            ticker_data,
            on="date",
            direction="backward",
        )
        merged.index = monthly_index

        prices = monthly_prices[ticker]
        valid  = (
            merged["net_income"].notna()
            & prices.notna()
            & (prices > 0)
            & (merged["shares"] > 0)
        )

        if valid.sum() == 0:
            continue

        market_cap = merged.loc[valid, "shares"] * prices[valid]
        result.loc[valid, ticker] = (
            merged.loc[valid, "net_income"].values / market_cap.values
        )

    return result


# ── Step 3: Winsorize ─────────────────────────────────────────────────────────

def winsorize(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    lo = series.quantile(lower)
    hi = series.quantile(upper)
    return series.clip(lo, hi)


def apply_monthly_winsorize(factor: pd.DataFrame) -> pd.DataFrame:
    return factor.apply(winsorize, axis=1)


# ── Main Entry Point ──────────────────────────────────────────────────────────

def compute_value(
    monthly_prices: pd.DataFrame,
    tickers:        list[str],
    data_dir:       str,
) -> pd.DataFrame:
    """
    Full value factor pipeline: download → build panel → winsorize.

    Parameters
    ----------
    monthly_prices : (months × tickers) end-of-month prices from the pipeline
    tickers        : list of ticker symbols (should match monthly_prices.columns)
    data_dir       : path to data directory for caching

    Returns
    -------
    value_signal : (months × tickers) winsorized earnings yield
    """
    fundamentals   = fetch_fundamental_data(tickers, data_dir)

    if fundamentals.empty:
        print("[Value] No fundamental data available — skipping value factor.")
        return pd.DataFrame(np.nan, index=monthly_prices.index,
                            columns=monthly_prices.columns)

    earnings_yield = build_earnings_yield(fundamentals, monthly_prices)
    signal         = apply_monthly_winsorize(earnings_yield)

    coverage = signal.notna().sum(axis=1).mean()
    print(f"[Value] Avg {coverage:.0f} stocks with earnings yield signal per month.")

    return signal
