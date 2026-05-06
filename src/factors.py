"""
================================================================================
MODULE 2: FACTOR CONSTRUCTION
================================================================================
Factors implemented:
  1. Momentum (MOM)     — 12-month return skipping the most recent month (12-1).
                          Academic basis: Jegadeesh & Titman (1993).
                          High past return → expect high future return.

  2. Low Volatility (LVOL) — Trailing 12-month realized volatility of daily returns
                              (annualized). Academic basis: Ang et al. (2006).
                              Lower volatility → expect higher risk-adjusted return.

  3. Short-term Reversal (REV) — Prior 1-month return.
                                  Academic basis: Jegadeesh (1990).
                                  High 1-month return → expect reversal (negative signal).

Methodology Notes:
  - All signals are computed at end-of-month t using only data available at t.
    The portfolio is then held through month t+1. This ensures zero lookahead bias.
  - Factor values are cross-sectionally winsorized at the 1st/99th percentile
    each month to reduce the influence of extreme outliers.
  - A stock must have sufficient return history to receive a signal; otherwise
    its signal for that month is NaN and it is excluded from the portfolio.
================================================================================
"""

import numpy as np
import pandas as pd


# ── Utilities ─────────────────────────────────────────────────────────────────

def winsorize(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """Clip a cross-sectional series at given quantiles to reduce outlier impact."""
    lo = series.quantile(lower)
    hi = series.quantile(upper)
    return series.clip(lo, hi)


def apply_monthly_winsorize(factor: pd.DataFrame) -> pd.DataFrame:
    """Apply winsorization row-wise (cross-sectionally, month by month)."""
    return factor.apply(winsorize, axis=1)


# ── Factor 1: Momentum (12-1) ─────────────────────────────────────────────────

def compute_momentum(monthly_returns: pd.DataFrame) -> pd.DataFrame:
    """
    12-1 momentum: cumulative return over months t-12 through t-2,
    skipping month t-1 to avoid short-term reversal contamination.

    Signal at month t is the compounded return from t-12 to t-2 (11 months).
    Requires at least 12 months of return history.

    Parameters
    ----------
    monthly_returns : (months × tickers) simple monthly returns

    Returns
    -------
    momentum_signal : (months × tickers) factor values, NaN where insufficient data
    """
    # (1 + r) cumulative product over [t-12, t-2], then subtract 1
    log_ret = np.log1p(monthly_returns)

    # Rolling sum of log returns for months t-12 to t-2:
    #   12-month rolling sum, then subtract the most recent month
    rolling_12 = log_ret.rolling(window=12, min_periods=12).sum()
    signal = np.expm1(rolling_12 - log_ret)  # remove month t-1

    return apply_monthly_winsorize(signal)


# ── Factor 2: Low Volatility ──────────────────────────────────────────────────

def compute_low_volatility(daily_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Trailing 12-month realized daily return volatility, annualized.
    We compute this on a daily frequency then resample to month-end.

    A lower value is a stronger signal (less volatile stocks outperform).
    We negate the output so that higher signal = more desirable (for
    consistent quintile ranking: Q5 = best, Q1 = worst).

    Parameters
    ----------
    daily_returns : (days × tickers) daily simple returns

    Returns
    -------
    lvol_signal : (months × tickers) factor values (negated volatility)
    """
    TRADING_DAYS = 252
    WINDOW_DAYS  = 252  # approx 12 months of trading days

    daily_vol = (
        daily_returns
        .rolling(window=WINDOW_DAYS, min_periods=int(WINDOW_DAYS * 0.8))
        .std()
        * np.sqrt(TRADING_DAYS)
    )

    # Resample to month-end (take last value of each month)
    monthly_vol = daily_vol.resample("ME").last()

    # Negate so that high signal = low volatility (consistent ranking direction)
    signal = -monthly_vol

    return apply_monthly_winsorize(signal)


# ── Factor 3: Short-term Reversal ─────────────────────────────────────────────

def compute_reversal(monthly_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Prior 1-month return, negated (high past return → low expected return).
    Academic literature documents a reversal over 1-month horizon.

    Parameters
    ----------
    monthly_returns : (months × tickers) simple monthly returns

    Returns
    -------
    reversal_signal : (months × tickers) factor values (negated 1-month return)
    """
    signal = -monthly_returns  # negate: past losers expected to gain
    return apply_monthly_winsorize(signal)


# ── Main Entry Point ──────────────────────────────────────────────────────────

def build_factors(
    monthly_returns: pd.DataFrame,
    daily_returns:   pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """
    Construct all factor signals and return them in a labeled dictionary.

    Returns
    -------
    factors : dict mapping factor name → (months × tickers) signal DataFrame
    """
    print("[Factors] Computing momentum (12-1)...")
    mom = compute_momentum(monthly_returns)

    print("[Factors] Computing low volatility (12-month realized vol)...")
    lvol = compute_low_volatility(daily_returns)

    # Align LVOL index to monthly_returns index (in case of minor date mismatches)
    lvol = lvol.reindex(monthly_returns.index)
    lvol = lvol.reindex(columns=monthly_returns.columns)

    print("[Factors] Computing short-term reversal (1-month)...")
    rev = compute_reversal(monthly_returns)

    factors = {
        "Momentum":   mom,
        "LowVol":     lvol,
        "Reversal":   rev,
    }

    for name, df in factors.items():
        coverage = df.notna().sum(axis=1).mean()
        print(f"[Factors]   {name}: avg {coverage:.0f} stocks with signal per month.")

    return factors