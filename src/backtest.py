"""
================================================================================
MODULE 3: BACKTESTING ENGINE
================================================================================
Methodology:
  - At the end of each month t, rank all stocks with a valid signal into quintiles.
  - Form equal-weighted portfolios for each quintile.
  - Hold each portfolio through month t+1, then rebalance.
  - The long-short portfolio is long Q5 (top quintile) and short Q1 (bottom quintile).

Portfolio Formation Rules:
  - Signal is computed at end of month t using only data available at t.
  - Return is measured during month t+1.
  - This one-period forward shift is the core lookahead-bias prevention step.
  - Stocks with NaN signals are excluded from that month's ranking.
  - Minimum 50 stocks with valid signals required to form a portfolio in a given month.

Performance Metrics:
  - Annualized return
  - Annualized Sharpe ratio (assumes risk-free rate = 0 for simplicity)
  - Maximum drawdown
  - Monthly turnover (average fraction of portfolio replaced each month)
  - Hit rate (fraction of months the long-short portfolio is positive)

Transaction Cost Sensitivity:
  - We compute net returns after applying a one-way transaction cost assumption.
  - Default: 10 basis points (0.10%) per side, i.e. 20bps round-trip.
  - This is conservative for large-cap S&P 500 stocks in modern markets.
================================================================================
"""

import numpy as np
import pandas as pd

# ── Constants ─────────────────────────────────────────────────────────────────

N_QUINTILES    = 5
MIN_STOCKS     = 50      # minimum valid signals per month to form portfolios
TRADING_MONTHS = 12
ONE_WAY_COST   = 0.0010  # 10 bps per side


# ── Step 1: Quintile Ranking ──────────────────────────────────────────────────

def rank_to_quintiles(signal_row: pd.Series) -> pd.Series:
    """
    Assign each stock with a valid signal to quintile 1–5 (1=lowest, 5=highest).
    NaN signals get NaN quintile.
    """
    valid = signal_row.dropna()
    if len(valid) < MIN_STOCKS:
        return pd.Series(np.nan, index=signal_row.index)

    labels = pd.qcut(valid, q=N_QUINTILES, labels=False, duplicates="drop") + 1
    return labels.reindex(signal_row.index)


def build_quintile_matrix(signal: pd.DataFrame) -> pd.DataFrame:
    """Apply quintile ranking cross-sectionally for every month."""
    return signal.apply(rank_to_quintiles, axis=1)


# ── Step 2: Portfolio Returns ─────────────────────────────────────────────────

def compute_quintile_returns(
    quintile_matrix: pd.DataFrame,
    monthly_returns: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each month t, use quintile assignments from t to compute equal-weighted
    portfolio returns during month t+1.

    The forward shift (shift(1)) is the mechanism that prevents lookahead bias:
    signals from month t can only invest starting at t+1.

    Returns
    -------
    quintile_returns : DataFrame (months × 5), each column = one quintile's return
    """
    # Shift signals forward by 1 month: quintile known at t, return earned at t+1
    shifted_quintiles = quintile_matrix.shift(1)

    results = {}
    for q in range(1, N_QUINTILES + 1):
        mask = shifted_quintiles == q
        # Equal-weighted return: mean of returns for stocks in this quintile
        q_ret = monthly_returns[mask].mean(axis=1)
        results[f"Q{q}"] = q_ret

    df = pd.DataFrame(results)
    df.index = monthly_returns.index
    return df


# ── Step 3: Long-Short Portfolio ──────────────────────────────────────────────

def compute_long_short(quintile_returns: pd.DataFrame) -> pd.Series:
    """
    Long Q5 (highest signal), short Q1 (lowest signal).
    Returns the monthly long-short spread.
    """
    return quintile_returns["Q5"] - quintile_returns["Q1"]


# ── Step 4: Transaction Costs ────────────────────────────────────────────────

def compute_turnover(quintile_matrix: pd.DataFrame, quintile: int) -> pd.Series:
    """
    Estimate monthly one-way turnover for a given quintile portfolio.
    Turnover = fraction of stocks that entered or exited the quintile this month.
    """
    in_quintile = (quintile_matrix == quintile).astype(float)
    entering    = (in_quintile.diff().clip(lower=0)).sum(axis=1)
    n_stocks    = in_quintile.sum(axis=1).replace(0, np.nan)
    return entering / n_stocks


def apply_transaction_costs(
    quintile_returns:  pd.DataFrame,
    quintile_matrix:   pd.DataFrame,
    one_way_cost:      float = ONE_WAY_COST,
) -> pd.DataFrame:
    """
    Subtract transaction costs from each quintile's returns.
    Cost = one_way_cost × turnover (entering + exiting = 2 sides, but we
    capture one-way here since both long and short sides incur costs).
    """
    net = quintile_returns.copy()
    for q in range(1, N_QUINTILES + 1):
        turnover     = compute_turnover(quintile_matrix, q)
        cost_per_mo  = turnover * one_way_cost * 2  # round-trip
        net[f"Q{q}"] = quintile_returns[f"Q{q}"] - cost_per_mo
    return net


# ── Step 5: Performance Metrics ──────────────────────────────────────────────

def sharpe_ratio(returns: pd.Series, periods_per_year: int = 12) -> float:
    """Annualized Sharpe ratio assuming zero risk-free rate."""
    if returns.std() == 0:
        return np.nan
    return (returns.mean() / returns.std()) * np.sqrt(periods_per_year)


def annualized_return(returns: pd.Series, periods_per_year: int = 12) -> float:
    """Compound annualized growth rate from monthly returns."""
    cumulative = (1 + returns.dropna()).prod()
    n_years    = len(returns.dropna()) / periods_per_year
    if n_years == 0:
        return np.nan
    return cumulative ** (1 / n_years) - 1


def max_drawdown(returns: pd.Series) -> float:
    """Maximum peak-to-trough drawdown of cumulative return series."""
    cum    = (1 + returns.dropna()).cumprod()
    peak   = cum.cummax()
    dd     = (cum - peak) / peak
    return dd.min()


def hit_rate(returns: pd.Series) -> float:
    """Fraction of months with positive return."""
    valid = returns.dropna()
    return (valid > 0).sum() / len(valid)


def summarize_performance(returns: pd.Series, label: str = "") -> dict:
    """Compute and return all performance metrics for a return series."""
    return {
        "Label":             label,
        "Ann. Return":       f"{annualized_return(returns):.2%}",
        "Sharpe Ratio":      f"{sharpe_ratio(returns):.2f}",
        "Max Drawdown":      f"{max_drawdown(returns):.2%}",
        "Hit Rate":          f"{hit_rate(returns):.2%}",
        "Monthly Obs":       int(returns.notna().sum()),
    }


# ── Main Entry Point ──────────────────────────────────────────────────────────

def run_backtest(
    signal:          pd.DataFrame,
    monthly_returns: pd.DataFrame,
    factor_name:     str = "Factor",
    cost_adjust:     bool = True,
) -> dict:
    """
    Full backtest for a single factor signal.

    Parameters
    ----------
    signal          : (months × tickers) factor signal DataFrame
    monthly_returns : (months × tickers) simple monthly returns
    factor_name     : label used in output
    cost_adjust     : if True, subtract estimated transaction costs

    Returns
    -------
    results : dict containing quintile returns, long-short series, and metrics table
    """
    print(f"\n[Backtest] Running backtest for: {factor_name}")

    # ── Align signal and returns to common index/columns ────────────────────
    common_idx  = signal.index.intersection(monthly_returns.index)
    common_cols = signal.columns.intersection(monthly_returns.columns)
    signal_     = signal.loc[common_idx, common_cols]
    returns_    = monthly_returns.loc[common_idx, common_cols]

    # ── Quintile ranking ─────────────────────────────────────────────────────
    print(f"[Backtest]   Ranking stocks into {N_QUINTILES} quintiles...")
    quintile_matrix = build_quintile_matrix(signal_)

    # ── Quintile portfolio returns ────────────────────────────────────────────
    print("[Backtest]   Computing quintile portfolio returns...")
    quintile_returns_gross = compute_quintile_returns(quintile_matrix, returns_)

    # ── Transaction cost adjustment ───────────────────────────────────────────
    if cost_adjust:
        print(f"[Backtest]   Applying {ONE_WAY_COST*10000:.0f}bps one-way transaction costs...")
        quintile_returns_net = apply_transaction_costs(quintile_returns_gross, quintile_matrix)
    else:
        quintile_returns_net = quintile_returns_gross.copy()

    # ── Long-short portfolios ─────────────────────────────────────────────────
    ls_gross = compute_long_short(quintile_returns_gross)
    ls_net   = compute_long_short(quintile_returns_net)

    # ── Drop the first N months where signals are NaN (warm-up period) ───────
    first_valid = quintile_returns_gross.dropna(how="all").index[0]
    quintile_returns_gross = quintile_returns_gross.loc[first_valid:]
    quintile_returns_net   = quintile_returns_net.loc[first_valid:]
    ls_gross               = ls_gross.loc[first_valid:]
    ls_net                 = ls_net.loc[first_valid:]

    # ── Average turnover ──────────────────────────────────────────────────────
    q5_turnover = compute_turnover(quintile_matrix, 5)
    avg_turnover = q5_turnover.mean()

    # ── Performance summary ───────────────────────────────────────────────────
    metrics_rows = []
    for col in quintile_returns_gross.columns:
        metrics_rows.append(summarize_performance(quintile_returns_gross[col], label=col))
    metrics_rows.append(summarize_performance(ls_gross, label="L/S Gross"))
    metrics_rows.append(summarize_performance(ls_net,   label="L/S Net"))
    metrics_df = pd.DataFrame(metrics_rows).set_index("Label")

    print(f"[Backtest]   Avg Q5 monthly turnover: {avg_turnover:.1%}")
    print(f"\n{'='*60}")
    print(f"  Results — {factor_name}")
    print(f"{'='*60}")
    print(metrics_df.to_string())
    print(f"{'='*60}\n")

    return {
        "factor_name":        factor_name,
        "quintile_matrix":    quintile_matrix,
        "quintile_returns":   quintile_returns_gross,
        "quintile_returns_net": quintile_returns_net,
        "ls_gross":           ls_gross,
        "ls_net":             ls_net,
        "metrics":            metrics_df,
        "avg_turnover":       avg_turnover,
    }


def run_all_backtests(
    factors:         dict[str, pd.DataFrame],
    monthly_returns: pd.DataFrame,
) -> dict[str, dict]:
    """Run backtest for each factor and return all results."""
    all_results = {}
    for name, signal in factors.items():
        all_results[name] = run_backtest(signal, monthly_returns, factor_name=name)
    return all_results