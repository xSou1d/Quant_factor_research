"""
================================================================================
QUANTITATIVE FACTOR RESEARCH STUDY
================================================================================
Entry point. Runs the full pipeline end-to-end:

  Section 1 — Data Pipeline
    Downloads and cleans daily price data for S&P 500 stocks.
    Computes monthly prices, monthly returns, and daily returns.
    Results are cached to disk so subsequent runs skip the download.

  Section 2 — Factor Construction
    Builds three factor signals from the return data:
      • Momentum    (12-1 month prior return)
      • Low Vol     (negated trailing 12-month realized volatility)
      • Reversal    (negated prior 1-month return)

  Section 3 — Backtesting
    Ranks stocks into quintiles each month, forms equal-weighted portfolios,
    and measures next-month returns. Computes gross and net performance metrics.

  Section 4 — Analysis & Visualization
    Generates charts and a summary performance table.
    All output saved to output/.

Usage:
    python main.py                    # run with defaults
    python main.py --refresh          # force re-download of data
    python main.py --start 2010-01-01 --end 2023-12-31

Known Limitations (document for intellectual honesty):
  1. Survivorship bias: we use today's S&P 500 list, not historical constituents.
     This biases returns upward by excluding companies that were delisted or removed.
  2. Point-in-time fundamentals: the value factor is omitted because yfinance
     fundamental data is not point-in-time safe. Stale book values would
     introduce lookahead bias.
  3. Price impact: we assume we can trade at month-end closing prices with no
     slippage. In practice, rebalancing 100+ positions creates market impact.
  4. Risk-free rate: Sharpe ratios assume a 0% risk-free rate. Adjusting for
     prevailing T-bill rates would reduce reported Sharpe ratios.
================================================================================
"""

import argparse
import sys
import time

from src.pipeline  import build_pipeline
from src.factors   import build_factors
from src.backtest  import run_all_backtests
from src.analysis  import run_analysis, print_summary_table


# ── CLI Arguments ─────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Quantitative Factor Research Study"
    )
    parser.add_argument(
        "--start", default="2005-01-01",
        help="Start date for data download (default: 2005-01-01)"
    )
    parser.add_argument(
        "--end", default="2024-12-31",
        help="End date for data download (default: 2024-12-31)"
    )
    parser.add_argument(
        "--refresh", action="store_true",
        help="Force re-download of data even if cache exists"
    )
    parser.add_argument(
        "--no-cost-adjust", action="store_true",
        help="Skip transaction cost adjustment"
    )
    return parser.parse_args()


# ── Section Banner Helper ─────────────────────────────────────────────────────

def section(title: str, number: int) -> None:
    width = 65
    print(f"\n{'='*width}")
    print(f"  SECTION {number}: {title.upper()}")
    print(f"{'='*width}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    t0   = time.time()

    print("\n" + "="*65)
    print("  QUANTITATIVE FACTOR RESEARCH STUDY")
    print(f"  Universe : S&P 500 (current constituents — survivorship biased)")
    print(f"  Period   : {args.start} → {args.end}")
    print(f"  Factors  : Momentum (12-1), Low Volatility, Short-term Reversal")
    print("="*65)

    # ── Section 1: Data Pipeline ─────────────────────────────────────────────
    section("Data Pipeline", 1)
    monthly_prices, monthly_returns, daily_returns = build_pipeline(
        start=args.start,
        end=args.end,
        force_refresh=args.refresh,
    )

    # ── Section 2: Factor Construction ───────────────────────────────────────
    section("Factor Construction", 2)
    factors = build_factors(monthly_returns, daily_returns)

    # ── Section 3: Backtesting ────────────────────────────────────────────────
    section("Backtesting", 3)
    cost_adjust = not args.no_cost_adjust
    all_results = run_all_backtests(factors, monthly_returns)

    # ── Section 4: Analysis & Visualization ──────────────────────────────────
    section("Analysis & Visualization", 4)
    run_analysis(all_results)
    print_summary_table(all_results)

    elapsed = time.time() - t0
    print(f"[Done] Total runtime: {elapsed/60:.1f} minutes.\n")


if __name__ == "__main__":
    main()