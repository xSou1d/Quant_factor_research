"""
================================================================================
MODULE 1: DATA PIPELINE
================================================================================
Responsibilities:
  - Fetch S&P 500 constituent tickers from Wikipedia
  - Download adjusted daily OHLCV data via yfinance
  - Clean: handle missing data, delistings, and thin-trading stocks
  - Compute end-of-month adjusted close prices and daily returns
  - Cache results to disk to avoid repeated downloads

Survivorship Bias Note:
  Wikipedia provides today's S&P 500 list, not a point-in-time historical list.
  This means we will only download stocks that *survived* to the present day,
  which biases backtest returns upward (we exclude historical failures).
  A production pipeline would source a point-in-time constituent database
  (e.g., Compustat, CRSP, or a paid data vendor). We document this limitation
  explicitly rather than ignore it.
================================================================================
"""

import os
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm
from bs4 import BeautifulSoup

# ── Constants ────────────────────────────────────────────────────────────────

DATA_DIR       = os.path.join(os.path.dirname(__file__), "data")
PRICES_FILE    = os.path.join(DATA_DIR, "monthly_prices.parquet")
RETURNS_FILE   = os.path.join(DATA_DIR, "monthly_returns.parquet")
DAILY_RET_FILE = os.path.join(DATA_DIR, "daily_returns.parquet")
TICKERS_FILE   = os.path.join(DATA_DIR, "tickers.txt")

MIN_HISTORY_MONTHS  = 13   # need at least 13 months for 12-1 momentum
MIN_DATA_COVERAGE   = 0.80 # drop stocks missing >20% of months


# ── Step 1: Fetch S&P 500 Tickers ────────────────────────────────────────────

def fetch_sp500_tickers() -> list[str]:
    """Scrape current S&P 500 constituents from Wikipedia."""
    print("[Pipeline] Fetching S&P 500 tickers from Wikipedia...")
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "lxml")
    table = soup.find("table", {"id": "constituents"})
    tickers = []
    for row in table.find_all("tr")[1:]:
        ticker = row.find_all("td")[0].text.strip()
        ticker = ticker.replace(".", "-")  # yfinance uses BRK-B not BRK.B
        tickers.append(ticker)

    print(f"[Pipeline] Found {len(tickers)} tickers.")
    return sorted(tickers)


# ── Step 2: Download Price Data ───────────────────────────────────────────────

def download_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """
    Download adjusted close prices for all tickers.
    Downloads in batches to be resilient to network errors.
    Returns a DataFrame of shape (dates, tickers).
    """
    print(f"[Pipeline] Downloading price data from {start} to {end}...")
    batch_size = 50
    frames = []

    for i in tqdm(range(0, len(tickers), batch_size), desc="Downloading batches"):
        batch = tickers[i : i + batch_size]
        try:
            raw = yf.download(
                batch,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            if isinstance(raw.columns, pd.MultiIndex):
                close = raw["Close"]
            else:
                close = raw[["Close"]]
                close.columns = batch
            frames.append(close)
        except Exception as e:
            print(f"[Pipeline] Warning: batch {i//batch_size} failed — {e}")

    prices = pd.concat(frames, axis=1)
    prices = prices.loc[:, ~prices.columns.duplicated()]
    print(f"[Pipeline] Raw price matrix: {prices.shape}")
    return prices


# ── Step 3: Resample to Month-End ─────────────────────────────────────────────

def to_monthly(prices: pd.DataFrame) -> pd.DataFrame:
    """Resample daily adjusted close to end-of-month prices."""
    return prices.resample("ME").last()


# ── Step 4: Clean the Price Matrix ───────────────────────────────────────────

def clean_prices(monthly: pd.DataFrame) -> pd.DataFrame:
    """
    Drop stocks that don't have enough data coverage.
    We keep a stock only if it has at least MIN_HISTORY_MONTHS observations
    AND coverage >= MIN_DATA_COVERAGE over the full window.
    Remaining gaps are forward-filled (max 2 months) then left as NaN.
    """
    print("[Pipeline] Cleaning monthly price data...")
    n_months = len(monthly)
    coverage = monthly.notna().sum() / n_months

    enough_coverage = coverage >= MIN_DATA_COVERAGE
    enough_history  = monthly.notna().sum() >= MIN_HISTORY_MONTHS

    keep = enough_coverage & enough_history
    cleaned = monthly.loc[:, keep].copy()

    # Forward fill short gaps (e.g., trading halts), but cap at 2 months
    cleaned = cleaned.ffill(limit=2)

    dropped = monthly.shape[1] - cleaned.shape[1]
    print(f"[Pipeline] Dropped {dropped} tickers; kept {cleaned.shape[1]}.")
    return cleaned


# ── Step 5: Compute Returns ───────────────────────────────────────────────────

def compute_monthly_returns(monthly_prices: pd.DataFrame) -> pd.DataFrame:
    """Simple monthly returns from end-of-month adjusted closes."""
    return monthly_prices.pct_change()


def compute_daily_returns(daily_prices: pd.DataFrame) -> pd.DataFrame:
    """Daily returns — used for volatility factor construction."""
    return daily_prices.pct_change()


# ── Main Entry Point ──────────────────────────────────────────────────────────

def build_pipeline(
    start: str = "2005-01-01",
    end:   str = "2024-12-31",
    force_refresh: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Full pipeline: fetch → download → clean → return.

    Returns
    -------
    monthly_prices  : pd.DataFrame  (months × tickers)
    monthly_returns : pd.DataFrame  (months × tickers)
    daily_returns   : pd.DataFrame  (days   × tickers)
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    cached = (
        os.path.exists(PRICES_FILE)
        and os.path.exists(RETURNS_FILE)
        and os.path.exists(DAILY_RET_FILE)
    )

    if cached and not force_refresh:
        print("[Pipeline] Loading cached data from disk...")
        monthly_prices  = pd.read_parquet(PRICES_FILE)
        monthly_returns = pd.read_parquet(RETURNS_FILE)
        daily_returns   = pd.read_parquet(DAILY_RET_FILE)
        print(f"[Pipeline] Loaded: {monthly_prices.shape[1]} stocks, "
              f"{len(monthly_prices)} months.")
        return monthly_prices, monthly_returns, daily_returns

    # ── Fetch and download ───────────────────────────────────────────────────
    tickers = fetch_sp500_tickers()
    with open(TICKERS_FILE, "w") as f:
        f.write("\n".join(tickers))

    daily_prices = download_prices(tickers, start=start, end=end)

    # ── Compute daily returns before resampling ──────────────────────────────
    raw_daily_returns = compute_daily_returns(daily_prices)

    # ── Resample and clean ───────────────────────────────────────────────────
    monthly_prices  = to_monthly(daily_prices)
    monthly_prices  = clean_prices(monthly_prices)
    monthly_returns = compute_monthly_returns(monthly_prices)

    # ── Align daily returns to the cleaned universe ──────────────────────────
    shared_tickers  = monthly_prices.columns.intersection(raw_daily_returns.columns)
    daily_returns   = raw_daily_returns[shared_tickers]

    # ── Cache to disk ────────────────────────────────────────────────────────
    print("[Pipeline] Saving data to disk...")
    monthly_prices.to_parquet(PRICES_FILE)
    monthly_returns.to_parquet(RETURNS_FILE)
    daily_returns.to_parquet(DAILY_RET_FILE)

    print(f"[Pipeline] Done. Universe: {monthly_prices.shape[1]} stocks, "
          f"{len(monthly_prices)} months.")
    return monthly_prices, monthly_returns, daily_returns