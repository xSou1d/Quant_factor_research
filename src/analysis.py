"""
================================================================================
MODULE 4: ANALYSIS & VISUALIZATION
================================================================================
Charts produced (one set per factor + one combined):
  1. Cumulative returns — Q1 through Q5 + long-short (gross and net of costs)
  2. Quintile bar chart — average annualized return per quintile (monotonicity test)
  3. Drawdown chart — underwater equity curve for the long-short portfolio
  4. Rolling Sharpe — 36-month rolling Sharpe ratio of long-short portfolio
  5. Factor correlation — heatmap of pairwise long-short return correlations

All charts are saved to the output/ directory as PNG files.
================================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

# ── Style ─────────────────────────────────────────────────────────────────────

PALETTE = {
    "Q1": "#d62728",   # red    — worst quintile
    "Q2": "#ff7f0e",   # orange
    "Q3": "#aec7e8",   # light blue
    "Q4": "#1f77b4",   # blue
    "Q5": "#2ca02c",   # green  — best quintile
    "LS_gross": "#9467bd",  # purple
    "LS_net":   "#8c564b",  # brown
}

plt.rcParams.update({
    "figure.dpi":      150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size":       10,
})


def _save(fig: plt.Figure, filename: str) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Analysis] Saved: {path}")


# ── Chart 1: Cumulative Returns ───────────────────────────────────────────────

def plot_cumulative_returns(
    quintile_returns: pd.DataFrame,
    ls_gross:         pd.Series,
    ls_net:           pd.Series,
    factor_name:      str,
) -> None:
    """
    Plot cumulative wealth index (starting at $1) for each quintile
    and for the long-short portfolio (gross and net of transaction costs).
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), gridspec_kw={"height_ratios": [2, 1]})

    ax_cum, ax_ls = axes

    # ── Top panel: all quintiles ─────────────────────────────────────────────
    for col in quintile_returns.columns:
        cum = (1 + quintile_returns[col].dropna()).cumprod()
        ax_cum.plot(cum.index, cum.values, label=col,
                    color=PALETTE.get(col, "grey"), linewidth=1.5)

    ax_cum.set_title(f"{factor_name} — Quintile Cumulative Returns ($1 start)",
                     fontsize=13, fontweight="bold")
    ax_cum.set_ylabel("Portfolio Value ($)")
    ax_cum.legend(loc="upper left", ncol=5, fontsize=8)
    ax_cum.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.2f"))
    ax_cum.set_xlabel("")

    # ── Bottom panel: long-short only (gross vs net) ─────────────────────────
    for series, label, color in [
        (ls_gross, "L/S Gross", PALETTE["LS_gross"]),
        (ls_net,   "L/S Net",   PALETTE["LS_net"]),
    ]:
        cum = (1 + series.dropna()).cumprod()
        ax_ls.plot(cum.index, cum.values, label=label, color=color, linewidth=1.8)

    ax_ls.axhline(1.0, color="grey", linewidth=0.8, linestyle="--")
    ax_ls.set_title("Long-Short Portfolio: Gross vs Net of Transaction Costs",
                    fontsize=11, fontweight="bold")
    ax_ls.set_ylabel("Portfolio Value ($)")
    ax_ls.set_xlabel("Date")
    ax_ls.legend(loc="upper left", fontsize=9)
    ax_ls.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.2f"))

    fig.tight_layout(pad=2.0)
    _save(fig, f"{factor_name}_cumulative_returns.png")


# ── Chart 2: Quintile Return Bar Chart ───────────────────────────────────────

def plot_quintile_bar(
    quintile_returns: pd.DataFrame,
    factor_name:      str,
) -> None:
    """
    Bar chart of average annualized return per quintile.
    A monotonically increasing pattern (Q1 < Q2 < ... < Q5) is the key
    test of factor validity — not just that the long-short spread is positive.
    """
    ann_returns = []
    for col in quintile_returns.columns:
        r = quintile_returns[col].dropna()
        ann = (1 + r).prod() ** (12 / len(r)) - 1
        ann_returns.append(ann * 100)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [PALETTE.get(f"Q{i+1}", "grey") for i in range(len(ann_returns))]
    bars = ax.bar(quintile_returns.columns, ann_returns, color=colors,
                  edgecolor="white", linewidth=0.8)

    for bar, val in zip(bars, ann_returns):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (0.3 if val >= 0 else -1.0),
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"{factor_name} — Annualized Return by Quintile",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Annualized Return (%)")
    ax.set_xlabel("Quintile (1=Lowest Signal, 5=Highest Signal)")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=1))

    fig.tight_layout()
    _save(fig, f"{factor_name}_quintile_bar.png")


# ── Chart 3: Drawdown ─────────────────────────────────────────────────────────

def plot_drawdown(ls_gross: pd.Series, ls_net: pd.Series, factor_name: str) -> None:
    """
    Underwater chart: shows how far the portfolio is below its prior high-water mark.
    Helps identify prolonged periods of underperformance.
    """
    def drawdown_series(returns: pd.Series) -> pd.Series:
        cum  = (1 + returns.dropna()).cumprod()
        peak = cum.cummax()
        return (cum - peak) / peak * 100

    fig, ax = plt.subplots(figsize=(12, 5))

    for series, label, color in [
        (ls_gross, "Gross", PALETTE["LS_gross"]),
        (ls_net,   "Net",   PALETTE["LS_net"]),
    ]:
        dd = drawdown_series(series)
        ax.fill_between(dd.index, dd.values, 0,
                        alpha=0.3, color=color, label=f"{label} Drawdown")
        ax.plot(dd.index, dd.values, color=color, linewidth=1.0)

    ax.set_title(f"{factor_name} — Long-Short Portfolio Drawdown",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Drawdown (%)")
    ax.set_xlabel("Date")
    ax.legend(loc="lower left", fontsize=9)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=1))

    fig.tight_layout()
    _save(fig, f"{factor_name}_drawdown.png")


# ── Chart 4: Rolling Sharpe ───────────────────────────────────────────────────

def plot_rolling_sharpe(ls_gross: pd.Series, factor_name: str, window: int = 36) -> None:
    """
    36-month rolling Sharpe ratio of the long-short portfolio.
    Periods below zero indicate sustained underperformance of the factor.
    """
    rolling_sharpe = (
        ls_gross.dropna()
        .rolling(window)
        .apply(lambda x: (x.mean() / x.std()) * np.sqrt(12) if x.std() > 0 else np.nan)
    )

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(rolling_sharpe.index, rolling_sharpe.values,
            color=PALETTE["LS_gross"], linewidth=1.5, label=f"{window}m Rolling Sharpe")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.axhline(0.5, color="green", linewidth=0.6, linestyle=":", alpha=0.7,
               label="Sharpe = 0.5")
    ax.fill_between(rolling_sharpe.index, rolling_sharpe.values, 0,
                    where=rolling_sharpe.values >= 0, alpha=0.15, color="green")
    ax.fill_between(rolling_sharpe.index, rolling_sharpe.values, 0,
                    where=rolling_sharpe.values < 0, alpha=0.15, color="red")

    ax.set_title(f"{factor_name} — {window}-Month Rolling Sharpe Ratio (Long-Short)",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_xlabel("Date")
    ax.legend(fontsize=9)

    fig.tight_layout()
    _save(fig, f"{factor_name}_rolling_sharpe.png")


# ── Chart 5: Factor Correlation Heatmap ──────────────────────────────────────

def plot_factor_correlation(all_results: dict) -> None:
    """
    Heatmap of pairwise Pearson correlations between long-short return series.
    Low correlation between factors is desirable — it means they capture
    distinct sources of return and combine well in a multi-factor portfolio.
    """
    ls_series = {}
    for name, res in all_results.items():
        ls_series[name] = res["ls_gross"]

    corr_df = pd.DataFrame(ls_series).corr()

    fig, ax = plt.subplots(figsize=(6, 5))
    mask = np.triu(np.ones_like(corr_df, dtype=bool), k=1)
    sns.heatmap(
        corr_df,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=-1, vmax=1,
        center=0,
        linewidths=0.5,
        ax=ax,
        square=True,
    )
    ax.set_title("Factor Long-Short Return Correlations",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "factor_correlation_heatmap.png")


# ── Chart 6: Combined Summary Dashboard ──────────────────────────────────────

def plot_combined_ls(all_results: dict) -> None:
    """
    Single chart comparing the cumulative long-short returns of all factors.
    Makes it easy to see which factor was strongest over the test period.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#ff7f0e"]

    for (name, res), color in zip(all_results.items(), colors):
        cum = (1 + res["ls_gross"].dropna()).cumprod()
        ax.plot(cum.index, cum.values, label=f"{name} L/S",
                color=color, linewidth=1.8)

    ax.axhline(1.0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_title("All Factors — Long-Short Cumulative Returns (Gross)",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Portfolio Value ($)")
    ax.set_xlabel("Date")
    ax.legend(fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.2f"))

    fig.tight_layout()
    _save(fig, "all_factors_combined.png")


# ── Main Entry Point ──────────────────────────────────────────────────────────

def run_analysis(all_results: dict) -> None:
    """
    Generate all charts for every factor and produce the combined view.

    Parameters
    ----------
    all_results : dict mapping factor_name → backtest results dict
    """
    print("\n[Analysis] Generating charts...")

    for name, res in all_results.items():
        print(f"[Analysis] Plotting {name}...")
        plot_cumulative_returns(
            res["quintile_returns"],
            res["ls_gross"],
            res["ls_net"],
            factor_name=name,
        )
        plot_quintile_bar(res["quintile_returns"], factor_name=name)
        plot_drawdown(res["ls_gross"], res["ls_net"], factor_name=name)
        plot_rolling_sharpe(res["ls_gross"], factor_name=name)

    print("[Analysis] Plotting cross-factor charts...")
    plot_factor_correlation(all_results)
    plot_combined_ls(all_results)

    print(f"\n[Analysis] All charts saved to: {os.path.abspath(OUTPUT_DIR)}")


# ── Summary Table ─────────────────────────────────────────────────────────────

def print_summary_table(all_results: dict) -> None:
    """Print a consolidated performance table across all factors."""
    rows = []
    for name, res in all_results.items():
        ls = res["ls_gross"]
        rows.append({
            "Factor":        name,
            "Ann. Return":   f"{(1+ls.dropna()).prod()**(12/len(ls.dropna()))-1:.2%}",
            "Sharpe":        f"{(ls.mean()/ls.std())*np.sqrt(12):.2f}",
            "Max Drawdown":  f"{((1+ls.dropna()).cumprod() / (1+ls.dropna()).cumprod().cummax() - 1).min():.2%}",
            "Hit Rate":      f"{(ls.dropna()>0).mean():.2%}",
            "Avg Turnover":  f"{res['avg_turnover']:.1%}",
        })

    summary = pd.DataFrame(rows).set_index("Factor")
    print("\n" + "="*65)
    print("  FACTOR PERFORMANCE SUMMARY (Long-Short, Gross of Costs)")
    print("="*65)
    print(summary.to_string())
    print("="*65 + "\n")