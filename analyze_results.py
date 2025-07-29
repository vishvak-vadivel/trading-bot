"""
Trade Log Analyzer
===================

This script loads a CSV file containing trade records (such as those exported by
vectorbt) and computes summary statistics to help identify the best and worst
trades. It also breaks down performance by signal type if available.

Usage:
    python analyze_trade_log.py <path_to_trade_log_csv>

If no path is provided, the script defaults to 'trade_log_with_signals.csv' in the
current working directory.

The script prints:
    - Total number of trades, wins, losses, and win rate
    - Average return per trade and separately for winners and losers
    - Top N best trades by return
    - Bottom N worst trades by return
    - Average return by signal type (if a 'Signal' column is present)

Note: This analysis is purely informational and not financial advice.
"""

import sys
import pandas as pd

def load_trade_log(path: str) -> pd.DataFrame:
    """Load the trade log from a CSV file into a DataFrame."""
    df = pd.read_csv(path)
    return df

def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the DataFrame has a 'Return [%]' column. Compute it if missing."""
    if 'Return [%]' in df.columns:
        return df
    if 'Return' in df.columns:
        df['Return [%]'] = df['Return'] * 100
        return df

    # Try to compute from known price columns
    entry_col_candidates = ['Entry Price', 'Avg Entry Price', 'entry_price', 'entry']
    exit_col_candidates = ['Exit Price', 'Avg Exit Price', 'exit_price', 'exit']
    entry_col = next((col for col in entry_col_candidates if col in df.columns), None)
    exit_col = next((col for col in exit_col_candidates if col in df.columns), None)
    if entry_col and exit_col:
        df['Return [%]'] = (df[exit_col] - df[entry_col]) / df[entry_col] * 100
        return df

    if 'PnL' in df.columns and entry_col:
        df['Return [%]'] = df['PnL'] / df[entry_col] * 100
        return df

    print("⚠️ Could not compute 'Return [%]' because expected columns are missing.")
    print("    Trade log columns available: {}".format(list(df.columns)))
    print("    Ensure your CSV includes either 'Return', 'Entry/Exit Price', or 'PnL' and entry price.")
    return df


def print_summary_statistics(df: pd.DataFrame) -> None:
    """Print overall summary statistics for the trade log."""
    n_trades = len(df)
    print("\n===== Trade Log Summary =====")
    print(f"Total trades: {n_trades}")
    # If 'Return [%]' column is absent, we cannot compute return stats
    if 'Return [%]' not in df.columns:
        print("Return data is not available. Please ensure 'Return [%]' exists or can be computed.")
        return
    n_wins = (df['Return [%]'] > 0).sum()
    n_losses = (df['Return [%]'] <= 0).sum()
    win_rate = n_wins / n_trades * 100 if n_trades > 0 else 0
    avg_return = df['Return [%]'].mean() if n_trades > 0 else 0
    avg_win = df.loc[df['Return [%]'] > 0, 'Return [%]'].mean() if n_wins > 0 else 0
    avg_loss = df.loc[df['Return [%]'] <= 0, 'Return [%]'].mean() if n_losses > 0 else 0
    print(f"Winning trades: {n_wins}")
    print(f"Losing trades: {n_losses}")
    print(f"Win rate: {win_rate:.2f}%")
    print(f"Average return per trade: {avg_return:.2f}%")
    print(f"Average winning trade return: {avg_win:.2f}%")
    print(f"Average losing trade return: {avg_loss:.2f}%")

    # Best and worst trades
    n_top = min(10, n_trades)
    df_sorted = df.sort_values('Return [%]', ascending=False)
    top_trades = df_sorted.head(n_top)
    bottom_trades = df_sorted.tail(n_top)
    print(f"\nTop {n_top} trades by return:")
    cols = ['Entry Timestamp', 'Exit Timestamp', 'Column', 'Return [%]']
    if 'Signal' in df.columns:
        cols.append('Signal')
    print(top_trades[cols])
    print(f"\nBottom {n_top} trades by return:")
    print(bottom_trades[cols])

    # If Signal column exists, compute average return by signal
    if 'Signal' in df.columns:
        print("\nAverage return by signal:")
        print(df.groupby('Signal')['Return [%]'].mean())

def main(path: str) -> None:
    df = load_trade_log(path)
    df = compute_returns(df)
    print_summary_statistics(df)

if __name__ == '__main__':
    file_path = sys.argv[1] if len(sys.argv) > 1 else 'filtered_impulse_trade_log.csv'
    main(file_path)

