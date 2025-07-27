import os
import time
import pandas as pd
import datetime as dt
import numpy as np
import alpaca_trade_api as tradeapi
import vectorbt as vbt
import json

# Load Alpaca API credentials
with open("config.json") as f:
    creds = json.load(f)

ALPACA_API_KEY = creds["api_key"]
ALPACA_SECRET_KEY = creds["secret_key"]
BASE_URL = "https://api.alpaca.markets"

# Timer start
start = time.time()
print("⏳ Starting backtest with strong momentum trend-following setup...")

# Connect to Alpaca
api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, BASE_URL)

# Tickers
tickers = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL",
    "META", "JPM", "XOM", "UNH", "TSLA"
]

market = "SPY"
start_date = "2023-01-01"
end_date = (dt.datetime.today() - dt.timedelta(days=1)).strftime("%Y-%m-%d")

# Fetch price data (Daily)
def get_daily_close(symbol):
    bars = api.get_bars(symbol, timeframe="1Day", start=start_date, end=end_date).df
    return bars['close'].rename(symbol)

print("📡 Fetching daily price data...")
df = pd.concat([get_daily_close(t) for t in tickers + [market]], axis=1).dropna()
price = df[tickers]
spy = df[market]

# Strategy logic
print("⚙️ Generating signals...")

# ✅ Clean trend-following (3-day > 20-day MA for strong breakouts)
short_ma = price.rolling(3).mean()
long_ma = price.rolling(20).mean()
momentum = short_ma > long_ma

downcross = short_ma < long_ma

# ✅ Market filter: SPY above 50-day SMA
spy_sma_50 = spy.rolling(50).mean()
spy_trend = (spy > spy_sma_50).reindex_like(price).ffill().bfill()

# Broadcast SPY trend filter
spy_trend_df = pd.DataFrame(
    np.broadcast_to(spy_trend.values[:, np.newaxis], price.shape),
    index=price.index,
    columns=price.columns
)

# Entry condition
entries = (momentum & spy_trend_df).fillna(False).astype(bool)

# Initialize exit signals
sl_exits = pd.DataFrame(False, index=price.index, columns=price.columns)
tp_exits = pd.DataFrame(False, index=price.index, columns=price.columns)
ma_exits = downcross.fillna(False).astype(bool)

# Manual SL/TP logic
for ticker in tickers:
    entry_dates = entries[ticker][entries[ticker]].index
    for entry_date in entry_dates:
        if entry_date not in price.index:
            continue
        entry_idx = price.index.get_loc(entry_date)
        if entry_idx + 1 >= len(price):
            continue
        entry_price = price.at[entry_date, ticker]
        forward_prices = price[ticker].iloc[entry_idx + 1:entry_idx + 16]  # 15-day SL/TP window

        if forward_prices.empty:
            continue

        stop_hit = forward_prices[forward_prices <= entry_price * 0.98]
        if not stop_hit.empty:
            sl_exits.at[stop_hit.index[0], ticker] = True
            continue

        take_hit = forward_prices[forward_prices >= entry_price * 1.06]
        if not take_hit.empty:
            tp_exits.at[take_hit.index[0], ticker] = True

# Combine exits
exits = (ma_exits | sl_exits | tp_exits).fillna(False).astype(bool)

# Debug stats
print("\n📊 Entry counts:")
print(entries.sum())
print("✅ Any entries?", entries.any().any())

# Backtest
print("\n📈 Running backtest (with fees & slippage)...")
pf = vbt.Portfolio.from_signals(
    close=price,
    entries=entries,
    exits=exits,
    size=0.5,
    init_cash=100_000,
    direction='longonly',
    freq='1D',
    fees=0.001,
    slippage=0.001
)

# Show performance
print("\n📊 Backtest Results:")
print(pf.stats())

# Export trades
trades_df = pf.trades.records_readable
trades_df.to_csv("trade_log.csv", index=False)
print("\n📁 Exported trades to trade_log.csv")

# Done
print(f"\n✅ Finished in {time.time() - start:.2f} seconds.")
