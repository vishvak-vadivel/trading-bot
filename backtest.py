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
print("⏳ Starting backtest with improved signal set...")

# Connect to Alpaca
api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, BASE_URL)

# Tickers
tickers = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL",
    "META", "JPM", "XOM", "UNH", "TSLA",
    "BRK.B", "V", "JNJ", "PG", "MA",
    "HD", "PEP", "KO", "AVGO", "COST",
    "LLY", "MRK", "ADBE", "BAC", "CRM",
    "NFLX", "ORCL", "INTC", "T", "WMT"
]

market = "SPY"
start_date = "2023-01-01"
end_date = (dt.datetime.today() - dt.timedelta(days=1)).strftime("%Y-%m-%d")

# Fetch price data (Daily)
def get_daily_close(symbol):
    bars = api.get_bars(symbol, timeframe="1Day", start=start_date, end=end_date).df
    return bars['close'].rename(symbol)

print("📡 Fetching daily price data...")
# Load price data for tickers + market
df = pd.concat([get_daily_close(t) for t in tickers + [market]], axis=1).dropna()
price = df[tickers]

# Use full ticker universe for maximum opportunity
print("🔍 Using full ticker universe for maximum opportunity")
selected_tickers = tickers
price = price[selected_tickers]
print(f"📈 Selected tickers: {selected_tickers}")

# Strategy logic
print("⚙️ Generating signals with loosened criteria...")

# Momentum: 3-day > 7-day MA for faster signals
short_ma = price.rolling(3).mean()
long_ma = price.rolling(7).mean()
momentum = short_ma > long_ma

# Breakout: close > 3-day high (more frequent)
breakout_high = price.rolling(3).max()
breakout = price > breakout_high.shift(1)

# Impulse: >0.5% up-day with volatility filter >0.5%
volatility = price.pct_change().rolling(5).std()
impulse = (price.pct_change(1) > 0.005) & (volatility > 0.005)

# Entry condition: any signal fires immediately
entries = (momentum | breakout | impulse).fillna(False)
persistent_momentum = momentum & momentum.shift(1)
entries = (persistent_momentum | breakout | impulse).fillna(False)

# Track signals
signal_source = pd.DataFrame("", index=price.index, columns=price.columns)
signal_source[persistent_momentum] = "momentum"
signal_source[breakout] = "breakout"
signal_source[impulse] = "impulse"

# Position sizing: 2x leverage per entry
entry_weights = entries.astype(float) * 2

# Exit logic
sl_exits = pd.DataFrame(False, index=price.index, columns=price.columns)
tp_exits = pd.DataFrame(False, index=price.index, columns=price.columns)
downcross = short_ma < long_ma
ma_exits = downcross.fillna(False)

# ATR-based SL/TP with extended window
atr = price.pct_change().rolling(14).std() * price
for col in price.columns:
    for entry_date in entries[col][entries[col]].index:
        idx = price.index.get_loc(entry_date)
        if idx + 1 >= len(price): continue
        entry_price = price.at[entry_date, col]
        atr_val = atr.at[entry_date, col] if not pd.isna(atr.at[entry_date, col]) else entry_price * 0.02
        sl = entry_price - 2 * atr_val
        tp = entry_price + 4 * atr_val
        fwd = price[col].iloc[idx+1:idx+26]
        hit_sl = fwd[fwd <= sl]
        hit_tp = fwd[fwd >= tp]
        if not hit_sl.empty:
            sl_exits.at[hit_sl.index[0], col] = True
        elif not hit_tp.empty:
            tp_exits.at[hit_tp.index[0], col] = True

exits = (ma_exits | sl_exits | tp_exits).fillna(False)

# Debug stats
print("\n📊 Entry counts:")
print(entries.sum())
print("✅ Any entries?", entries.any().any())

# Backtest
print("\n📈 Running backtest...")
pf = vbt.Portfolio.from_signals(
    close=price,
    entries=entries,
    exits=exits,
    size=entry_weights,
    init_cash=100_000,
    direction='longonly',
    freq='1D',
    fees=0.001,
    slippage=0.001
)

# Results
print("\n📊 Backtest Results:")
print(pf.stats())
# Plot
# pf.plot().show()

# # Export trades
# df_trades = pf.trades.records_readable.copy()
# df_trades['Signal'] = df_trades.apply(
#     lambda r: signal_source.at[r['Entry Timestamp'], r['Column']]
#                   if r['Column'] in signal_source.columns and r['Entry Timestamp'] in signal_source.index else "unknown",
#     axis=1
# )
# df_trades.to_csv("trade_log_with_signals.csv", index=False)
# print("\n📁 Exported trade_log_with_signals.csv")

# print(f"\n✅ Finished in {time.time() - start:.2f}s.")
