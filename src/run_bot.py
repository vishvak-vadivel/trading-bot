# execute_bot.py
import pandas as pd
import datetime as dt
import time
from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, PAPER_BASE_URL
import alpaca_trade_api as tradeapi
from sentiment import fetch_gnews_headlines, classify_sentiment
from ml import predict_trade_success, save_trade_history

api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, PAPER_BASE_URL, api_version="v2")

TICKERS = ["AAPL", "MSFT", "NVDA"]
HOLDING_DURATION = 5
ORDER_SIZE = 1

print("📦 Loading FinBERT model...")

def get_latest_price(symbol):
    try:
        bars = api.get_bars(symbol, timeframe="1Day", limit=5).df
        bars.index = pd.to_datetime(bars.index)
        now = pd.Timestamp.now(tz=bars.index.tz)
        bars = bars[bars.index <= now]  # allow today's bar if available
        print(bars)
        if bars.empty:
            return None, None
        return bars['close'].iloc[-1], bars.index[-1].date()
    except Exception as e:
        print(f"⚠️ Error fetching price for {symbol}: {e}")
        return None, None

def main():
    print("\n🚀 Executing trading bot...")
    today = dt.date.today()
    new_trades = []

    for symbol in TICKERS:
        print(f"\n🔍 Evaluating {symbol}...")

        price, price_date = get_latest_price(symbol)
        if price is None:
            print("⚠️ No recent price data.")
            continue

        headlines = fetch_gnews_headlines(symbol)
        sentiment = classify_sentiment(headlines)

        # Using 5-day dummy volatility until live prices are stored
        volatility = 0.02  # Placeholder, can be improved with rolling std dev from live data

        for signal in ["impulse", "momentum", "breakout"]:
            is_good = predict_trade_success(signal, volatility, sentiment, HOLDING_DURATION)
            if is_good:
                print(f"✅ Signal {signal.upper()} passed ML check for {symbol}. Placing paper trade.")
                try:
                    api.submit_order(
                        symbol=symbol,
                        qty=ORDER_SIZE,
                        side="buy",
                        type="market",
                        time_in_force="gtc"
                    )
                    new_trades.append({
                        "Entry Timestamp": dt.datetime.now(),
                        "Symbol": symbol,
                        "Signal": signal,
                        "Sentiment": sentiment,
                        "Price": price
                    })
                    break  # only one trade per symbol
                except Exception as e:
                    print(f"❌ Error placing order for {symbol}: {e}")

    if new_trades:
        df_new = pd.DataFrame(new_trades)
        save_trade_history(df_new)
        print(f"\n💾 Saved {len(df_new)} new trades to history.")
    else:
        print("\n📭 No trades placed today.")

if __name__ == "__main__":
    main()
