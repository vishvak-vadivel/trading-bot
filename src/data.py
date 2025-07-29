import time
import pandas as pd
import alpaca_trade_api as tradeapi
from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, BASE_URL

api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, BASE_URL)

def get_daily_close(symbol, start_date, end_date):
    for attempt in range(3):
        try:
            bars = api.get_bars(symbol, timeframe="1Day", start=start_date, end=end_date).df
            if bars.empty:
                print(f"⚠️ No data for {symbol} between {start_date} and {end_date}.")
                return pd.Series(name=symbol)
            return bars['close'].rename(symbol)
        except Exception as e:
            print(f"⚠️ Attempt {attempt+1}/3 failed for {symbol}: {e}")
            time.sleep(1)
    raise RuntimeError(f"❌ Failed to fetch data for {symbol} after 3 attempts.")