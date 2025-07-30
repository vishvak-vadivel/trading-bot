import os
import alpaca_trade_api as tradeapi
from config.config import load_config

cfg = load_config()

api = tradeapi.REST(
    cfg["api_key"],
    cfg["secret_key"],
    "https://api.alpaca.markets"
)

def get_price(symbol: str):
    barset = api.get_bars(symbol, "minute", limit=1)
    return barset[0].c if barset else None

def place_order(symbol: str, qty: int, side: str):
    try:
        api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type='market',
            time_in_force='gtc'
        )
        print(f"Placed {side} order for {qty} shares of {symbol}")
    except Exception as e:
        print("Order failed:", e)
