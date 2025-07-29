import json

with open("../config.json") as f:
    creds = json.load(f)

ALPACA_API_KEY = creds["api_key"]
ALPACA_SECRET_KEY = creds["secret_key"]
BASE_URL = "https://api.alpaca.markets"
PAPER_BASE_URL= "https://paper-api.alpaca.markets"
GNEWS_API_KEY = creds.get("gnews_api_key")