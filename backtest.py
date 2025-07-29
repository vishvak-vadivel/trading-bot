import time
import datetime as dt
import pandas as pd
import numpy as np
import json
import alpaca_trade_api as tradeapi
import vectorbt as vbt
import requests
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

# 🔑 Load Alpaca API credentials
with open("config.json") as f:
    creds = json.load(f)

ALPACA_API_KEY = creds["api_key"]
ALPACA_SECRET_KEY = creds["secret_key"]
BASE_URL = "https://api.alpaca.markets"
GNEWS_API_KEY = creds.get("gnews_api_key")

# 👥 Connect to Alpaca
api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, BASE_URL)

# 📈 Universe and dates
tickers = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "JPM", "XOM", "UNH", "TSLA",
    "BRK.B", "V", "JNJ", "PG", "MA", "HD", "PEP", "KO", "AVGO", "COST",
    "LLY", "MRK", "ADBE", "BAC", "CRM", "NFLX", "ORCL", "INTC", "T", "WMT"
]
market = "SPY"
start_date = "2013-01-01"
end_date = (dt.datetime.today() - dt.timedelta(days=1)).strftime("%Y-%m-%d")

RETRAIN_INTERVAL_DAYS = 7
MODEL_META_PATH = "trade_classifier_meta.json"
TRADE_LOG_PATH = "trade_history.csv"
tight_sl_symbols = {"NVDA", "WMT"}

# 🧐 Load FinBERT model for financial sentiment analysis
print("📦 Loading FinBERT model...")
finbert_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
finbert_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

def should_retrain():
    try:
        with open(MODEL_META_PATH, "r") as f:
            meta = json.load(f)
        last_trained = pd.to_datetime(meta["last_trained"])
        return (dt.datetime.now() - last_trained).days >= RETRAIN_INTERVAL_DAYS
    except:
        return True

def save_model_meta():
    with open(MODEL_META_PATH, "w") as f:
        json.dump({"last_trained": dt.datetime.now().isoformat()}, f)

def fetch_gnews_headlines(ticker):
    print(f"🔍 Fetching news for {ticker}...")
    url = f"https://gnews.io/api/v4/search?q={ticker}&token={GNEWS_API_KEY}&lang=en&max=10"
    try:
        response = requests.get(url)
        articles = response.json().get("articles", [])
        return [a["title"] for a in articles]
    except Exception as e:
        print(f"⚠️ Failed to fetch news for {ticker}: {e}")
        return []

def classify_sentiment(headlines):
    if not headlines:
        return 0.0
    try:
        print(f"🧠 Analyzing sentiment...")
        inputs = finbert_tokenizer(headlines, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = finbert_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        scores = probs[:, 2] - probs[:, 0]
        return float(scores.mean())
    except Exception:
        return 0.0

def get_signal(row):
    ts = row['Entry Timestamp']
    col = row['Column']
    if col in signal_source.columns and ts in signal_source.index:
        return signal_source.at[ts, col] if signal_source.at[ts, col] != "" else "unknown"
    return "unknown"

def predict_trade_success(signal, volatility, sentiment, holding_duration):
    signal_impulse = int(signal == 'impulse')
    signal_momentum = int(signal == 'momentum')
    signal_breakout = int(signal == 'breakout')
    features = pd.DataFrame([[
        signal_impulse, signal_momentum, signal_breakout, volatility, sentiment, holding_duration
    ]], columns=[
        "signal_impulse", "signal_momentum", "signal_breakout", "volatility", "sentiment", "holding_duration"
    ])
    try:
        clf = joblib.load("trade_classifier.pkl")
        prob = clf.predict_proba(features)[0][1]
        return prob >= 0.55
    except Exception:
        return True

def train_trade_classifier(df_trades):
    df = df_trades.copy()
    if df.empty:
        print("⚠️ No trades to train on. Skipping ML training.")
        return None
    df['return_pct'] = df['PnL'] / df['Avg Entry Price']
    df['duration'] = (
        pd.to_datetime(df['Exit Timestamp']) - pd.to_datetime(df['Entry Timestamp'])
    ).dt.days.fillna(0)
    df['target'] = ((df['return_pct'] > 0.04) & (df['duration'] < 10)).astype(int)
    df['signal_impulse'] = (df['Signal'] == 'impulse').astype(int)
    df['signal_momentum'] = (df['Signal'] == 'momentum').astype(int)
    df['signal_breakout'] = (df['Signal'] == 'breakout').astype(int)
    df['volatility'] = df['Avg Entry Price'].rolling(5).std().fillna(0)
    df['sentiment'] = df.get('Sentiment', 0.0)
    df['holding_duration'] = df['duration']

    features = ['signal_impulse', 'signal_momentum', 'signal_breakout', 'volatility', 'sentiment', 'holding_duration']
    X = df[features]
    y = df['target']

    if len(X) == 0:
        print("⚠️ Not enough samples to train ML model.")
        return None

    print("🎓 Training ML model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("\n🧠 ML Classifier Performance:\n", classification_report(y_test, y_pred))
    joblib.dump(clf, "trade_classifier.pkl")
    save_model_meta()
    print("💾 Saved trade_classifier.pkl")
    return clf

def save_trade_history(df):
    try:
        old = pd.read_csv(TRADE_LOG_PATH)
        combined = pd.concat([old, df], ignore_index=True).drop_duplicates()
    except:
        combined = df
    combined.to_csv(TRADE_LOG_PATH, index=False)

# MAIN FUNCTION
if __name__ == '__main__':
    start_time = time.time()
    print("⏳ Starting backtest with impulse filtering...")

    print("📊 Fetching price data...")
    def get_daily_close(symbol):
        for attempt in range(3):
            try:
                bars = api.get_bars(symbol, timeframe="1Day", start=start_date, end=end_date).df
                return bars['close'].rename(symbol)
            except:
                time.sleep(1)
        return pd.Series(name=symbol)

    df = pd.concat([get_daily_close(t) for t in tickers + [market]], axis=1).dropna()
    price = df[tickers]

    print("⚙️ Computing indicators...")
    short_ma = price.rolling(3).mean()
    long_ma = price.rolling(7).mean()
    momentum = short_ma > long_ma
    persistent_momentum = momentum & momentum.shift(1)
    breakout_high = price.rolling(3).max()
    breakout = price > breakout_high.shift(1)
    daily_return = price.pct_change(1)
    volatility = daily_return.rolling(5).std()
    price_above_ma = price > price.rolling(3).mean()

    capped_return = daily_return.clip(upper=0.15)
    max_5d_drop = price.pct_change(5).rolling(5).min()
    avoid_extreme_losses = max_5d_drop > -0.3
    gap_up = price.pct_change() > 0.08
    bad_risk_profile = (max_5d_drop < -0.15) | gap_up

    impulse_filtered = (
        (capped_return > 0.01) &
        (volatility > 0.01) &
        momentum &
        price_above_ma &
        avoid_extreme_losses &
        ~bad_risk_profile
    )

    print("📰 Running sentiment analysis...")
    sentiment_scores = {}
    for ticker in tickers:
        headlines = fetch_gnews_headlines(ticker)
        score = classify_sentiment(headlines)
        sentiment_scores[ticker] = score

    print("🔍 Filtering entries using ML predictions...")
    entry_weights = pd.DataFrame(0.0, index=price.index, columns=price.columns)
    for col in impulse_filtered.columns:
        for date in impulse_filtered.index:
            if impulse_filtered.at[date, col]:
                vol = volatility.at[date, col] if date in volatility.index else 0.01
                sent = sentiment_scores.get(col, 0.0)
                hold = 5
                try:
                    clf = joblib.load("trade_classifier.pkl")
                    features = pd.DataFrame([[int(col in impulse_filtered.columns), 0, 0, vol, sent, hold]],
                        columns=["signal_impulse", "signal_momentum", "signal_breakout", "volatility", "sentiment", "holding_duration"])
                    prob = clf.predict_proba(features)[0][1]
                    if prob < 0.5:
                        impulse_filtered.at[date, col] = False
                    else:
                        entry_weights.at[date, col] = 1 + (prob - 0.5) * 2
                except:
                    pass

    global signal_source
    signal_source = pd.DataFrame("", index=price.index, columns=price.columns)
    entries = (persistent_momentum | breakout | impulse_filtered).fillna(False)
    signal_source[persistent_momentum] = "momentum"
    signal_source[breakout] = "breakout"
    signal_source[impulse_filtered] = "impulse"

    print("📦 Assigning weights and computing exits...")
    entry_weights[persistent_momentum] = 2
    entry_weights[breakout] = 2
    atr = price.pct_change().rolling(14).std() * price
    exits = price < price.rolling(7).mean()

    print("📈 Running portfolio simulation...")
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

    df_trades = pf.trades.records_readable.copy()
    df_trades['Signal'] = df_trades.apply(get_signal, axis=1)

    save_trade_history(df_trades)
    if should_retrain():
        clf = train_trade_classifier(df_trades)

    print("\n📊 Backtest Results:")
    print(pf.stats())
    print(f"✅ All done in {time.time() - start_time:.2f}s")
