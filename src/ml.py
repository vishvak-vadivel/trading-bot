import pandas as pd
import joblib
import datetime as dt
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

MODEL_META_PATH = "trade_classifier_meta.json"
TRADE_LOG_PATH = "trade_history.csv"

RETRAIN_INTERVAL_DAYS = 7


def should_retrain(force=False):
    if force:
        return True
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


def save_trade_history(df):
    try:
        old = pd.read_csv(TRADE_LOG_PATH)
        combined = pd.concat([old, df], ignore_index=True).drop_duplicates()
    except:
        combined = df
    combined.to_csv(TRADE_LOG_PATH, index=False)


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
