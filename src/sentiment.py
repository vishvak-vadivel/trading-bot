import requests
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from config import GNEWS_API_KEY

print("📦 Loading FinBERT model...")
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

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
        print("⚠️ No headlines provided for sentiment analysis.")
        return 0.0
    try:
        print(f"🧠 Analyzing sentiment...")
        inputs = tokenizer(headlines, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        scores = probs[:, 2] - probs[:, 0]
        return float(scores.mean())
    except Exception as e:
        print(f"⚠️ FinBERT classification error: {e}")
        return 0.0