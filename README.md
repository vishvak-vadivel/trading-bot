# Momentum Trend-Following Backtester

This bot uses Alpaca for data and VectorBT for signal generation & backtesting.

## Strategy
- Enters when 3-day MA > 20-day MA
- Exits on:
  - MA reversal
  - 6% Take-Profit
  - 2% Stop-Loss
  - or 15-day timeout

## Setup
1. Install requirements: `pip install -r requirements.txt`
2. Add `config.json` like:
```json
{
  "api_key": "your_alpaca_key",
  "secret_key": "your_secret_key"
}
