# 🧠 Trading Bot

A Python-based trading bot framework for developing, backtesting, analyzing, and running machine-learning-enhanced trading strategies in both historical and live environments.

---

## 🚀 Features

- 🧪 **Backtesting** with historical data
- 📉 **Live trading** using Alpaca or custom brokers
- 🧠 **ML-driven signal generation** and trade classification
- 🔍 **Data analysis and strategy evaluation**
- 📊 **Sentiment analysis** (e.g. news/Reddit)
- 🧰 Modular structure for fast iteration

---

## 🗂 Project Structure

```
trading-bot/
├── analysis/                # Post-trade analysis scripts
├── config/                 # Config files (API keys, parameters)
├── data/                   # Trade logs, CSV outputs
├── live/                   # Live trading engine and signal logic
├── models/                 # Trained ML models for classification
├── results/                # Plots and diagnostics
├── src/                    # Core trading logic
├── strategies/             # Strategy definitions and backtesting
├── utils/                  # Helpers/utilities
├── Pipfile                 # Dependency management (Pipenv)
└── README.md
```

---

## ⚙️ Setup

1. **Install dependencies** (use [Pipenv](https://pipenv.pypa.io/en/latest/)):

```bash
pipenv install
```

2. **Configure your secrets and parameters**:

Update `config/config.json` with your Alpaca keys and any parameters.

---

## 🧪 Backtesting

```bash
python strategies/backtest.py
```

Or load `backtest.py` into a Jupyter notebook for rapid iteration.

---

## 📈 Live Trading

```bash
python src/run_bot.py
```

This uses real-time data and integrates signal logic from `live/signal_generator.py` and trade execution from `live/broker.py`.

---

## 🧠 Machine Learning

ML models (in `models/`) are used to classify trade setups. Train or retrain using `src/ml.py`.

---

## 📊 Analysis

After backtesting or live trading, run:

```bash
python analysis/analyze_results.py
```

Generates reports and heatmaps like:
- `rejection_heatmap.png`
- `TSLA_rejection_timeline.png`

---

## ✅ TODO

- [ ] Add unit tests
- [ ] Auto-save best strategy configs
- [ ] Add new feature importance visualization
- [ ] Improve robustness of signal generator

---

## 📜 License

MIT License
