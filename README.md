# Trading AI System

A comprehensive Python-based trading prediction system designed to generate **Buy/Sell/Hold** signals with entry prices, stop losses, take profits, and confidence scores.

## 🚀 Features

- **Multi-Source Data**: Fetch OHLCV data from Yahoo Finance and Binance.
- **Advanced Feature Engineering**: 50+ indicators including EMA, RSI, MACD, Bollinger Bands, ATR, VWAP, and market structure analysis.
- **Smart Labeling**: Forward-looking signal generation with "Triple Barrier" support.
- **Machine Learning Models**: Support for XGBoost, Random Forest, and Gradient Boosting.
- **Robust Backtesting**: Simulation engine with commission, slippage, and position sizing logic.
- **Risk Management**: Kill switch, daily loss limits, and volatility-adjusted position sizing.
- **Paper Trading**: Live trading simulation with real-time data streaming and logging.

## 🛠️ Installation

1.  **Clone the repository** (if applicable) or navigate to the project directory.

2.  **Create and activate a virtual environment**:
    ```bash
    python -m venv venv
    
    # Windows
    .\venv\Scripts\activate
    
    # Linux/Mac
    source venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## 📖 Usage

The project uses a central CLI script `main.py` for all operations.

### 1. Train a Model
Fetch data, generate features, and train a new model:
```bash
python main.py --mode train --symbol SPY --download
```

### 2. Run Backtest
Test the trained model against historical data:
```bash
python main.py --mode backtest --symbol SPY
```

### 3. Generate Prediction
Get a prediction for the current market state:
```bash
python main.py --mode predict --symbol SPY
```

### 4. Live Paper Trading
Run a live paper trading session (simulated execution):
```bash
python main.py --mode live --symbol SPY --paper --duration 60
```
*(Runs for 60 minutes)*

## 📂 Project Structure

```
trading_ai/
├── backtest/       # Backtesting engine and metrics
├── config/         # Configuration settings
├── data/           # Data fetching and storage
├── features/       # Feature engineering modules
├── live/           # Live/Paper trading engine
├── models/         # Model training and prediction logic
├── risk/           # Risk management modules
├── utils/          # Helper functions
├── main.py         # Main CLI entry point
└── requirements.txt
```

## 📊 Modules

-   **`data`**: Handles data ingestion from Yahoo Finance (`yfinance`) and Binance.
-   **`features`**: Calculates technical indicators (RSI, MACD, etc.) and market structure features.
-   **`models`**: Wrappers for `scikit-learn` and `xgboost` models with time-series cross-validation.
-   **`backtest`**: Event-driven backtesting engine with realistic cost simulation.
-   **`risk`**: logic for position sizing and portfolio protection.

## ⚠️ Disclaimer

This software is for educational purposes only. Do not risk money you cannot afford to lose. Past performance is not indicative of future results.
