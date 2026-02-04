"""
Trading AI System Configuration
"""
import os
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models" / "saved"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


@dataclass
class TradingConfig:
    """Trading parameters configuration"""
    # Instrument settings
    symbol: str = "SPY"
    timeframe: str = "1d"
    
    # Labeling thresholds
    profit_threshold: float = 0.005  # 0.5% profit target
    loss_threshold: float = 0.003    # 0.3% stop loss
    max_holding_period: int = 10     # Max bars to hold
    
    # Position sizing
    initial_capital: float = 100000.0
    risk_per_trade: float = 0.02     # 2% risk per trade
    max_position_size: float = 0.25  # 25% max position
    
    # Trading costs
    commission: float = 0.001        # 0.1% commission
    slippage: float = 0.0005         # 0.05% slippage


@dataclass
class FeatureConfig:
    """Feature engineering configuration"""
    # Moving averages
    ema_periods: List[int] = field(default_factory=lambda: [9, 20, 50, 200])
    
    # RSI
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    
    # MACD
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    
    # Bollinger Bands
    bb_period: int = 20
    bb_std: float = 2.0
    
    # ATR
    atr_period: int = 14
    
    # Volume
    volume_ma_period: int = 20


@dataclass
class ModelConfig:
    """Model training configuration"""
    # Model selection
    model_type: str = "xgboost"  # logistic, random_forest, xgboost
    
    # Training parameters
    test_size: float = 0.2
    val_size: float = 0.1
    random_state: int = 42
    
    # Cross-validation
    n_splits: int = 5
    
    # XGBoost parameters
    xgb_params: dict = field(default_factory=lambda: {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    })
    
    # Random Forest parameters
    rf_params: dict = field(default_factory=lambda: {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
    })


@dataclass
class BacktestConfig:
    """Backtesting configuration"""
    # Simulation settings
    initial_capital: float = 100000.0
    commission: float = 0.001
    slippage: float = 0.0005
    
    # Position settings
    max_positions: int = 1
    position_size: float = 0.95  # 95% of available capital
    
    # Risk settings
    stop_loss_pct: float = 0.02   # 2% stop loss
    take_profit_pct: float = 0.04  # 4% take profit


@dataclass 
class RiskConfig:
    """Risk management configuration"""
    # Per-trade limits
    max_risk_per_trade: float = 0.02  # 2% max risk
    max_position_pct: float = 0.25    # 25% max position
    
    # Daily limits
    max_daily_loss: float = 0.05      # 5% daily loss limit
    max_daily_trades: int = 10
    
    # Drawdown protection
    max_drawdown: float = 0.15        # 15% max drawdown
    kill_switch_enabled: bool = True
    
    # Volatility adjustment
    volatility_adjustment: bool = True
    atr_multiplier: float = 2.0


@dataclass
class LiveConfig:
    """Live trading configuration"""
    # Data source
    data_source: str = "yahoo"  # yahoo, binance
    
    # Update interval (seconds)
    update_interval: int = 60
    
    # Paper trading
    paper_trading: bool = True
    
    # Logging
    log_trades: bool = True
    log_predictions: bool = True


# Default configurations
DEFAULT_TRADING_CONFIG = TradingConfig()
DEFAULT_FEATURE_CONFIG = FeatureConfig()
DEFAULT_MODEL_CONFIG = ModelConfig()
DEFAULT_BACKTEST_CONFIG = BacktestConfig()
DEFAULT_RISK_CONFIG = RiskConfig()
DEFAULT_LIVE_CONFIG = LiveConfig()
