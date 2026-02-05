#!/usr/bin/env python
"""
Trading AI System - Main Entry Point

A comprehensive trading prediction system that outputs:
- Buy/Sell/Hold signals
- Entry price
- Stop loss
- Take profit  
- Confidence score

Usage:
    python main.py --mode train --symbol SPY --download
    python main.py --mode backtest --symbol SPY
    python main.py --mode predict --symbol SPY
    python main.py --mode live --symbol SPY --paper
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import logging

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    DEFAULT_TRADING_CONFIG,
    DEFAULT_FEATURE_CONFIG,
    DEFAULT_MODEL_CONFIG,
    DEFAULT_BACKTEST_CONFIG,
    DEFAULT_RISK_CONFIG,
    MODELS_DIR,
    DATA_DIR
)
from utils.helpers import setup_logging


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Trading AI System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Train a model:
    python main.py --mode train --symbol SPY --download
    
  Backtest a strategy:
    python main.py --mode backtest --symbol SPY
    
  Generate predictions:
    python main.py --mode predict --symbol SPY
    
  Paper trading:
    python main.py --mode live --symbol SPY --paper --duration 60
        """
    )
    
    parser.add_argument(
        '--mode', 
        choices=['train', 'backtest', 'predict', 'live'],
        required=True,
        help='Operation mode'
    )
    
    parser.add_argument(
        '--symbol',
        default='SPY',
        help='Trading symbol (default: SPY)'
    )
    
    parser.add_argument(
        '--source',
        choices=['yahoo', 'binance'],
        default='yahoo',
        help='Data source (default: yahoo)'
    )
    
    parser.add_argument(
        '--timeframe',
        default='1d',
        help='Data timeframe (default: 1d)'
    )
    
    parser.add_argument(
        '--download',
        action='store_true',
        help='Download fresh data'
    )
    
    parser.add_argument(
        '--model-type',
        choices=['logistic', 'random_forest', 'xgboost', 'gradient_boosting'],
        default='xgboost',
        help='Model type (default: xgboost)'
    )
    
    parser.add_argument(
        '--paper',
        action='store_true',
        help='Paper trading mode (for live)'
    )
    
    parser.add_argument(
        '--duration',
        type=int,
        default=None,
        help='Duration in minutes (for live mode)'
    )
    
    parser.add_argument(
        '--capital',
        type=float,
        default=100000.0,
        help='Initial capital (default: 100000)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    
    return parser.parse_args()


def run_train(args, logger):
    """Train a trading model."""
    from data import DataFetcher
    from features import FeatureEngineer
    from models import ModelTrainer
    
    logger.info(f"Training model for {args.symbol}...")
    
    # Fetch data
    fetcher = DataFetcher(data_dir=DATA_DIR)
    
    if args.download:
        logger.info("Downloading fresh data...")
        # Adjust period for intraday data
        period = '59d' if args.timeframe in ['1m', '5m', '15m', '30m', '60m', '1h'] else '2y'
        
        df = fetcher.fetch_ohlcv(
            symbol=args.symbol,
            source=args.source,
            timeframe=args.timeframe,
            period=period
        )
        fetcher.save_to_csv(df, f"{args.symbol}_{args.source}_{args.timeframe}")
    else:
        df = fetcher.get_cached_or_fetch(
            symbol=args.symbol,
            source=args.source,
            timeframe=args.timeframe
        )
    
    logger.info(f"Data shape: {df.shape}")
    
    # Feature engineering
    logger.info("Generating features...")
    engineer = FeatureEngineer(
        label_config={
            'profit_threshold': DEFAULT_TRADING_CONFIG.profit_threshold,
            'loss_threshold': DEFAULT_TRADING_CONFIG.loss_threshold,
            'max_holding_period': DEFAULT_TRADING_CONFIG.max_holding_period
        },
        select_k_best=50
    )
    
    full_df, X, y = engineer.fit_transform(df)
    
    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Features: {engineer.feature_columns[:10]}...")
    
    # Train model
    logger.info(f"Training {args.model_type} model...")
    trainer = ModelTrainer(
        model_type=args.model_type,
        model_params=DEFAULT_MODEL_CONFIG.xgb_params if args.model_type == 'xgboost' else {}
    )
    
    metrics = trainer.train(
        X, y,
        feature_names=engineer.selected_feature_columns if hasattr(engineer, 'selected_feature_columns') else engineer.feature_columns,
        test_size=DEFAULT_MODEL_CONFIG.test_size,
        val_size=DEFAULT_MODEL_CONFIG.val_size
    )
    
    # Print metrics
    print("\n" + "="*60)
    print("TRAINING RESULTS")
    print("="*60)
    print(f"Test Accuracy:  {metrics['test']['accuracy']:.4f}")
    print(f"Test Precision: {metrics['test']['precision']:.4f}")
    print(f"Test Recall:    {metrics['test']['recall']:.4f}")
    print(f"Test F1:        {metrics['test']['f1']:.4f}")
    print("="*60)
    
    # Save model and pipeline
    model_path = MODELS_DIR / f"{args.symbol}_{args.model_type}_model.joblib"
    pipeline_path = MODELS_DIR / f"{args.symbol}_pipeline"
    
    trainer.save(model_path)
    engineer.save(pipeline_path)
    
    logger.info(f"Model saved to {model_path}")
    logger.info(f"Pipeline saved to {pipeline_path}")
    
    # Feature importance
    importance = trainer.get_feature_importance()
    if not importance.empty:
        print("\nTop 10 Features:")
        print(importance.head(10).to_string(index=False))
    
    return trainer, engineer


def run_backtest(args, logger):
    """Run backtest on trained model."""
    from data import DataFetcher
    from features import FeatureEngineer
    from models import ModelTrainer, TradingPredictor
    from backtest import BacktestEngine, PerformanceMetrics
    
    logger.info(f"Running backtest for {args.symbol}...")
    
    # Load model and pipeline
    model_path = MODELS_DIR / f"{args.symbol}_{args.model_type}_model.joblib"
    pipeline_path = MODELS_DIR / f"{args.symbol}_pipeline"
    
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        logger.error("Run training first: python main.py --mode train --symbol " + args.symbol)
        return None
    
    trainer = ModelTrainer()
    trainer.load(model_path)
    
    engineer = FeatureEngineer()
    engineer.load(pipeline_path)
    
    # Fetch data
    fetcher = DataFetcher(data_dir=DATA_DIR)
    
    # Adjust period for intraday data (passed to fetch_ohlcv if fetching is needed)
    period = '59d' if args.timeframe in ['1m', '5m', '15m', '30m', '60m', '1h'] else '2y'
    
    df = fetcher.get_cached_or_fetch(
        symbol=args.symbol,
        source=args.source,
        timeframe=args.timeframe,
        period=period
    )
    
    logger.info(f"Data shape: {df.shape}")
    
    # Generate predictions
    logger.info("Generating predictions...")
    predictor = TradingPredictor(
        model_trainer=trainer,
        feature_engineer=engineer,
        min_confidence=0.55
    )
    
    signals = predictor.predict(df)
    
    # Filter valid signals (after feature warmup period)
    warmup = 100  # Skip first 100 bars for feature warmup
    df_test = df.iloc[warmup:].reset_index(drop=True)
    signals_test = signals[warmup:]
    
    # Run backtest
    logger.info("Running backtest simulation...")
    engine = BacktestEngine(
        initial_capital=args.capital,
        commission=DEFAULT_BACKTEST_CONFIG.commission,
        slippage=DEFAULT_BACKTEST_CONFIG.slippage
    )
    
    result = engine.run(df_test, signals_test)
    
    # Print performance report
    metrics = PerformanceMetrics()
    report = metrics.generate_report(
        result.trades,
        result.equity_curve,
        args.capital
    )
    print(report)
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = PROJECT_ROOT / 'backtest' / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save equity curve
    result.equity_curve.to_csv(results_dir / f'{args.symbol}_equity_{timestamp}.csv')
    
    # Save trade summary
    trades_df = engine.get_trade_summary()
    if not trades_df.empty:
        trades_df.to_csv(results_dir / f'{args.symbol}_trades_{timestamp}.csv', index=False)
    
    logger.info(f"Results saved to {results_dir}")
    
    return result


def run_predict(args, logger):
    """Generate predictions for current market state."""
    from data import DataFetcher
    from features import FeatureEngineer
    from models import ModelTrainer, TradingPredictor
    
    logger.info(f"Generating prediction for {args.symbol}...")
    
    # Load model and pipeline
    model_path = MODELS_DIR / f"{args.symbol}_{args.model_type}_model.joblib"
    pipeline_path = MODELS_DIR / f"{args.symbol}_pipeline"
    
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        return None
    
    trainer = ModelTrainer()
    trainer.load(model_path)
    
    engineer = FeatureEngineer()
    engineer.load(pipeline_path)
    
    # Fetch latest data
    fetcher = DataFetcher(data_dir=DATA_DIR)
    
    # Adjust period for intraday data
    period = '1mo' if args.timeframe in ['1m', '5m', '15m', '30m', '60m', '1h'] else '3mo'
    
    df = fetcher.fetch_ohlcv(
        symbol=args.symbol,
        source=args.source,
        timeframe=args.timeframe,
        period=period
    )
    
    # Generate signal
    predictor = TradingPredictor(
        model_trainer=trainer,
        feature_engineer=engineer
    )
    
    signal = predictor.predict_single(df)
    
    # Display signal
    print("\n" + "="*60)
    print(f"TRADING SIGNAL - {args.symbol}")
    print("="*60)
    print(f"Timestamp:      {signal.timestamp}")
    print(f"Signal:         {signal.signal_name}")
    print(f"Confidence:     {signal.confidence*100:.1f}%")
    print("-"*60)
    print(f"Entry Price:    ${signal.entry_price:.2f}")
    print(f"Stop Loss:      ${signal.stop_loss:.2f}")
    print(f"Take Profit:    ${signal.take_profit:.2f}")
    print(f"Risk/Reward:    {signal.risk_reward_ratio:.2f}")
    print(f"Position Size:  {signal.position_size_pct*100:.1f}% of capital")
    print("="*60 + "\n")
    
    return signal


def run_live(args, logger):
    """Run live paper trading."""
    from data import DataFetcher
    from features import FeatureEngineer
    from models import ModelTrainer
    from risk import RiskManager
    from live import LiveTrader
    
    logger.info(f"Starting live trading for {args.symbol}...")
    
    # Load model and pipeline
    model_path = MODELS_DIR / f"{args.symbol}_{args.model_type}_model.joblib"
    pipeline_path = MODELS_DIR / f"{args.symbol}_pipeline"
    
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        return None
    
    trainer = ModelTrainer()
    trainer.load(model_path)
    
    engineer = FeatureEngineer()
    engineer.load(pipeline_path)
    
    # Initialize risk manager
    risk_manager = RiskManager(
        initial_capital=args.capital,
        max_risk_per_trade=DEFAULT_RISK_CONFIG.max_risk_per_trade,
        max_daily_loss=DEFAULT_RISK_CONFIG.max_daily_loss,
        max_drawdown=DEFAULT_RISK_CONFIG.max_drawdown
    )
    
    # Initialize live trader
    trader = LiveTrader(
        model_trainer=trainer,
        feature_engineer=engineer,
        risk_manager=risk_manager,
        initial_capital=args.capital,
        symbol=args.symbol,
        data_source=args.source,
        timeframe=args.timeframe,
        update_interval=60
    )
    
    # Start trading
    print(f"\nStarting paper trading for {args.symbol}...")
    print("Press Ctrl+C to stop\n")
    
    trader.start(duration_minutes=args.duration)
    
    # Print summary
    summary = trader.get_performance_summary()
    print("\n" + "="*60)
    print("PAPER TRADING SUMMARY")
    print("="*60)
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")
    print("="*60)
    
    return trader


def main():
    """Main entry point."""
    args = parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logging(level=log_level)
    
    logger.info("="*60)
    logger.info("Trading AI System")
    logger.info("="*60)
    
    try:
        if args.mode == 'train':
            run_train(args, logger)
        elif args.mode == 'backtest':
            run_backtest(args, logger)
        elif args.mode == 'predict':
            run_predict(args, logger)
        elif args.mode == 'live':
            run_live(args, logger)
        
        logger.info("Done!")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
