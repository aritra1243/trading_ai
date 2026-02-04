"""
Helper Utilities Module
Common helper functions for the trading system.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union
import logging
import sys
from datetime import datetime


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_str: Optional[str] = None
) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        level: Logging level
        log_file: Optional file to write logs to
        format_str: Custom format string
    
    Returns:
        Configured logger
    """
    if format_str is None:
        format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format=format_str,
        handlers=handlers
    )
    
    return logging.getLogger('trading_ai')


def format_currency(value: float, currency: str = '$') -> str:
    """Format a value as currency."""
    if value >= 0:
        return f"{currency}{value:,.2f}"
    else:
        return f"-{currency}{abs(value):,.2f}"


def format_percent(value: float, decimals: int = 2) -> str:
    """Format a value as percentage."""
    return f"{value:.{decimals}f}%"


def calculate_returns(
    prices: pd.Series,
    method: str = 'simple'
) -> pd.Series:
    """
    Calculate returns from price series.
    
    Args:
        prices: Price series
        method: 'simple' or 'log'
    
    Returns:
        Returns series
    """
    if method == 'log':
        return np.log(prices / prices.shift(1))
    else:
        return prices.pct_change()


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure a directory exists, creating it if necessary."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def validate_ohlcv(df: pd.DataFrame) -> bool:
    """
    Validate OHLCV data format.
    
    Args:
        df: DataFrame to validate
    
    Returns:
        True if valid, raises ValueError otherwise
    """
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Check for valid OHLC relationships
    invalid = (
        (df['high'] < df['low']) |
        (df['high'] < df['open']) |
        (df['high'] < df['close']) |
        (df['low'] > df['open']) |
        (df['low'] > df['close'])
    )
    
    if invalid.any():
        raise ValueError(f"Invalid OHLC relationships in {invalid.sum()} rows")
    
    return True


def resample_ohlcv(
    df: pd.DataFrame,
    timeframe: str
) -> pd.DataFrame:
    """
    Resample OHLCV data to different timeframe.
    
    Args:
        df: DataFrame with OHLCV data
        timeframe: Target timeframe (e.g., '1H', '4H', '1D')
    
    Returns:
        Resampled DataFrame
    """
    if 'timestamp' in df.columns:
        df = df.set_index('timestamp')
    
    resampled = df.resample(timeframe).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    return resampled.reset_index()


def calculate_drawdown(equity_curve: pd.Series) -> pd.DataFrame:
    """
    Calculate drawdown series from equity curve.
    
    Args:
        equity_curve: Series of equity values
    
    Returns:
        DataFrame with drawdown information
    """
    peak = equity_curve.expanding().max()
    drawdown = (equity_curve - peak) / peak
    
    df = pd.DataFrame({
        'equity': equity_curve,
        'peak': peak,
        'drawdown': drawdown,
        'drawdown_pct': drawdown * 100
    })
    
    return df


def print_trade_summary(trades: list):
    """Print a formatted trade summary."""
    if not trades:
        print("No trades to display.")
        return
    
    print("\n" + "="*60)
    print("TRADE SUMMARY")
    print("="*60)
    
    total_pnl = sum(t.pnl for t in trades)
    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl < 0]
    
    print(f"Total Trades: {len(trades)}")
    print(f"Winning: {len(wins)}")
    print(f"Losing: {len(losses)}")
    print(f"Win Rate: {len(wins)/len(trades)*100:.1f}%")
    print(f"Net P&L: {format_currency(total_pnl)}")
    
    if wins:
        print(f"Avg Win: {format_currency(np.mean([t.pnl for t in wins]))}")
    if losses:
        print(f"Avg Loss: {format_currency(np.mean([t.pnl for t in losses]))}")
    
    print("="*60 + "\n")


def create_performance_chart(
    equity_curve: pd.Series,
    benchmark: Optional[pd.Series] = None,
    save_path: Optional[str] = None
):
    """
    Create a performance chart.
    
    Args:
        equity_curve: Strategy equity curve
        benchmark: Optional benchmark for comparison
        save_path: Optional path to save the chart
    """
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Equity curve
        ax1 = axes[0]
        ax1.plot(equity_curve.index, equity_curve.values, label='Strategy', linewidth=1.5)
        
        if benchmark is not None:
            # Normalize benchmark to same starting value
            benchmark_norm = benchmark / benchmark.iloc[0] * equity_curve.iloc[0]
            ax1.plot(benchmark_norm.index, benchmark_norm.values, 
                    label='Benchmark', linewidth=1.5, alpha=0.7)
        
        ax1.set_ylabel('Equity')
        ax1.set_title('Strategy Performance')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Drawdown
        ax2 = axes[1]
        dd = calculate_drawdown(equity_curve)
        ax2.fill_between(dd.index, dd['drawdown_pct'], 0, 
                        alpha=0.3, color='red')
        ax2.plot(dd.index, dd['drawdown_pct'], color='red', linewidth=1)
        ax2.set_ylabel('Drawdown %')
        ax2.set_xlabel('Date')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Chart saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
        
    except ImportError:
        print("matplotlib not installed. Skipping chart creation.")


def get_market_hours(symbol: str) -> dict:
    """Get market hours for a symbol."""
    # Simplified market hours
    us_markets = {
        'open': '09:30',
        'close': '16:00',
        'timezone': 'US/Eastern',
        'trading_days': [0, 1, 2, 3, 4]  # Mon-Fri
    }
    
    crypto_markets = {
        'open': '00:00',
        'close': '23:59',
        'timezone': 'UTC',
        'trading_days': [0, 1, 2, 3, 4, 5, 6]  # Every day
    }
    
    # Detect crypto
    if symbol.endswith('USDT') or symbol.endswith('BTC') or symbol.endswith('ETH'):
        return crypto_markets
    
    return us_markets


def is_market_open(symbol: str, check_time: Optional[datetime] = None) -> bool:
    """Check if market is currently open for a symbol."""
    hours = get_market_hours(symbol)
    
    now = check_time or datetime.now()
    
    # Check trading day
    if now.weekday() not in hours['trading_days']:
        return False
    
    # Check hours (simplified - doesn't handle timezone properly)
    current_time = now.strftime('%H:%M')
    return hours['open'] <= current_time <= hours['close']
