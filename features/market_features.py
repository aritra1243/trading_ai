"""
Market Structure Features Module
Detects market structure patterns for trading signals.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


class MarketStructureFeatures:
    """
    Generates market structure features from OHLCV data.
    
    Features include:
    - Higher High / Lower Low detection
    - Breakout detection
    - Pullback depth
    - Trend strength
    - Support/Resistance levels
    - Volatility regime
    """
    
    def __init__(
        self,
        swing_lookback: int = 5,
        breakout_lookback: int = 20,
        trend_lookback: int = 50
    ):
        """Initialize with lookback periods."""
        self.swing_lookback = swing_lookback
        self.breakout_lookback = breakout_lookback
        self.trend_lookback = trend_lookback
    
    def add_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all market structure features to DataFrame."""
        df = df.copy()
        
        df = self.add_swing_points(df)
        df = self.add_trend_features(df)
        df = self.add_breakout_features(df)
        df = self.add_support_resistance(df)
        df = self.add_volatility_regime(df)
        df = self.add_candle_patterns(df)
        
        return df
    
    def add_swing_points(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect swing highs and lows (Higher High / Lower Low)."""
        df = df.copy()
        lookback = self.swing_lookback
        
        # Find local highs and lows
        df['swing_high'] = df['high'].rolling(
            window=2*lookback+1, center=True
        ).apply(lambda x: x.iloc[lookback] == x.max(), raw=False).fillna(0).astype(int)
        
        df['swing_low'] = df['low'].rolling(
            window=2*lookback+1, center=True
        ).apply(lambda x: x.iloc[lookback] == x.min(), raw=False).fillna(0).astype(int)
        
        # Track swing high/low values
        df['last_swing_high'] = df.loc[df['swing_high'] == 1, 'high']
        df['last_swing_high'] = df['last_swing_high'].ffill()
        
        df['last_swing_low'] = df.loc[df['swing_low'] == 1, 'low']
        df['last_swing_low'] = df['last_swing_low'].ffill()
        
        # Higher High / Lower Low detection
        df['higher_high'] = (
            (df['swing_high'] == 1) & 
            (df['high'] > df['last_swing_high'].shift(1))
        ).astype(int)
        
        df['lower_low'] = (
            (df['swing_low'] == 1) & 
            (df['low'] < df['last_swing_low'].shift(1))
        ).astype(int)
        
        df['higher_low'] = (
            (df['swing_low'] == 1) & 
            (df['low'] > df['last_swing_low'].shift(1))
        ).astype(int)
        
        df['lower_high'] = (
            (df['swing_high'] == 1) & 
            (df['high'] < df['last_swing_high'].shift(1))
        ).astype(int)
        
        return df
    
    def add_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend direction and strength features."""
        df = df.copy()
        
        # Price trend (linear regression slope)
        def rolling_slope(series, window):
            x = np.arange(window)
            slopes = series.rolling(window).apply(
                lambda y: np.polyfit(x, y, 1)[0] if len(y) == window else np.nan,
                raw=True
            )
            return slopes
        
        df['trend_slope_20'] = rolling_slope(df['close'], 20)
        df['trend_slope_50'] = rolling_slope(df['close'], 50)
        
        # Trend direction (-1, 0, 1)
        slope_threshold = df['close'].std() * 0.01
        df['trend_direction'] = np.select(
            [df['trend_slope_20'] > slope_threshold,
             df['trend_slope_20'] < -slope_threshold],
            [1, -1],
            default=0
        )
        
        # Trend strength (ADX approximation)
        df['price_change'] = df['close'].diff()
        df['high_change'] = df['high'].diff()
        df['low_change'] = df['low'].diff()
        
        df['plus_dm'] = np.where(
            (df['high_change'] > df['low_change'].abs()) & (df['high_change'] > 0),
            df['high_change'], 0
        )
        df['minus_dm'] = np.where(
            (df['low_change'].abs() > df['high_change']) & (df['low_change'] < 0),
            df['low_change'].abs(), 0
        )
        
        # Smoothed +DI and -DI
        period = 14
        df['plus_di'] = 100 * df['plus_dm'].rolling(period).mean() / (
            df['high'] - df['low']).rolling(period).mean()
        df['minus_di'] = 100 * df['minus_dm'].rolling(period).mean() / (
            df['high'] - df['low']).rolling(period).mean()
        
        # ADX calculation
        df['dx'] = 100 * np.abs(df['plus_di'] - df['minus_di']) / (
            df['plus_di'] + df['minus_di'] + 1e-10)
        df['adx'] = df['dx'].rolling(period).mean()
        
        # Clean up intermediate columns
        df = df.drop(columns=['price_change', 'high_change', 'low_change', 
                              'plus_dm', 'minus_dm', 'dx'], errors='ignore')
        
        return df
    
    def add_breakout_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect breakout patterns."""
        df = df.copy()
        lookback = self.breakout_lookback
        
        # Rolling high/low
        df['rolling_high'] = df['high'].rolling(window=lookback).max()
        df['rolling_low'] = df['low'].rolling(window=lookback).min()
        
        # Breakout detection
        df['breakout_up'] = (
            (df['close'] > df['rolling_high'].shift(1)) &
            (df['close'].shift(1) <= df['rolling_high'].shift(2))
        ).astype(int)
        
        df['breakout_down'] = (
            (df['close'] < df['rolling_low'].shift(1)) &
            (df['close'].shift(1) >= df['rolling_low'].shift(2))
        ).astype(int)
        
        # Distance from range boundaries
        df['dist_from_high'] = (df['rolling_high'] - df['close']) / df['close']
        df['dist_from_low'] = (df['close'] - df['rolling_low']) / df['close']
        
        # Pullback depth (retracement from recent high)
        df['pullback_depth'] = (df['rolling_high'] - df['close']) / (
            df['rolling_high'] - df['rolling_low'] + 1e-10)
        
        return df
    
    def add_support_resistance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate distance from support/resistance levels."""
        df = df.copy()
        
        # Simple S/R based on pivot points
        df['pivot'] = (df['high'].shift(1) + df['low'].shift(1) + df['close'].shift(1)) / 3
        df['r1'] = 2 * df['pivot'] - df['low'].shift(1)
        df['s1'] = 2 * df['pivot'] - df['high'].shift(1)
        df['r2'] = df['pivot'] + (df['high'].shift(1) - df['low'].shift(1))
        df['s2'] = df['pivot'] - (df['high'].shift(1) - df['low'].shift(1))
        
        # Distance from pivot levels
        df['dist_from_pivot'] = (df['close'] - df['pivot']) / df['pivot']
        df['dist_from_r1'] = (df['r1'] - df['close']) / df['close']
        df['dist_from_s1'] = (df['close'] - df['s1']) / df['close']
        
        return df
    
    def add_volatility_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect volatility regime (high/low volatility periods)."""
        df = df.copy()
        
        # Calculate realized volatility
        df['returns'] = df['close'].pct_change()
        df['volatility_20'] = df['returns'].rolling(20).std() * np.sqrt(252)
        df['volatility_60'] = df['returns'].rolling(60).std() * np.sqrt(252)
        
        # Volatility ratio
        df['vol_ratio'] = df['volatility_20'] / (df['volatility_60'] + 1e-10)
        
        # Volatility regime (high = 1, normal = 0, low = -1)
        vol_mean = df['volatility_20'].rolling(100).mean()
        vol_std = df['volatility_20'].rolling(100).std()
        
        df['vol_regime'] = np.select(
            [df['volatility_20'] > vol_mean + vol_std,
             df['volatility_20'] < vol_mean - vol_std],
            [1, -1],
            default=0
        )
        
        return df
    
    def add_candle_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic candlestick pattern features."""
        df = df.copy()
        
        # Candle body and wick sizes
        df['body_size'] = np.abs(df['close'] - df['open'])
        df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
        df['candle_range'] = df['high'] - df['low']
        
        # Body to range ratio
        df['body_ratio'] = df['body_size'] / (df['candle_range'] + 1e-10)
        
        # Bullish/Bearish candle
        df['bullish_candle'] = (df['close'] > df['open']).astype(int)
        df['bearish_candle'] = (df['close'] < df['open']).astype(int)
        
        # Doji detection (small body)
        avg_body = df['body_size'].rolling(20).mean()
        df['doji'] = (df['body_size'] < avg_body * 0.1).astype(int)
        
        # Engulfing patterns
        df['bullish_engulf'] = (
            (df['bullish_candle'] == 1) &
            (df['bearish_candle'].shift(1) == 1) &
            (df['open'] < df['close'].shift(1)) &
            (df['close'] > df['open'].shift(1))
        ).astype(int)
        
        df['bearish_engulf'] = (
            (df['bearish_candle'] == 1) &
            (df['bullish_candle'].shift(1) == 1) &
            (df['open'] > df['close'].shift(1)) &
            (df['close'] < df['open'].shift(1))
        ).astype(int)
        
        return df


# Example usage
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    np.random.seed(42)
    
    base_price = 100 + np.random.randn(100).cumsum()
    df = pd.DataFrame({
        'timestamp': dates,
        'open': base_price + np.random.randn(100) * 0.5,
        'close': base_price + np.random.randn(100) * 0.5,
        'volume': np.random.randint(1000000, 5000000, 100)
    })
    df['high'] = df[['open', 'close']].max(axis=1) + np.abs(np.random.randn(100) * 0.5)
    df['low'] = df[['open', 'close']].min(axis=1) - np.abs(np.random.randn(100) * 0.5)
    
    market = MarketStructureFeatures()
    df = market.add_all_features(df)
    
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
