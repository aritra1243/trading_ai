"""
Technical Features Module
Generates technical analysis indicators for trading.
"""

import pandas as pd
import numpy as np
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class TechnicalFeatures:
    """
    Generates technical analysis features from OHLCV data.
    
    Features include:
    - Moving Averages (EMA, SMA)
    - RSI (Relative Strength Index)
    - MACD (Moving Average Convergence Divergence)
    - Bollinger Bands
    - ATR (Average True Range)
    - VWAP (Volume Weighted Average Price)
    - Momentum indicators
    """
    
    def __init__(
        self,
        ema_periods: List[int] = [9, 20, 50, 200],
        rsi_period: int = 14,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        bb_period: int = 20,
        bb_std: float = 2.0,
        atr_period: int = 14,
        volume_ma_period: int = 20
    ):
        """Initialize with indicator parameters."""
        self.ema_periods = ema_periods
        self.rsi_period = rsi_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.atr_period = atr_period
        self.volume_ma_period = volume_ma_period
    
    def add_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all technical features to DataFrame."""
        df = df.copy()
        
        df = self.add_moving_averages(df)
        df = self.add_rsi(df)
        df = self.add_macd(df)
        df = self.add_bollinger_bands(df)
        df = self.add_atr(df)
        df = self.add_vwap(df)
        df = self.add_momentum(df)
        df = self.add_volume_features(df)
        
        return df
    
    def add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add EMA and SMA indicators."""
        df = df.copy()
        
        for period in self.ema_periods:
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
        
        # Price position relative to EMAs
        if 20 in self.ema_periods and 50 in self.ema_periods:
            df['price_vs_ema20'] = (df['close'] - df['ema_20']) / df['ema_20']
            df['price_vs_ema50'] = (df['close'] - df['ema_50']) / df['ema_50']
            df['ema_20_50_cross'] = (df['ema_20'] > df['ema_50']).astype(int)
        
        return df
    
    def add_rsi(self, df: pd.DataFrame, period: Optional[int] = None) -> pd.DataFrame:
        """Add RSI indicator."""
        df = df.copy()
        period = period or self.rsi_period
        
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        
        avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
        
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # RSI zones
        df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
        df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
        
        return df
    
    def add_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add MACD indicator."""
        df = df.copy()
        
        ema_fast = df['close'].ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=self.macd_slow, adjust=False).mean()
        
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=self.macd_signal, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # MACD crossovers
        df['macd_cross_up'] = ((df['macd'] > df['macd_signal']) & 
                               (df['macd'].shift(1) <= df['macd_signal'].shift(1))).astype(int)
        df['macd_cross_down'] = ((df['macd'] < df['macd_signal']) & 
                                  (df['macd'].shift(1) >= df['macd_signal'].shift(1))).astype(int)
        
        return df
    
    def add_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Bollinger Bands indicator."""
        df = df.copy()
        
        df['bb_middle'] = df['close'].rolling(window=self.bb_period).mean()
        bb_std = df['close'].rolling(window=self.bb_period).std()
        
        df['bb_upper'] = df['bb_middle'] + (self.bb_std * bb_std)
        df['bb_lower'] = df['bb_middle'] - (self.bb_std * bb_std)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # Price position within bands (0 to 1)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        return df
    
    def add_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add ATR (Average True Range) indicator."""
        df = df.copy()
        
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = true_range.rolling(window=self.atr_period).mean()
        
        # ATR as percentage of price
        df['atr_pct'] = df['atr'] / df['close']
        
        return df
    
    def add_vwap(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add VWAP (Volume Weighted Average Price)."""
        df = df.copy()
        
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        df['vwap'] = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
        
        # Price vs VWAP
        df['price_vs_vwap'] = (df['close'] - df['vwap']) / df['vwap']
        
        return df
    
    def add_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators."""
        df = df.copy()
        
        # Rate of change
        for period in [5, 10, 20]:
            df[f'roc_{period}'] = df['close'].pct_change(periods=period)
        
        # Price momentum
        df['momentum_10'] = df['close'] - df['close'].shift(10)
        
        # Stochastic oscillator
        low_14 = df['low'].rolling(window=14).min()
        high_14 = df['high'].rolling(window=14).max()
        df['stoch_k'] = 100 * (df['close'] - low_14) / (high_14 - low_14)
        df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
        
        return df
    
    def add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features."""
        df = df.copy()
        
        # Volume moving average
        df['volume_ma'] = df['volume'].rolling(window=self.volume_ma_period).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # On-Balance Volume (OBV)
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).cumsum()
        
        # Volume price trend
        df['vpt'] = (df['close'].pct_change() * df['volume']).cumsum()
        
        # Volume delta (approximation)
        df['volume_delta'] = df['volume'] * np.sign(df['close'] - df['open'])
        
        return df


# Example usage
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    np.random.seed(42)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': 100 + np.random.randn(100).cumsum(),
        'high': 101 + np.random.randn(100).cumsum(),
        'low': 99 + np.random.randn(100).cumsum(),
        'close': 100 + np.random.randn(100).cumsum(),
        'volume': np.random.randint(1000000, 5000000, 100)
    })
    
    # Fix OHLC relationships
    df['high'] = df[['open', 'high', 'low', 'close']].max(axis=1)
    df['low'] = df[['open', 'high', 'low', 'close']].min(axis=1)
    
    tech = TechnicalFeatures()
    df = tech.add_all_features(df)
    
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(df.tail())
