"""
Data Fetcher Module
Fetches OHLCV data from multiple sources (Yahoo Finance, Binance)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Literal
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataFetcher:
    """
    Fetches and manages OHLCV data from multiple sources.
    
    Supports:
    - Yahoo Finance (stocks, ETFs, indices)
    - Binance (cryptocurrency)
    """
    
    VALID_SOURCES = ['yahoo', 'binance']
    VALID_TIMEFRAMES = ['1m', '3m', '5m', '10m', '15m', '30m', '1h', '4h', '1d', '1w']
    
    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize DataFetcher.
        
        Args:
            data_dir: Directory to store cached data. Defaults to project data folder.
        """
        if data_dir is None:
            self.data_dir = Path(__file__).parent
        else:
            self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def fetch_ohlcv(
        self,
        symbol: str,
        source: Literal['yahoo', 'binance'] = 'yahoo',
        timeframe: str = '1d',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: str = '2y'
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data from specified source.
        
        Args:
            symbol: Trading symbol (e.g., 'SPY', 'BTCUSDT')
            source: Data source ('yahoo' or 'binance')
            timeframe: Data timeframe (e.g., '1d', '1h', '5m')
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)
            period: Period for Yahoo Finance (e.g., '1y', '2y', 'max')
        
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        if source not in self.VALID_SOURCES:
            raise ValueError(f"Invalid source: {source}. Must be one of {self.VALID_SOURCES}")
        
        logger.info(f"Fetching {symbol} data from {source} ({timeframe})")
        
        if source == 'yahoo':
            df = self._fetch_yahoo(symbol, timeframe, start_date, end_date, period)
        elif source == 'binance':
            df = self._fetch_binance(symbol, timeframe, start_date, end_date)
        else:
            raise ValueError(f"Unsupported source: {source}")
        
        # Validate and clean data
        df = self._validate_and_clean(df)
        
        logger.info(f"Fetched {len(df)} rows of data")
        return df
    
    def _fetch_yahoo(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[str],
        end_date: Optional[str],
        period: str
    ) -> pd.DataFrame:
        """Fetch data from Yahoo Finance."""
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError("yfinance not installed. Run: pip install yfinance")
        
        # Determine base timeframe for resampling
        resample_rule = None
        target_timeframe = timeframe

        if timeframe == '3m':
            target_timeframe = '1m'
            resample_rule = '3min'
        elif timeframe == '10m':
            target_timeframe = '5m'
            resample_rule = '10min'
        
        # Map timeframe to yfinance interval
        interval_map = {
            '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
            '1h': '1h', '4h': '4h', '1d': '1d', '1w': '1wk'
        }
        interval = interval_map.get(target_timeframe, '1d')
        
        ticker = yf.Ticker(symbol)
        
        if start_date and end_date:
            df = ticker.history(start=start_date, end=end_date, interval=interval)
        else:
            df = ticker.history(period=period, interval=interval)
        
        if df.empty:
            raise ValueError(f"No data returned for {symbol}")
        
        # Standardize column names
        df = df.reset_index()
        df.columns = [col.lower() for col in df.columns]
        
        # Rename 'date' or 'datetime' to 'timestamp'
        if 'date' in df.columns:
            df = df.rename(columns={'date': 'timestamp'})
        elif 'datetime' in df.columns:
            df = df.rename(columns={'datetime': 'timestamp'})
        
        # Select only OHLCV columns first
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        df = df[[col for col in required_cols if col in df.columns]]

        # Perform Resampling if needed
        if resample_rule:
            logger.info(f"Resampling data from {target_timeframe} to {timeframe}...")
            # Set timestamp as index for resampling
            df = df.set_index('timestamp')
            
            # Resample logic
            df_resampled = df.resample(resample_rule).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            })
            
            # Drop empty rows (e.g. market closed periods)
            df_resampled = df_resampled.dropna()
            
            # Reset index to get timestamp back as column
            df = df_resampled.reset_index()
        
        return df
    
    def _fetch_binance(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> pd.DataFrame:
        """Fetch data from Binance."""
        try:
            from binance.client import Client
        except ImportError:
            raise ImportError("python-binance not installed. Run: pip install python-binance")
        
        # Initialize Binance client (no API keys needed for public data)
        client = Client()
        
        # Map timeframe to Binance interval
        interval_map = {
            '1m': Client.KLINE_INTERVAL_1MINUTE,
            '5m': Client.KLINE_INTERVAL_5MINUTE,
            '15m': Client.KLINE_INTERVAL_15MINUTE,
            '30m': Client.KLINE_INTERVAL_30MINUTE,
            '1h': Client.KLINE_INTERVAL_1HOUR,
            '4h': Client.KLINE_INTERVAL_4HOUR,
            '1d': Client.KLINE_INTERVAL_1DAY,
            '1w': Client.KLINE_INTERVAL_1WEEK,
        }
        interval = interval_map.get(timeframe, Client.KLINE_INTERVAL_1DAY)
        
        # Fetch klines
        if start_date:
            klines = client.get_historical_klines(
                symbol, interval, start_date, end_date
            )
        else:
            # Default to last 2 years
            start = (datetime.now() - timedelta(days=730)).strftime('%d %b %Y')
            klines = client.get_historical_klines(symbol, interval, start)
        
        if not klines:
            raise ValueError(f"No data returned for {symbol}")
        
        # Convert to DataFrame
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        # Select only OHLCV columns
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Convert string values to float
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        return df
    
    def _validate_and_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clean OHLCV data.
        
        - Remove duplicates
        - Handle missing values
        - Ensure correct data types
        - Sort by timestamp
        """
        # Remove duplicates
        df = df.drop_duplicates(subset=['timestamp'], keep='last')
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Ensure timestamp is datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        
        # Ensure numeric columns are float
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove rows with missing OHLC values
        df = df.dropna(subset=['open', 'high', 'low', 'close'])
        
        # Fill missing volume with 0
        if 'volume' in df.columns:
            df['volume'] = df['volume'].fillna(0)
        
        # Validate OHLC relationships
        # High should be >= Open, Close, Low
        # Low should be <= Open, Close, High
        valid_mask = (
            (df['high'] >= df['open']) &
            (df['high'] >= df['close']) &
            (df['high'] >= df['low']) &
            (df['low'] <= df['open']) &
            (df['low'] <= df['close'])
        )
        
        invalid_count = (~valid_mask).sum()
        if invalid_count > 0:
            logger.warning(f"Removing {invalid_count} rows with invalid OHLC relationships")
            df = df[valid_mask]
        
        return df.reset_index(drop=True)
    
    def save_to_csv(self, df: pd.DataFrame, filename: str) -> Path:
        """
        Save DataFrame to CSV file.
        
        Args:
            df: DataFrame to save
            filename: Name of the file (without extension)
        
        Returns:
            Path to saved file
        """
        filepath = self.data_dir / f"{filename}.csv"
        df.to_csv(filepath, index=False)
        logger.info(f"Saved data to {filepath}")
        return filepath
    
    def load_from_csv(self, filename: str) -> pd.DataFrame:
        """
        Load DataFrame from CSV file.
        
        Args:
            filename: Name of the file (without extension)
        
        Returns:
            Loaded DataFrame
        """
        filepath = self.data_dir / f"{filename}.csv"
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        df = pd.read_csv(filepath)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        logger.info(f"Loaded {len(df)} rows from {filepath}")
        return df
    
    def get_cached_or_fetch(
        self,
        symbol: str,
        source: str = 'yahoo',
        timeframe: str = '1d',
        force_refresh: bool = False,
        **kwargs
    ) -> pd.DataFrame:
        """
        Get data from cache or fetch fresh data if not available.
        
        Args:
            symbol: Trading symbol
            source: Data source
            timeframe: Data timeframe
            force_refresh: Force fetch fresh data even if cache exists
            **kwargs: Additional arguments for fetch_ohlcv
        
        Returns:
            DataFrame with OHLCV data
        """
        cache_name = f"{symbol}_{source}_{timeframe}"
        cache_path = self.data_dir / f"{cache_name}.csv"
        
        if cache_path.exists() and not force_refresh:
            logger.info(f"Loading cached data for {symbol}")
            return self.load_from_csv(cache_name)
        
        # Fetch fresh data
        df = self.fetch_ohlcv(symbol, source, timeframe, **kwargs)
        self.save_to_csv(df, cache_name)
        return df


# Example usage
if __name__ == "__main__":
    fetcher = DataFetcher()
    
    # Fetch SPY data from Yahoo Finance
    df = fetcher.fetch_ohlcv("SPY", source="yahoo", timeframe="1d", period="2y")
    print(f"\nSPY Data Shape: {df.shape}")
    print(df.head())
    print(df.tail())
    
    # Save to CSV
    fetcher.save_to_csv(df, "SPY_yahoo_1d")
