"""
Time Features Module
Generates time-based features for trading.
"""

import pandas as pd
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class TimeFeatures:
    """
    Generates time-based features from timestamp data.
    
    Features include:
    - Session time (market hours)
    - Day of week
    - Month of year
    - Market open/close behavior
    - Holiday proximity
    """
    
    def __init__(
        self,
        market_open_hour: int = 9,
        market_close_hour: int = 16,
        timezone: str = 'US/Eastern'
    ):
        """Initialize with market hours."""
        self.market_open_hour = market_open_hour
        self.market_close_hour = market_close_hour
        self.timezone = timezone
    
    def add_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all time-based features to DataFrame."""
        df = df.copy()
        
        # Ensure timestamp is datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        
        df = self.add_date_features(df)
        df = self.add_session_features(df)
        df = self.add_cyclical_features(df)
        df = self.add_lag_features(df)
        
        return df
    
    def add_date_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic date features."""
        df = df.copy()
        
        ts = df['timestamp']
        
        # Day of week (0=Monday, 6=Sunday)
        df['day_of_week'] = ts.dt.dayofweek
        
        # Is weekend (usually no trading, but included for completeness)
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Day of month
        df['day_of_month'] = ts.dt.day
        
        # Week of year
        df['week_of_year'] = ts.dt.isocalendar().week.astype(int)
        
        # Month of year
        df['month'] = ts.dt.month
        
        # Quarter
        df['quarter'] = ts.dt.quarter
        
        # Year
        df['year'] = ts.dt.year
        
        # Is month start/end
        df['is_month_start'] = ts.dt.is_month_start.astype(int)
        df['is_month_end'] = ts.dt.is_month_end.astype(int)
        
        # Is quarter start/end
        df['is_quarter_start'] = ts.dt.is_quarter_start.astype(int)
        df['is_quarter_end'] = ts.dt.is_quarter_end.astype(int)
        
        return df
    
    def add_session_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trading session features (for intraday data)."""
        df = df.copy()
        
        ts = df['timestamp']
        
        # Hour of day
        df['hour'] = ts.dt.hour
        
        # Minute of day
        df['minute_of_day'] = ts.dt.hour * 60 + ts.dt.minute
        
        # Session periods
        df['is_market_open'] = (
            (df['hour'] >= self.market_open_hour) & 
            (df['hour'] < self.market_close_hour)
        ).astype(int)
        
        # Opening hour (first hour of trading)
        df['is_opening_hour'] = (df['hour'] == self.market_open_hour).astype(int)
        
        # Closing hour (last hour of trading)
        df['is_closing_hour'] = (df['hour'] == self.market_close_hour - 1).astype(int)
        
        # Pre-market / After-hours (if data available)
        df['is_premarket'] = (df['hour'] < self.market_open_hour).astype(int)
        df['is_afterhours'] = (df['hour'] >= self.market_close_hour).astype(int)
        
        # Time since market open (in hours)
        df['hours_since_open'] = np.maximum(0, df['hour'] - self.market_open_hour)
        
        # Time until market close (in hours)
        df['hours_until_close'] = np.maximum(0, self.market_close_hour - df['hour'])
        
        return df
    
    def add_cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cyclical encoding for time features."""
        df = df.copy()
        
        # Cyclical encoding for day of week
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Cyclical encoding for month
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Cyclical encoding for hour (if available)
        if 'hour' in df.columns:
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Cyclical encoding for day of month
        df['dom_sin'] = np.sin(2 * np.pi * df['day_of_month'] / 31)
        df['dom_cos'] = np.cos(2 * np.pi * df['day_of_month'] / 31)
        
        return df
    
    def add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-lagged return features."""
        df = df.copy()
        
        # Returns at different time lags
        if 'close' in df.columns:
            df['return_1d'] = df['close'].pct_change(1)
            df['return_5d'] = df['close'].pct_change(5)
            df['return_10d'] = df['close'].pct_change(10)
            df['return_20d'] = df['close'].pct_change(20)
            
            # Lagged returns (previous day's return)
            df['prev_return_1d'] = df['return_1d'].shift(1)
            df['prev_return_5d'] = df['return_5d'].shift(1)
        
        return df
    
    def add_special_dates(
        self, 
        df: pd.DataFrame, 
        holidays: Optional[pd.DatetimeIndex] = None
    ) -> pd.DataFrame:
        """Add features for special dates (holidays, FOMC, etc.)."""
        df = df.copy()
        
        if holidays is not None:
            # Distance to nearest holiday (in days)
            def days_to_holiday(date, holidays):
                diffs = [(h - date).days for h in holidays]
                future_diffs = [d for d in diffs if d >= 0]
                return min(future_diffs) if future_diffs else 999
            
            df['days_to_holiday'] = df['timestamp'].apply(
                lambda x: days_to_holiday(x, holidays)
            )
            
            # Is day before/after holiday
            df['is_pre_holiday'] = (df['days_to_holiday'] == 1).astype(int)
        
        # Known high-volatility periods (approximate)
        # Triple witching (3rd Friday of March, June, Sept, Dec)
        df['is_triple_witching'] = (
            (df['day_of_week'] == 4) &  # Friday
            (df['timestamp'].dt.day >= 15) & 
            (df['timestamp'].dt.day <= 21) &
            (df['month'].isin([3, 6, 9, 12]))
        ).astype(int)
        
        # Month-end rebalancing period (last 3 days of month)
        df['is_month_end_period'] = (df['day_of_month'] >= 28).astype(int)
        
        return df


# Example usage
if __name__ == "__main__":
    # Create sample intraday data
    dates = pd.date_range('2023-01-01 09:30', periods=100, freq='1h')
    np.random.seed(42)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': 100 + np.random.randn(100).cumsum(),
        'high': 101 + np.random.randn(100).cumsum(),
        'low': 99 + np.random.randn(100).cumsum(),
        'close': 100 + np.random.randn(100).cumsum(),
        'volume': np.random.randint(1000000, 5000000, 100)
    })
    
    time_feat = TimeFeatures()
    df = time_feat.add_all_features(df)
    
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(df[['timestamp', 'day_of_week', 'hour', 'dow_sin', 'dow_cos']].head(10))
