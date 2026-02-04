"""
Trade Labeling Module
Generates supervised learning labels for trading signals.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class TradeLabelGenerator:
    """
    Generates trading signal labels using forward-looking logic.
    
    Labeling rules:
    - Buy (1): Price goes up by profit_threshold before going down by loss_threshold
    - Sell (-1): Price goes down by profit_threshold before going up by loss_threshold
    - Hold (0): Neither condition is met within max_holding_period
    """
    
    def __init__(
        self,
        profit_threshold: float = 0.005,  # 0.5% profit target
        loss_threshold: float = 0.003,    # 0.3% stop loss
        max_holding_period: int = 10,     # Max bars to hold
        min_risk_reward: float = 1.5      # Minimum risk/reward ratio
    ):
        """
        Initialize label generator.
        
        Args:
            profit_threshold: Percentage gain to trigger buy/sell signal
            loss_threshold: Percentage loss threshold
            max_holding_period: Maximum bars to look forward
            min_risk_reward: Minimum risk/reward ratio for valid signals
        """
        self.profit_threshold = profit_threshold
        self.loss_threshold = loss_threshold
        self.max_holding_period = max_holding_period
        self.min_risk_reward = min_risk_reward
    
    def generate_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading labels for the entire dataset.
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            DataFrame with additional label columns:
            - signal: Trading signal (1=buy, -1=sell, 0=hold)
            - target_price: Target price for the trade
            - stop_loss: Stop loss price
            - bars_to_target: Number of bars until target was hit
        """
        df = df.copy()
        
        n = len(df)
        signals = np.zeros(n)
        target_prices = np.zeros(n)
        stop_losses = np.zeros(n)
        bars_to_target = np.zeros(n)
        
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        logger.info(f"Generating labels for {n} samples...")
        
        for i in range(n - self.max_holding_period):
            entry_price = close[i]
            
            # Define buy thresholds
            buy_target = entry_price * (1 + self.profit_threshold)
            buy_stop = entry_price * (1 - self.loss_threshold)
            
            # Define sell thresholds
            sell_target = entry_price * (1 - self.profit_threshold)
            sell_stop = entry_price * (1 + self.loss_threshold)
            
            # Look forward to determine signal
            signal, bars, target, stop = self._evaluate_trade(
                high[i+1:i+1+self.max_holding_period],
                low[i+1:i+1+self.max_holding_period],
                entry_price,
                buy_target, buy_stop,
                sell_target, sell_stop
            )
            
            signals[i] = signal
            target_prices[i] = target
            stop_losses[i] = stop
            bars_to_target[i] = bars
        
        df['signal'] = signals.astype(int)
        df['target_price'] = target_prices
        df['stop_loss'] = stop_losses
        df['bars_to_target'] = bars_to_target.astype(int)
        
        # Calculate potential profit/loss
        df['potential_profit'] = np.where(
            df['signal'] == 1,
            (df['target_price'] - df['close']) / df['close'],
            np.where(
                df['signal'] == -1,
                (df['close'] - df['target_price']) / df['close'],
                0
            )
        )
        
        df['potential_loss'] = np.where(
            df['signal'] == 1,
            (df['close'] - df['stop_loss']) / df['close'],
            np.where(
                df['signal'] == -1,
                (df['stop_loss'] - df['close']) / df['close'],
                0
            )
        )
        
        # Log statistics
        buy_count = (df['signal'] == 1).sum()
        sell_count = (df['signal'] == -1).sum()
        hold_count = (df['signal'] == 0).sum()
        
        logger.info(f"Labels generated: Buy={buy_count}, Sell={sell_count}, Hold={hold_count}")
        logger.info(f"Class distribution: Buy={buy_count/n*100:.1f}%, "
                   f"Sell={sell_count/n*100:.1f}%, Hold={hold_count/n*100:.1f}%")
        
        return df
    
    def _evaluate_trade(
        self,
        future_highs: np.ndarray,
        future_lows: np.ndarray,
        entry_price: float,
        buy_target: float,
        buy_stop: float,
        sell_target: float,
        sell_stop: float
    ) -> Tuple[int, int, float, float]:
        """
        Evaluate if a trade would be profitable.
        
        Returns:
            Tuple of (signal, bars_to_target, target_price, stop_loss)
        """
        buy_target_hit = -1
        buy_stop_hit = -1
        sell_target_hit = -1
        sell_stop_hit = -1
        
        # Find when each threshold is hit
        for i, (high, low) in enumerate(zip(future_highs, future_lows)):
            # Check buy trade
            if buy_target_hit < 0 and high >= buy_target:
                buy_target_hit = i
            if buy_stop_hit < 0 and low <= buy_stop:
                buy_stop_hit = i
            
            # Check sell trade
            if sell_target_hit < 0 and low <= sell_target:
                sell_target_hit = i
            if sell_stop_hit < 0 and high >= sell_stop:
                sell_stop_hit = i
        
        # Determine signal based on what got hit first
        # Buy signal: target hit before stop
        buy_valid = (
            buy_target_hit >= 0 and 
            (buy_stop_hit < 0 or buy_target_hit <= buy_stop_hit)
        )
        
        # Sell signal: target hit before stop
        sell_valid = (
            sell_target_hit >= 0 and 
            (sell_stop_hit < 0 or sell_target_hit <= sell_stop_hit)
        )
        
        if buy_valid and not sell_valid:
            return 1, buy_target_hit + 1, buy_target, buy_stop
        elif sell_valid and not buy_valid:
            return -1, sell_target_hit + 1, sell_target, sell_stop
        elif buy_valid and sell_valid:
            # Both valid, choose the one that hits first
            if buy_target_hit <= sell_target_hit:
                return 1, buy_target_hit + 1, buy_target, buy_stop
            else:
                return -1, sell_target_hit + 1, sell_target, sell_stop
        else:
            return 0, 0, entry_price, entry_price
    
    def add_triple_barrier_labels(
        self,
        df: pd.DataFrame,
        take_profit: float = 0.02,
        stop_loss: float = 0.01,
        max_bars: int = 20
    ) -> pd.DataFrame:
        """
        Alternative labeling using triple-barrier method.
        
        Barriers:
        1. Upper barrier (take profit)
        2. Lower barrier (stop loss)
        3. Time barrier (max holding period)
        
        Label = which barrier was touched first
        """
        df = df.copy()
        
        n = len(df)
        labels = np.zeros(n)
        
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        for i in range(n - max_bars):
            entry = close[i]
            upper = entry * (1 + take_profit)
            lower = entry * (1 - stop_loss)
            
            touched = 0  # 0 = time barrier, 1 = upper, -1 = lower
            
            for j in range(i + 1, min(i + max_bars + 1, n)):
                if high[j] >= upper:
                    touched = 1
                    break
                elif low[j] <= lower:
                    touched = -1
                    break
            
            labels[i] = touched
        
        df['triple_barrier_label'] = labels.astype(int)
        
        return df
    
    def calculate_forward_returns(
        self,
        df: pd.DataFrame,
        periods: list = [1, 5, 10, 20]
    ) -> pd.DataFrame:
        """
        Calculate forward returns for various horizons.
        Useful for regression-based models.
        """
        df = df.copy()
        
        for period in periods:
            df[f'forward_return_{period}'] = (
                df['close'].shift(-period) - df['close']
            ) / df['close']
        
        return df


# Example usage
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=200, freq='D')
    np.random.seed(42)
    
    price = 100 + np.random.randn(200).cumsum() * 0.5
    df = pd.DataFrame({
        'timestamp': dates,
        'open': price + np.random.randn(200) * 0.2,
        'high': price + np.abs(np.random.randn(200) * 0.5),
        'low': price - np.abs(np.random.randn(200) * 0.5),
        'close': price,
        'volume': np.random.randint(1000000, 5000000, 200)
    })
    
    labeler = TradeLabelGenerator(
        profit_threshold=0.01,  # 1%
        loss_threshold=0.005,   # 0.5%
        max_holding_period=10
    )
    
    df = labeler.generate_labels(df)
    
    print(f"\nLabel Distribution:")
    print(df['signal'].value_counts())
    print(f"\nSample with signals:")
    print(df[df['signal'] != 0][['timestamp', 'close', 'signal', 'target_price', 'stop_loss']].head(10))
