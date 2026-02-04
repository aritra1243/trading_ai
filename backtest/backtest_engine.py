"""
Backtest Engine Module
Simulates trading strategies on historical data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Represents a single trade."""
    entry_time: datetime
    exit_time: Optional[datetime]
    direction: int  # 1=Long, -1=Short
    entry_price: float
    exit_price: Optional[float]
    stop_loss: float
    take_profit: float
    position_size: float
    pnl: float = 0.0
    pnl_pct: float = 0.0
    status: str = 'open'  # open, closed, stopped, target_hit
    bars_held: int = 0
    
    def close(self, exit_price: float, exit_time: datetime, status: str = 'closed'):
        """Close the trade."""
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.status = status
        
        if self.direction == 1:  # Long
            self.pnl_pct = (exit_price - self.entry_price) / self.entry_price
        else:  # Short
            self.pnl_pct = (self.entry_price - exit_price) / self.entry_price
        
        self.pnl = self.position_size * self.pnl_pct


@dataclass
class BacktestResult:
    """Stores backtest results."""
    trades: List[Trade]
    equity_curve: pd.Series
    daily_returns: pd.Series
    metrics: Dict[str, float]
    signals_df: pd.DataFrame
    config: Dict[str, Any]


class BacktestEngine:
    """
    Backtests trading signals on historical data.
    
    Features:
    - Simulates entry, stop loss, take profit
    - Accounts for commissions and slippage
    - Position sizing
    - Equity curve tracking
    - Comprehensive metrics
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission: float = 0.001,        # 0.1%
        slippage: float = 0.0005,         # 0.05%
        position_size: float = 0.95,      # 95% of available capital
        max_positions: int = 1,
        allow_shorting: bool = True
    ):
        """
        Initialize backtest engine.
        
        Args:
            initial_capital: Starting capital
            commission: Commission rate per trade (as decimal)
            slippage: Slippage rate (as decimal)
            position_size: Fraction of capital to use per trade
            max_positions: Maximum concurrent positions
            allow_shorting: Whether to allow short positions
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.position_size = position_size
        self.max_positions = max_positions
        self.allow_shorting = allow_shorting
        
        self.reset()
    
    def reset(self):
        """Reset backtest state."""
        self.capital = self.initial_capital
        self.equity = self.initial_capital
        self.trades: List[Trade] = []
        self.open_trades: List[Trade] = []
        self.equity_history: List[Tuple[datetime, float]] = []
    
    def run(
        self,
        df: pd.DataFrame,
        signals: List[Any],  # List of TradingSignal
        use_signal_stops: bool = True
    ) -> BacktestResult:
        """
        Run backtest on historical data.
        
        Args:
            df: DataFrame with OHLCV data
            signals: List of TradingSignal objects
            use_signal_stops: Use stop loss/take profit from signals
        
        Returns:
            BacktestResult with trades, equity curve, and metrics
        """
        logger.info(f"Running backtest on {len(df)} bars...")
        self.reset()
        
        # Ensure we have matching lengths
        if len(signals) != len(df):
            raise ValueError("signals must have same length as df")
        
        # Convert signals to DataFrame for easier handling
        signals_df = pd.DataFrame([s.to_dict() for s in signals])
        signals_df.index = df.index
        
        # Run simulation bar by bar
        for i in range(len(df)):
            bar = df.iloc[i]
            signal = signals[i]
            timestamp = bar['timestamp'] if 'timestamp' in bar else df.index[i]
            
            # Update open trades (check stops and targets)
            self._update_open_trades(bar, timestamp)
            
            # Record equity
            self.equity_history.append((timestamp, self._calculate_equity(bar)))
            
            # Check for new signals
            if signal.signal != 0 and signal.confidence >= 0.6:
                # Check if we can open a new position
                if len(self.open_trades) < self.max_positions:
                    # Skip short signals if not allowed
                    if signal.signal == -1 and not self.allow_shorting:
                        continue
                    
                    # Open new trade
                    self._open_trade(bar, signal, timestamp, use_signal_stops)
        
        # Close all remaining open trades at end
        if len(df) > 0:
            final_bar = df.iloc[-1]
            final_time = final_bar['timestamp'] if 'timestamp' in final_bar else df.index[-1]
            for trade in self.open_trades[:]:
                self._close_trade(trade, final_bar['close'], final_time, 'end_of_data')
        
        # Calculate metrics
        from .metrics import PerformanceMetrics
        metrics_calc = PerformanceMetrics()
        
        equity_curve = pd.Series(
            [e[1] for e in self.equity_history],
            index=[e[0] for e in self.equity_history]
        )
        
        daily_returns = equity_curve.pct_change().dropna()
        
        metrics = metrics_calc.calculate_all(
            trades=self.trades,
            equity_curve=equity_curve,
            initial_capital=self.initial_capital
        )
        
        # Log summary
        logger.info(f"Backtest complete: {len(self.trades)} trades")
        logger.info(f"Final capital: ${self.capital:,.2f} "
                   f"(Return: {(self.capital/self.initial_capital - 1)*100:.2f}%)")
        
        return BacktestResult(
            trades=self.trades,
            equity_curve=equity_curve,
            daily_returns=daily_returns,
            metrics=metrics,
            signals_df=signals_df,
            config={
                'initial_capital': self.initial_capital,
                'commission': self.commission,
                'slippage': self.slippage,
                'position_size': self.position_size
            }
        )
    
    def _open_trade(
        self,
        bar: pd.Series,
        signal: Any,
        timestamp: datetime,
        use_signal_stops: bool
    ):
        """Open a new trade."""
        # Calculate entry price with slippage
        if signal.signal == 1:  # Long
            entry_price = bar['close'] * (1 + self.slippage)
        else:  # Short
            entry_price = bar['close'] * (1 - self.slippage)
        
        # Calculate position size
        available_capital = self.capital * self.position_size
        position_value = available_capital
        
        # Apply commission
        position_value *= (1 - self.commission)
        
        # Use signal stops or defaults
        if use_signal_stops:
            stop_loss = signal.stop_loss
            take_profit = signal.take_profit
        else:
            if signal.signal == 1:
                stop_loss = entry_price * 0.98
                take_profit = entry_price * 1.04
            else:
                stop_loss = entry_price * 1.02
                take_profit = entry_price * 0.96
        
        trade = Trade(
            entry_time=timestamp,
            exit_time=None,
            direction=signal.signal,
            entry_price=entry_price,
            exit_price=None,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=position_value
        )
        
        self.open_trades.append(trade)
        self.capital -= position_value
        
        logger.debug(f"Opened {'LONG' if signal.signal == 1 else 'SHORT'} at {entry_price:.4f}")
    
    def _update_open_trades(self, bar: pd.Series, timestamp: datetime):
        """Update open trades, check for stops and targets."""
        for trade in self.open_trades[:]:  # Copy list to allow modification
            trade.bars_held += 1
            
            # Check stop loss
            if trade.direction == 1:  # Long
                if bar['low'] <= trade.stop_loss:
                    self._close_trade(trade, trade.stop_loss, timestamp, 'stopped')
                    continue
                elif bar['high'] >= trade.take_profit:
                    self._close_trade(trade, trade.take_profit, timestamp, 'target_hit')
                    continue
            else:  # Short
                if bar['high'] >= trade.stop_loss:
                    self._close_trade(trade, trade.stop_loss, timestamp, 'stopped')
                    continue
                elif bar['low'] <= trade.take_profit:
                    self._close_trade(trade, trade.take_profit, timestamp, 'target_hit')
                    continue
    
    def _close_trade(
        self,
        trade: Trade,
        exit_price: float,
        exit_time: datetime,
        status: str
    ):
        """Close a trade."""
        # Apply slippage to exit
        if trade.direction == 1:  # Long
            exit_price *= (1 - self.slippage)
        else:  # Short
            exit_price *= (1 + self.slippage)
        
        trade.close(exit_price, exit_time, status)
        
        # Apply commission
        trade.pnl *= (1 - self.commission)
        
        # Update capital
        self.capital += trade.position_size + trade.pnl
        
        # Move from open to closed
        self.open_trades.remove(trade)
        self.trades.append(trade)
        
        logger.debug(f"Closed trade: {status}, PnL: {trade.pnl:.2f} ({trade.pnl_pct*100:.2f}%)")
    
    def _calculate_equity(self, bar: pd.Series) -> float:
        """Calculate current equity including open positions."""
        equity = self.capital
        
        for trade in self.open_trades:
            current_price = bar['close']
            if trade.direction == 1:
                unrealized_pnl = (current_price - trade.entry_price) / trade.entry_price
            else:
                unrealized_pnl = (trade.entry_price - current_price) / trade.entry_price
            
            equity += trade.position_size * (1 + unrealized_pnl)
        
        return equity
    
    def get_trade_summary(self) -> pd.DataFrame:
        """Get summary DataFrame of all trades."""
        if not self.trades:
            return pd.DataFrame()
        
        data = []
        for trade in self.trades:
            data.append({
                'entry_time': trade.entry_time,
                'exit_time': trade.exit_time,
                'direction': 'LONG' if trade.direction == 1 else 'SHORT',
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'stop_loss': trade.stop_loss,
                'take_profit': trade.take_profit,
                'position_size': trade.position_size,
                'pnl': trade.pnl,
                'pnl_pct': trade.pnl_pct * 100,
                'status': trade.status,
                'bars_held': trade.bars_held
            })
        
        return pd.DataFrame(data)


# Example usage
if __name__ == "__main__":
    print("BacktestEngine requires signals. See main.py for usage.")
