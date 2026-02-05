"""
Live Trader Module
Paper trading simulation with real-time data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import time
import json
import logging
import threading

logger = logging.getLogger(__name__)


@dataclass
class PaperPosition:
    """Represents a paper trading position."""
    symbol: str
    direction: int  # 1=Long, -1=Short
    entry_price: float
    entry_time: datetime
    quantity: float
    stop_loss: float
    take_profit: float
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    status: str = 'open'


@dataclass
class PaperTrade:
    """Represents a completed paper trade."""
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    quantity: float
    pnl: float
    pnl_pct: float
    exit_reason: str


class LiveTrader:
    """
    Paper trading engine for strategy validation.
    
    Features:
    - Real-time data fetching
    - Paper position management
    - Trade logging
    - Performance tracking
    - Signal execution simulation
    """
    
    def __init__(
        self,
        model_trainer,
        feature_engineer,
        risk_manager,
        # Capital
        initial_capital: float = 100000.0,
        # Data settings
        symbol: str = 'SPY',
        data_source: str = 'yahoo',
        timeframe: str = '1d',  # Added timeframe
        update_interval: int = 60,  # seconds
        # Trading settings
        allow_shorting: bool = False,
        max_positions: int = 1,
        # Logging
        log_dir: Optional[Path] = None,
        log_predictions: bool = True,
        log_trades: bool = True
    ):
        """
        Initialize live trader.
        
        Args:
            model_trainer: Trained ModelTrainer instance
            feature_engineer: Fitted FeatureEngineer instance
            risk_manager: RiskManager instance
            initial_capital: Starting paper capital
            symbol: Trading symbol
            data_source: Data source ('yahoo' or 'binance')
            timeframe: Data timeframe (e.g., '1d', '5m')
            update_interval: Seconds between updates
            allow_shorting: Allow short positions
            max_positions: Maximum concurrent positions
            log_dir: Directory for logs
            log_predictions: Log all predictions
            log_trades: Log all trades
        """
        self.model_trainer = model_trainer
        self.feature_engineer = feature_engineer
        self.risk_manager = risk_manager
        
        self.initial_capital = initial_capital
        self.capital = initial_capital
        
        self.symbol = symbol
        self.data_source = data_source
        self.timeframe = timeframe
        self.update_interval = update_interval
        
        self.allow_shorting = allow_shorting
        self.max_positions = max_positions
        
        # Logging
        self.log_dir = log_dir or Path(__file__).parent.parent / 'logs'
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_predictions = log_predictions
        self.log_trades = log_trades
        
        # State
        self.positions: List[PaperPosition] = []
        self.trades: List[PaperTrade] = []
        self.predictions_log: List[Dict] = []
        self.equity_history: List[Dict] = []
        self.signal_history: List[Dict] = [] # Track all signals for the session
        
        # Control
        self.is_running = False
        self.running = False # Changed from is_running
        self._stop_event = threading.Event()
        
        # Callbacks
        self.on_signal: Optional[Callable] = None
        self.on_trade: Optional[Callable] = None
        self.on_update: Optional[Callable] = None
        self.on_history: Optional[Callable] = None
        self.on_signal_history: Optional[Callable] = None # New callback for signal history
    
    def start(self, duration_minutes: Optional[int] = None):
        """
        Start the paper trading session.
        
        Args:
            duration_minutes: Optional duration to run in minutes. If None, runs indefinitely.
        """
        self.running = True
        logger.info(f"Starting paper trading for {self.symbol}...")
        
        if self.data_source == 'yahoo':
            logger.info("Note: Yahoo Finance data may be delayed. For real-time data, consider using Binance.")
            
        try:
            self._run_loop(duration_minutes)
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Error in trading loop: {e}")
            traceback.print_exc()
        finally:
            self.stop()

    def _run_loop(self, duration_minutes: Optional[int] = None):
        """Main trading loop."""
        start_time = datetime.now()
        
        history_sent = False 

        while self.running:
            try:
                loop_start = datetime.now()
                
                # Check duration
                if duration_minutes:
                    elapsed = (loop_start - start_time).total_seconds() / 60
                    if elapsed >= duration_minutes:
                        logger.info("Duration reached. Stopping session.")
                        break
                
                # Fetch latest data
                df = self._fetch_latest_data()
                
                if df is None or df.empty:
                    logger.warning("Insufficient data")
                    time.sleep(self.update_interval)
                    continue
                
                # Broadcast History (Once)
                if not history_sent:
                    if self.on_history:
                        self.on_history(df)
                    if self.on_signal_history and self.signal_history:
                        self.on_signal_history(self.signal_history)
                    history_sent = True

                # Process latest candle
                current_candle = df.iloc[-1]
                current_price = current_candle['close']
                current_time = pd.to_datetime(current_candle['timestamp']) if 'timestamp' in current_candle else df.index[-1]
                
                # Feature Engineering
                try:
                    # Generate features (using internal method as create_features doesn't exist)
                    df_features = self.feature_engineer._generate_all_features(df, generate_labels=False)
                except Exception as e:
                    logger.error(f"Feature engineering error: {e}")
                    time.sleep(self.update_interval)
                    continue
                
                # Process positions (sl/tp)
                self._update_positions(current_price, current_time)
                
                # Record equity
                self._record_equity(current_time)
                
                # Broadcast update (Full Candle)
                if self.on_update:
                    self.on_update({
                        'timestamp': current_time.isoformat(),
                        'open': float(current_candle['open']),
                        'high': float(current_candle['high']),
                        'low': float(current_candle['low']),
                        'close': float(current_candle['close']),
                        'volume': float(current_candle['volume']),
                        'equity': self.capital + sum(p.unrealized_pnl for p in self.positions),
                        'open_positions': len(self.positions)
                    })
                
                # Generate signal
                signal = self._generate_signal(df)
                
                if signal is None:
                    time.sleep(self.update_interval)
                    continue
                
                # Log prediction
                if self.log_predictions:
                    self._log_prediction(signal, current_price, current_time)
                    if signal.signal != 0:
                        self.signal_history.append(signal.to_dict())

                # Broadcast signal to UI
                if self.on_signal:
                    self.on_signal(signal)
                
                # Execute signal if conditions met
                if signal.signal != 0:
                    self._execute_signal(signal, current_price, current_time)
                
                # Wait for next update
                time.sleep(self.update_interval)

            except Exception as e:
                logger.error(f"Error in trading iteration: {e}")
                time.sleep(self.update_interval)
    
    def _fetch_latest_data(self) -> Optional[pd.DataFrame]:
        """Fetch latest OHLCV data."""
        try:
            from data import DataFetcher
            fetcher = DataFetcher()
            
            # Determine suitable period based on timeframe
            if self.timeframe in ['1m', '3m']:
                period = '5d'
            elif self.timeframe in ['5m', '10m', '15m', '30m', '1h']:
                period = '1mo'
            else:
                period = '3mo'
            
            # Fetch recent data (need history for features)
            df = fetcher.fetch_ohlcv(
                symbol=self.symbol,
                source=self.data_source,
                timeframe=self.timeframe,
                period=period  # Last 3 months for feature calculation
            )
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            return None
    
    def _generate_signal(self, df: pd.DataFrame) -> Optional[Any]:
        """Generate trading signal from latest data."""
        try:
            from models import TradingPredictor
            
            predictor = TradingPredictor(
                model_trainer=self.model_trainer,
                feature_engineer=self.feature_engineer
            )
            
            signal = predictor.predict_single(df)
            return signal
            
        except Exception as e:
            logger.error(f"Error generating signal: {e}")
            return None
    
    def _execute_signal(self, signal: Any, current_price: float, current_time: datetime):
        """Execute a trading signal."""
        # Check if we can open new position
        if len(self.positions) >= self.max_positions:
            logger.debug("Max positions reached, skipping signal")
            return
        
        # Skip shorts if not allowed
        if signal.signal == -1 and not self.allow_shorting:
            logger.debug("Shorting not allowed, skipping sell signal")
            return
        
        # Check with risk manager
        risk_check = self.risk_manager.check_trade(
            signal=signal,
            current_price=current_price,
            current_date=current_time.date()
        )
        
        if not risk_check.allowed:
            logger.info(f"Trade rejected by risk manager: {risk_check.reason}")
            return
        
        # Calculate position size
        position_value = self.capital * risk_check.adjusted_size
        quantity = position_value / current_price
        
        # Open position
        position = PaperPosition(
            symbol=self.symbol,
            direction=signal.signal,
            entry_price=current_price,
            entry_time=current_time,
            quantity=quantity,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            current_price=current_price
        )
        
        self.positions.append(position)
        self.capital -= position_value
        
        direction_str = 'LONG' if signal.signal == 1 else 'SHORT'
        logger.info(f"Opened {direction_str} position: {quantity:.2f} @ {current_price:.2f}")
        
        # Callback
        if self.on_signal:
            self.on_signal(signal, position)
    
    def _update_positions(self, current_price: float, current_time: datetime):
        """Update open positions and check for exits."""
        for position in self.positions[:]:
            position.current_price = current_price
            
            # Calculate unrealized PnL
            if position.direction == 1:
                position.unrealized_pnl = (current_price - position.entry_price) * position.quantity
            else:
                position.unrealized_pnl = (position.entry_price - current_price) * position.quantity
            
            # Check stop loss
            should_close = False
            exit_reason = ""
            exit_price = current_price
            
            if position.direction == 1:  # Long
                if current_price <= position.stop_loss:
                    should_close = True
                    exit_reason = "stop_loss"
                    exit_price = position.stop_loss
                elif current_price >= position.take_profit:
                    should_close = True
                    exit_reason = "take_profit"
                    exit_price = position.take_profit
            else:  # Short
                if current_price >= position.stop_loss:
                    should_close = True
                    exit_reason = "stop_loss"
                    exit_price = position.stop_loss
                elif current_price <= position.take_profit:
                    should_close = True
                    exit_reason = "take_profit"
                    exit_price = position.take_profit
            
            if should_close:
                self._close_position(position, exit_price, current_time, exit_reason)
    
    def _close_position(
        self,
        position: PaperPosition,
        exit_price: float,
        exit_time: datetime,
        exit_reason: str
    ):
        """Close a position and record the trade."""
        # Calculate PnL
        if position.direction == 1:
            pnl = (exit_price - position.entry_price) * position.quantity
            pnl_pct = (exit_price - position.entry_price) / position.entry_price
        else:
            pnl = (position.entry_price - exit_price) * position.quantity
            pnl_pct = (position.entry_price - exit_price) / position.entry_price
        
        # Create trade record
        trade = PaperTrade(
            symbol=position.symbol,
            direction='LONG' if position.direction == 1 else 'SHORT',
            entry_price=position.entry_price,
            exit_price=exit_price,
            entry_time=position.entry_time,
            exit_time=exit_time,
            quantity=position.quantity,
            pnl=pnl,
            pnl_pct=pnl_pct,
            exit_reason=exit_reason
        )
        
        self.trades.append(trade)
        
        # Update capital
        position_value = position.quantity * position.entry_price
        self.capital += position_value + pnl
        
        # Update risk manager
        self.risk_manager.update_trade(pnl, self.capital)
        
        # Remove position
        self.positions.remove(position)
        
        logger.info(f"Closed {trade.direction}: PnL=${pnl:.2f} ({pnl_pct*100:.2f}%), "
                   f"Reason: {exit_reason}")
        
        # Log trade
        if self.log_trades:
            self._log_trade(trade)
        
        # Callback
        if self.on_trade:
            self.on_trade(trade)
    
    def _record_equity(self, timestamp: datetime):
        """Record current equity."""
        # Calculate total equity including open positions
        equity = self.capital
        for position in self.positions:
            equity += position.quantity * position.current_price
        
        self.equity_history.append({
            'timestamp': timestamp.isoformat(),
            'equity': equity,
            'capital': self.capital,
            'open_positions': len(self.positions)
        })
    
    def _log_prediction(self, signal: Any, price: float, timestamp: datetime):
        """Log a prediction."""
        prediction = {
            'timestamp': timestamp.isoformat(),
            'symbol': self.symbol,
            'price': price,
            'signal': signal.signal,
            'signal_name': signal.signal_name,
            'confidence': signal.confidence,
            'stop_loss': signal.stop_loss,
            'take_profit': signal.take_profit
        }
        self.predictions_log.append(prediction)
        
        # Add to signal history for UI
        if signal.signal != 0:
            sig_entry = {
                'timestamp': timestamp.isoformat(),
                'signal_name': signal.signal_name,
                'direction': signal.signal,
                'price': price,
                'confidence': signal.confidence
            }
            self.signal_history.append(sig_entry)
            
            # Broadcast update if new callback exists (optional, or rely on on_signal)
            if self.on_signal_history:
                 # In a real app we might broadcast just the new one, but for simplicity here we assume
                 # on_signal handles the live event. on_signal_history is for initial load.
                 pass
    
    def _log_trade(self, trade: PaperTrade):
        """Log a trade to file."""
        trade_log = {
            'symbol': trade.symbol,
            'direction': trade.direction,
            'entry_price': trade.entry_price,
            'exit_price': trade.exit_price,
            'entry_time': trade.entry_time.isoformat(),
            'exit_time': trade.exit_time.isoformat(),
            'quantity': trade.quantity,
            'pnl': trade.pnl,
            'pnl_pct': trade.pnl_pct * 100,
            'exit_reason': trade.exit_reason
        }
        
        log_file = self.log_dir / 'trades.jsonl'
        with open(log_file, 'a') as f:
            f.write(json.dumps(trade_log) + '\n')
    
    def _save_session(self):
        """Save session data."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save equity history
        if self.equity_history:
            equity_df = pd.DataFrame(self.equity_history)
            equity_df.to_csv(self.log_dir / f'equity_{timestamp}.csv', index=False)
        
        # Save predictions log
        if self.predictions_log:
            pred_df = pd.DataFrame(self.predictions_log)
            pred_df.to_csv(self.log_dir / f'predictions_{timestamp}.csv', index=False)
        
        # Save trades
        if self.trades:
            trades_data = [{
                'symbol': t.symbol,
                'direction': t.direction,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'entry_time': t.entry_time.isoformat(),
                'exit_time': t.exit_time.isoformat(),
                'quantity': t.quantity,
                'pnl': t.pnl,
                'pnl_pct': t.pnl_pct * 100,
                'exit_reason': t.exit_reason
            } for t in self.trades]
            trades_df = pd.DataFrame(trades_data)
            trades_df.to_csv(self.log_dir / f'trades_{timestamp}.csv', index=False)
        
        logger.info(f"Session saved to {self.log_dir}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get current performance summary."""
        if not self.trades:
            return {
                'total_trades': 0,
                'net_pnl': 0,
                'win_rate': 0,
                'current_equity': self.capital
            }
        
        total_pnl = sum(t.pnl for t in self.trades)
        wins = sum(1 for t in self.trades if t.pnl > 0)
        
        return {
            'total_trades': len(self.trades),
            'winning_trades': wins,
            'losing_trades': len(self.trades) - wins,
            'net_pnl': total_pnl,
            'net_pnl_pct': (self.capital / self.initial_capital - 1) * 100,
            'win_rate': wins / len(self.trades) * 100,
            'current_equity': self.capital + sum(p.unrealized_pnl for p in self.positions),
            'open_positions': len(self.positions)
        }
    
    def close_all_positions(self, reason: str = 'manual'):
        """Close all open positions."""
        current_time = datetime.now()
        for position in self.positions[:]:
            self._close_position(
                position, 
                position.current_price, 
                current_time, 
                reason
            )
        logger.info(f"All positions closed: {reason}")


# Example usage
if __name__ == "__main__":
    print("LiveTrader requires trained models. See main.py for usage.")
