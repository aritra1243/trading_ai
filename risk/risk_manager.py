"""
Risk Manager Module
Implements comprehensive risk management for trading.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, date
import logging

logger = logging.getLogger(__name__)


@dataclass
class RiskState:
    """Tracks current risk state."""
    current_capital: float
    peak_capital: float
    daily_pnl: float
    daily_trades: int
    current_drawdown_pct: float
    is_killed: bool = False
    kill_reason: str = ""
    last_reset: date = field(default_factory=date.today)
    
    def reset_daily(self, current_date: date):
        """Reset daily counters."""
        if current_date != self.last_reset:
            self.daily_pnl = 0
            self.daily_trades = 0
            self.last_reset = current_date


@dataclass
class RiskCheck:
    """Result of a risk check."""
    allowed: bool
    reason: str
    adjusted_size: float  # Suggested position size
    warning: Optional[str] = None


class RiskManager:
    """
    Comprehensive risk management for trading.
    
    Features:
    - Per-trade risk limits
    - Daily loss limits
    - Maximum drawdown protection (kill switch)
    - Position sizing with volatility adjustment
    - Correlation checks (future)
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        # Per-trade limits
        max_risk_per_trade: float = 0.02,     # 2% max risk per trade
        max_position_pct: float = 0.25,        # 25% max position size
        min_position_pct: float = 0.05,        # 5% min position size
        # Daily limits
        max_daily_loss: float = 0.05,          # 5% max daily loss
        max_daily_trades: int = 10,
        # Drawdown protection
        max_drawdown: float = 0.15,            # 15% max drawdown kill switch
        drawdown_reduction_start: float = 0.10, # Start reducing size at 10% DD
        # Volatility adjustment
        use_volatility_sizing: bool = True,
        target_volatility: float = 0.15,       # 15% target volatility
        atr_multiplier: float = 2.0,
        # General
        cooling_period_trades: int = 3         # Trades to skip after loss streak
    ):
        """Initialize risk manager."""
        self.initial_capital = initial_capital
        
        # Per-trade limits
        self.max_risk_per_trade = max_risk_per_trade
        self.max_position_pct = max_position_pct
        self.min_position_pct = min_position_pct
        
        # Daily limits
        self.max_daily_loss = max_daily_loss
        self.max_daily_trades = max_daily_trades
        
        # Drawdown protection
        self.max_drawdown = max_drawdown
        self.drawdown_reduction_start = drawdown_reduction_start
        
        # Volatility sizing
        self.use_volatility_sizing = use_volatility_sizing
        self.target_volatility = target_volatility
        self.atr_multiplier = atr_multiplier
        
        # Cooling period
        self.cooling_period_trades = cooling_period_trades
        
        # Initialize state
        self.state = RiskState(
            current_capital=initial_capital,
            peak_capital=initial_capital,
            daily_pnl=0,
            daily_trades=0,
            current_drawdown_pct=0
        )
        
        # Trade history for streak detection
        self.recent_trades: List[float] = []  # Recent PnL values
    
    def check_trade(
        self,
        signal: Any,
        current_price: float,
        current_date: date,
        volatility: Optional[float] = None
    ) -> RiskCheck:
        """
        Check if a trade is allowed and calculate position size.
        
        Args:
            signal: TradingSignal object
            current_price: Current market price
            current_date: Current date for daily reset
            volatility: Current volatility (ATR or similar)
        
        Returns:
            RiskCheck with allowed status and position size
        """
        # Reset daily counters if new day
        self.state.reset_daily(current_date)
        
        # Check kill switch
        if self.state.is_killed:
            return RiskCheck(
                allowed=False,
                reason=f"Kill switch active: {self.state.kill_reason}",
                adjusted_size=0
            )
        
        # Check drawdown kill switch
        if self.state.current_drawdown_pct >= self.max_drawdown:
            self.state.is_killed = True
            self.state.kill_reason = f"Max drawdown {self.max_drawdown*100:.1f}% exceeded"
            logger.warning(f"KILL SWITCH ACTIVATED: {self.state.kill_reason}")
            return RiskCheck(
                allowed=False,
                reason=self.state.kill_reason,
                adjusted_size=0
            )
        
        # Check daily loss limit
        if self.state.daily_pnl <= -self.max_daily_loss * self.state.current_capital:
            return RiskCheck(
                allowed=False,
                reason=f"Daily loss limit {self.max_daily_loss*100:.1f}% reached",
                adjusted_size=0
            )
        
        # Check daily trade limit
        if self.state.daily_trades >= self.max_daily_trades:
            return RiskCheck(
                allowed=False,
                reason=f"Daily trade limit {self.max_daily_trades} reached",
                adjusted_size=0
            )
        
        # Check loss streak cooling period
        if self._check_loss_streak():
            return RiskCheck(
                allowed=False,
                reason="Cooling period after loss streak",
                adjusted_size=0,
                warning="Consider reviewing strategy after consecutive losses"
            )
        
        # Calculate position size
        position_size = self._calculate_position_size(
            signal, current_price, volatility
        )
        
        # Apply drawdown reduction
        if self.state.current_drawdown_pct >= self.drawdown_reduction_start:
            reduction_factor = 1 - (
                (self.state.current_drawdown_pct - self.drawdown_reduction_start) /
                (self.max_drawdown - self.drawdown_reduction_start)
            )
            reduction_factor = max(0.25, reduction_factor)  # Min 25% of normal size
            position_size *= reduction_factor
            
            warning = f"Position reduced to {reduction_factor*100:.0f}% due to {self.state.current_drawdown_pct*100:.1f}% drawdown"
        else:
            warning = None
        
        # Final bounds check
        position_size = min(position_size, self.max_position_pct)
        position_size = max(position_size, self.min_position_pct)
        
        return RiskCheck(
            allowed=True,
            reason="Trade approved",
            adjusted_size=position_size,
            warning=warning
        )
    
    def _calculate_position_size(
        self,
        signal: Any,
        current_price: float,
        volatility: Optional[float] = None
    ) -> float:
        """Calculate position size based on risk parameters."""
        # Base position from signal confidence
        base_size = signal.position_size_pct if hasattr(signal, 'position_size_pct') else 0.1
        
        # Risk-based sizing
        stop_distance = abs(current_price - signal.stop_loss) / current_price
        if stop_distance > 0:
            risk_based_size = self.max_risk_per_trade / stop_distance
        else:
            risk_based_size = self.max_position_pct
        
        # Volatility-based sizing
        if self.use_volatility_sizing and volatility is not None and volatility > 0:
            # Scale position inversely with volatility
            vol_ratio = self.target_volatility / (volatility * np.sqrt(252))
            vol_based_size = self.max_position_pct * min(vol_ratio, 1.5)
        else:
            vol_based_size = self.max_position_pct
        
        # Use the most conservative size
        position_size = min(base_size, risk_based_size, vol_based_size, self.max_position_pct)
        
        return position_size
    
    def _check_loss_streak(self) -> bool:
        """Check if in a losing streak requiring cooling period."""
        if len(self.recent_trades) < self.cooling_period_trades:
            return False
        
        # Check if last N trades were all losses
        recent = self.recent_trades[-self.cooling_period_trades:]
        return all(pnl < 0 for pnl in recent)
    
    def update_trade(self, pnl: float, current_capital: float):
        """
        Update risk state after a trade.
        
        Args:
            pnl: Trade profit/loss
            current_capital: Current capital after trade
        """
        # Update capital tracking
        self.state.current_capital = current_capital
        self.state.peak_capital = max(self.state.peak_capital, current_capital)
        
        # Update drawdown
        self.state.current_drawdown_pct = (
            (self.state.peak_capital - current_capital) / self.state.peak_capital
        )
        
        # Update daily stats
        self.state.daily_pnl += pnl
        self.state.daily_trades += 1
        
        # Track for loss streak
        self.recent_trades.append(pnl)
        if len(self.recent_trades) > 10:
            self.recent_trades.pop(0)
        
        logger.debug(f"Trade updated: PnL={pnl:.2f}, Capital={current_capital:.2f}, "
                    f"DD={self.state.current_drawdown_pct*100:.1f}%")
    
    def reset_kill_switch(self, new_capital: Optional[float] = None):
        """Reset kill switch (use with caution)."""
        self.state.is_killed = False
        self.state.kill_reason = ""
        
        if new_capital:
            self.state.current_capital = new_capital
            self.state.peak_capital = new_capital
            self.state.current_drawdown_pct = 0
        
        logger.info("Kill switch reset")
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get current risk state summary."""
        return {
            'current_capital': self.state.current_capital,
            'peak_capital': self.state.peak_capital,
            'drawdown_pct': self.state.current_drawdown_pct * 100,
            'daily_pnl': self.state.daily_pnl,
            'daily_trades': self.state.daily_trades,
            'is_killed': self.state.is_killed,
            'kill_reason': self.state.kill_reason,
            'recent_win_rate': self._get_recent_win_rate(),
            'capital_at_risk': self._get_capital_at_risk()
        }
    
    def _get_recent_win_rate(self) -> float:
        """Calculate recent win rate."""
        if not self.recent_trades:
            return 0
        wins = sum(1 for pnl in self.recent_trades if pnl > 0)
        return wins / len(self.recent_trades) * 100
    
    def _get_capital_at_risk(self) -> float:
        """Calculate current capital at risk."""
        return self.state.current_capital * self.max_risk_per_trade
    
    def calculate_var(
        self,
        returns: pd.Series,
        confidence: float = 0.95,
        horizon_days: int = 1
    ) -> float:
        """
        Calculate Value at Risk.
        
        Args:
            returns: Historical returns
            confidence: Confidence level (0.95 = 95%)
            horizon_days: Time horizon in days
        
        Returns:
            VaR as a positive percentage
        """
        if len(returns) < 30:
            return self.max_risk_per_trade * 100
        
        # Parametric VaR
        mean = returns.mean()
        std = returns.std()
        z_score = abs(np.percentile(np.random.randn(10000), (1 - confidence) * 100))
        
        var = -(mean - z_score * std) * np.sqrt(horizon_days)
        return max(var * 100, 0)
    
    def suggest_position_size(
        self,
        entry_price: float,
        stop_loss: float,
        account_value: float,
        confidence: float = 0.5
    ) -> Dict[str, float]:
        """
        Suggest optimal position size.
        
        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            account_value: Current account value
            confidence: Signal confidence (0-1)
        
        Returns:
            Dictionary with position size details
        """
        # Risk per share
        risk_per_share = abs(entry_price - stop_loss)
        
        # Account risk (adjusted by confidence)
        effective_risk = self.max_risk_per_trade * confidence
        account_risk = account_value * effective_risk
        
        # Shares to buy
        if risk_per_share > 0:
            shares = account_risk / risk_per_share
            position_value = shares * entry_price
            position_pct = position_value / account_value
        else:
            shares = 0
            position_value = 0
            position_pct = 0
        
        # Cap at max position
        if position_pct > self.max_position_pct:
            position_pct = self.max_position_pct
            position_value = account_value * position_pct
            shares = position_value / entry_price
        
        return {
            'shares': int(shares),
            'position_value': position_value,
            'position_pct': position_pct * 100,
            'risk_amount': account_risk,
            'risk_per_share': risk_per_share
        }


# Example usage
if __name__ == "__main__":
    from dataclasses import dataclass
    
    @dataclass
    class MockSignal:
        signal: int
        stop_loss: float
        position_size_pct: float
    
    rm = RiskManager(initial_capital=100000)
    
    # Simulate some trades
    signal = MockSignal(signal=1, stop_loss=95.0, position_size_pct=0.1)
    
    check = rm.check_trade(signal, current_price=100, current_date=date.today())
    print(f"Trade allowed: {check.allowed}")
    print(f"Position size: {check.adjusted_size*100:.1f}%")
    
    # Update with a loss
    rm.update_trade(pnl=-1000, current_capital=99000)
    
    print(f"\nRisk Summary:")
    for key, value in rm.get_risk_summary().items():
        print(f"  {key}: {value}")
