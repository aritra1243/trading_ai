"""
Performance Metrics Module
Calculates trading performance metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class PerformanceMetrics:
    """
    Calculates comprehensive trading performance metrics.
    
    Metrics include:
    - Return metrics (total, annual, CAGR)
    - Risk metrics (volatility, max drawdown, VaR)
    - Risk-adjusted metrics (Sharpe, Sortino, Calmar)
    - Trade metrics (win rate, profit factor, expectancy)
    """
    
    def __init__(
        self,
        risk_free_rate: float = 0.02,  # 2% annual
        trading_days: int = 252
    ):
        """
        Initialize metrics calculator.
        
        Args:
            risk_free_rate: Annual risk-free rate
            trading_days: Trading days per year
        """
        self.risk_free_rate = risk_free_rate
        self.trading_days = trading_days
    
    def calculate_all(
        self,
        trades: List[Any],
        equity_curve: pd.Series,
        initial_capital: float
    ) -> Dict[str, float]:
        """
        Calculate all performance metrics.
        
        Args:
            trades: List of Trade objects
            equity_curve: Series of equity values over time
            initial_capital: Starting capital
        
        Returns:
            Dictionary of metric name to value
        """
        metrics = {}
        
        # Return metrics
        returns_metrics = self.calculate_returns(equity_curve, initial_capital)
        metrics.update(returns_metrics)
        
        # Risk metrics
        risk_metrics = self.calculate_risk_metrics(equity_curve)
        metrics.update(risk_metrics)
        
        # Risk-adjusted metrics
        adjusted_metrics = self.calculate_risk_adjusted_metrics(equity_curve)
        metrics.update(adjusted_metrics)
        
        # Trade metrics
        if trades:
            trade_metrics = self.calculate_trade_metrics(trades)
            metrics.update(trade_metrics)
        
        return metrics
    
    def calculate_returns(
        self,
        equity_curve: pd.Series,
        initial_capital: float
    ) -> Dict[str, float]:
        """Calculate return-related metrics."""
        if len(equity_curve) < 2:
            return {
                'total_return': 0,
                'total_return_pct': 0,
                'annual_return': 0,
                'cagr': 0
            }
        
        final_equity = equity_curve.iloc[-1]
        total_return = final_equity - initial_capital
        total_return_pct = (final_equity / initial_capital - 1) * 100
        
        # Estimate trading days in data
        n_days = len(equity_curve)
        years = n_days / self.trading_days
        
        # Annualized return
        daily_returns = equity_curve.pct_change().dropna()
        annual_return = (daily_returns.mean() * self.trading_days) * 100 if len(daily_returns) > 0 else 0
        
        # CAGR
        if years > 0 and initial_capital > 0:
            cagr = ((final_equity / initial_capital) ** (1 / years) - 1) * 100
        else:
            cagr = 0
        
        return {
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'annual_return': annual_return,
            'cagr': cagr
        }
    
    def calculate_risk_metrics(self, equity_curve: pd.Series) -> Dict[str, float]:
        """Calculate risk-related metrics."""
        if len(equity_curve) < 2:
            return {
                'volatility': 0,
                'max_drawdown': 0,
                'max_drawdown_pct': 0,
                'var_95': 0,
                'cvar_95': 0
            }
        
        daily_returns = equity_curve.pct_change().dropna()
        
        # Volatility (annualized)
        volatility = daily_returns.std() * np.sqrt(self.trading_days) * 100
        
        # Maximum drawdown
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak
        max_drawdown = abs(drawdown.min()) * 100
        max_drawdown_value = (peak - equity_curve).max()
        
        # Value at Risk (95%)
        var_95 = abs(np.percentile(daily_returns, 5)) * 100 if len(daily_returns) > 0 else 0
        
        # Conditional VaR (Expected Shortfall)
        var_threshold = np.percentile(daily_returns, 5) if len(daily_returns) > 0 else 0
        cvar_95 = abs(daily_returns[daily_returns <= var_threshold].mean()) * 100 if len(daily_returns[daily_returns <= var_threshold]) > 0 else 0
        
        return {
            'volatility': volatility,
            'max_drawdown': max_drawdown_value,
            'max_drawdown_pct': max_drawdown,
            'var_95': var_95,
            'cvar_95': cvar_95
        }
    
    def calculate_risk_adjusted_metrics(self, equity_curve: pd.Series) -> Dict[str, float]:
        """Calculate risk-adjusted return metrics."""
        if len(equity_curve) < 2:
            return {
                'sharpe_ratio': 0,
                'sortino_ratio': 0,
                'calmar_ratio': 0
            }
        
        daily_returns = equity_curve.pct_change().dropna()
        
        if len(daily_returns) == 0:
            return {
                'sharpe_ratio': 0,
                'sortino_ratio': 0,
                'calmar_ratio': 0
            }
        
        # Daily risk-free rate
        daily_rf = self.risk_free_rate / self.trading_days
        excess_returns = daily_returns - daily_rf
        
        # Sharpe Ratio
        if daily_returns.std() > 0:
            sharpe = (excess_returns.mean() / daily_returns.std()) * np.sqrt(self.trading_days)
        else:
            sharpe = 0
        
        # Sortino Ratio (uses downside deviation)
        downside_returns = daily_returns[daily_returns < 0]
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            sortino = (excess_returns.mean() / downside_returns.std()) * np.sqrt(self.trading_days)
        else:
            sortino = 0
        
        # Calmar Ratio (annual return / max drawdown)
        annual_return = daily_returns.mean() * self.trading_days
        peak = equity_curve.expanding().max()
        max_dd = abs((equity_curve - peak) / peak).max()
        
        if max_dd > 0:
            calmar = annual_return / max_dd
        else:
            calmar = 0
        
        return {
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar
        }
    
    def calculate_trade_metrics(self, trades: List[Any]) -> Dict[str, float]:
        """Calculate trade-level metrics."""
        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'expectancy': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'largest_win': 0,
                'largest_loss': 0,
                'avg_bars_held': 0
            }
        
        pnls = [t.pnl for t in trades]
        pnl_pcts = [t.pnl_pct for t in trades]
        bars_held = [t.bars_held for t in trades]
        
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        
        win_pcts = [p for p in pnl_pcts if p > 0]
        loss_pcts = [p for p in pnl_pcts if p < 0]
        
        total_trades = len(trades)
        win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0
        
        # Profit Factor
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0
        
        # Expectancy (average profit per trade)
        expectancy = np.mean(pnls) if pnls else 0
        
        # Average win/loss
        avg_win = np.mean(win_pcts) * 100 if win_pcts else 0
        avg_loss = np.mean(loss_pcts) * 100 if loss_pcts else 0
        
        # Largest win/loss
        largest_win = max(pnl_pcts) * 100 if pnl_pcts else 0
        largest_loss = min(pnl_pcts) * 100 if pnl_pcts else 0
        
        # Average holding period
        avg_bars = np.mean(bars_held) if bars_held else 0
        
        # Trade outcomes by status
        stopped = len([t for t in trades if t.status == 'stopped'])
        target_hit = len([t for t in trades if t.status == 'target_hit'])
        
        return {
            'total_trades': total_trades,
            'winning_trades': len(wins),
            'losing_trades': len(losses),
            'win_rate': win_rate,
            'profit_factor': profit_factor if profit_factor != float('inf') else 999.99,
            'expectancy': expectancy,
            'avg_win_pct': avg_win,
            'avg_loss_pct': avg_loss,
            'largest_win_pct': largest_win,
            'largest_loss_pct': largest_loss,
            'avg_bars_held': avg_bars,
            'trades_stopped': stopped,
            'trades_target_hit': target_hit
        }
    
    def calculate_monthly_returns(self, equity_curve: pd.Series) -> pd.DataFrame:
        """Calculate monthly returns."""
        if len(equity_curve) < 2:
            return pd.DataFrame()
        
        # Resample to monthly
        monthly = equity_curve.resample('M').last()
        monthly_returns = monthly.pct_change().dropna() * 100
        
        # Create pivot table by year and month
        monthly_returns.index = pd.to_datetime(monthly_returns.index)
        
        df = pd.DataFrame({
            'year': monthly_returns.index.year,
            'month': monthly_returns.index.month,
            'return': monthly_returns.values
        })
        
        pivot = df.pivot(index='year', columns='month', values='return')
        pivot.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][:len(pivot.columns)]
        
        return pivot
    
    def generate_report(
        self,
        trades: List[Any],
        equity_curve: pd.Series,
        initial_capital: float
    ) -> str:
        """Generate a text performance report."""
        metrics = self.calculate_all(trades, equity_curve, initial_capital)
        
        report = """
═══════════════════════════════════════════════════════════════
                    BACKTEST PERFORMANCE REPORT
═══════════════════════════════════════════════════════════════

RETURN METRICS
──────────────────────────────────────────────────────────────
  Total Return:          ${:>12,.2f}  ({:>7.2f}%)
  Annual Return:         {:>12.2f}%
  CAGR:                  {:>12.2f}%

RISK METRICS  
──────────────────────────────────────────────────────────────
  Volatility (annual):   {:>12.2f}%
  Max Drawdown:          {:>12.2f}%
  Value at Risk (95%):   {:>12.2f}%

RISK-ADJUSTED METRICS
──────────────────────────────────────────────────────────────
  Sharpe Ratio:          {:>12.2f}
  Sortino Ratio:         {:>12.2f}
  Calmar Ratio:          {:>12.2f}

TRADE STATISTICS
──────────────────────────────────────────────────────────────
  Total Trades:          {:>12}
  Win Rate:              {:>12.2f}%
  Profit Factor:         {:>12.2f}
  Expectancy:            ${:>11,.2f}
  
  Avg Win:               {:>12.2f}%
  Avg Loss:              {:>12.2f}%
  Largest Win:           {:>12.2f}%
  Largest Loss:          {:>12.2f}%
  
  Avg Bars Held:         {:>12.1f}

═══════════════════════════════════════════════════════════════
""".format(
            metrics.get('total_return', 0),
            metrics.get('total_return_pct', 0),
            metrics.get('annual_return', 0),
            metrics.get('cagr', 0),
            
            metrics.get('volatility', 0),
            metrics.get('max_drawdown_pct', 0),
            metrics.get('var_95', 0),
            
            metrics.get('sharpe_ratio', 0),
            metrics.get('sortino_ratio', 0),
            metrics.get('calmar_ratio', 0),
            
            metrics.get('total_trades', 0),
            metrics.get('win_rate', 0),
            metrics.get('profit_factor', 0),
            metrics.get('expectancy', 0),
            
            metrics.get('avg_win_pct', 0),
            metrics.get('avg_loss_pct', 0),
            metrics.get('largest_win_pct', 0),
            metrics.get('largest_loss_pct', 0),
            
            metrics.get('avg_bars_held', 0)
        )
        
        return report


# Example usage
if __name__ == "__main__":
    # Create sample equity curve
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    returns = np.random.randn(252) * 0.02 + 0.0005
    equity = 100000 * (1 + returns).cumprod()
    equity_curve = pd.Series(equity, index=dates)
    
    metrics = PerformanceMetrics()
    all_metrics = metrics.calculate_all([], equity_curve, 100000)
    
    print("Performance Metrics:")
    for key, value in all_metrics.items():
        print(f"  {key}: {value:.4f}")
