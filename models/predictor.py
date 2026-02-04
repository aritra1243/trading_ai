"""
Trading Predictor Module
Generates trading predictions with entry, stop loss, take profit, and confidence.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class TradingSignal:
    """Represents a trading signal with all relevant information."""
    timestamp: pd.Timestamp
    signal: int  # 1=Buy, -1=Sell, 0=Hold
    signal_name: str  # 'BUY', 'SELL', 'HOLD'
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float  # 0-1 probability
    risk_reward_ratio: float
    position_size_pct: float  # Suggested position size as % of capital
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'signal': self.signal,
            'signal_name': self.signal_name,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'confidence': self.confidence,
            'risk_reward_ratio': self.risk_reward_ratio,
            'position_size_pct': self.position_size_pct
        }


class TradingPredictor:
    """
    Generates complete trading signals from model predictions.
    
    Outputs:
    - Signal direction (Buy/Sell/Hold)
    - Entry price
    - Stop loss
    - Take profit
    - Confidence score
    - Position sizing suggestion
    """
    
    def __init__(
        self,
        model_trainer,
        feature_engineer,
        # Trade parameters
        default_stop_pct: float = 0.02,     # 2% stop loss
        default_target_pct: float = 0.04,   # 4% take profit
        min_confidence: float = 0.6,        # Minimum confidence to signal
        use_atr_stops: bool = True,         # Use ATR for dynamic stops
        atr_stop_multiplier: float = 2.0,   # ATR multiplier for stops
        atr_target_multiplier: float = 3.0, # ATR multiplier for targets
        # Position sizing
        max_risk_per_trade: float = 0.02,   # 2% max risk per trade
        max_position_size: float = 0.25     # 25% max position
    ):
        """
        Initialize trading predictor.
        
        Args:
            model_trainer: Trained ModelTrainer instance
            feature_engineer: Fitted FeatureEngineer instance
            default_stop_pct: Default stop loss percentage
            default_target_pct: Default take profit percentage
            min_confidence: Minimum confidence to generate signal
            use_atr_stops: Use ATR-based dynamic stops
            atr_stop_multiplier: ATR multiplier for stop loss
            atr_target_multiplier: ATR multiplier for take profit
            max_risk_per_trade: Maximum risk per trade
            max_position_size: Maximum position size
        """
        self.model_trainer = model_trainer
        self.feature_engineer = feature_engineer
        
        self.default_stop_pct = default_stop_pct
        self.default_target_pct = default_target_pct
        self.min_confidence = min_confidence
        self.use_atr_stops = use_atr_stops
        self.atr_stop_multiplier = atr_stop_multiplier
        self.atr_target_multiplier = atr_target_multiplier
        
        self.max_risk_per_trade = max_risk_per_trade
        self.max_position_size = max_position_size
    
    def predict(self, df: pd.DataFrame) -> List[TradingSignal]:
        """
        Generate trading signals for the given data.
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            List of TradingSignal objects
        """
        logger.info(f"Generating predictions for {len(df)} rows...")
        
        # Generate features
        full_df, X = self.feature_engineer.transform(df)
        
        # Get model predictions
        predictions = self.model_trainer.predict(X)
        probabilities = self.model_trainer.predict_proba(X)
        
        # Generate signals
        signals = []
        
        for i in range(len(predictions)):
            signal = self._create_signal(
                timestamp=full_df.iloc[i]['timestamp'],
                prediction=predictions[i],
                probabilities=probabilities[i],
                current_price=full_df.iloc[i]['close'],
                atr=full_df.iloc[i].get('atr', None)
            )
            signals.append(signal)
        
        # Filter by confidence
        high_confidence = [s for s in signals if s.confidence >= self.min_confidence and s.signal != 0]
        logger.info(f"Generated {len(high_confidence)} high-confidence signals")
        
        return signals
    
    def predict_single(
        self,
        df: pd.DataFrame,
        current_price: Optional[float] = None
    ) -> TradingSignal:
        """
        Generate a single trading signal for the latest data point.
        
        Args:
            df: DataFrame with OHLCV data (needs enough history for features)
            current_price: Current market price (uses last close if not provided)
        
        Returns:
            Single TradingSignal object
        """
        # Generate features
        full_df, X = self.feature_engineer.transform(df)
        
        # Use only the last row
        X_last = X[-1:] if len(X.shape) > 1 else X[-1].reshape(1, -1)
        
        # Get prediction
        prediction = self.model_trainer.predict(X_last)[0]
        probabilities = self.model_trainer.predict_proba(X_last)[0]
        
        # Get ATR if available
        atr = full_df.iloc[-1].get('atr', None)
        
        price = current_price or full_df.iloc[-1]['close']
        
        return self._create_signal(
            timestamp=full_df.iloc[-1]['timestamp'],
            prediction=prediction,
            probabilities=probabilities,
            current_price=price,
            atr=atr
        )
    
    def _create_signal(
        self,
        timestamp: pd.Timestamp,
        prediction: int,
        probabilities: np.ndarray,
        current_price: float,
        atr: Optional[float] = None
    ) -> TradingSignal:
        """Create a TradingSignal from model output."""
        # Map prediction to signal
        signal = int(prediction)
        signal_name = {1: 'BUY', -1: 'SELL', 0: 'HOLD'}.get(signal, 'HOLD')
        
        # Get confidence (probability of predicted class)
        class_idx = list(self.model_trainer.classes_).index(prediction)
        confidence = float(probabilities[class_idx])
        
        # Calculate stop loss and take profit
        if self.use_atr_stops and atr is not None and not np.isnan(atr):
            stop_distance = atr * self.atr_stop_multiplier
            target_distance = atr * self.atr_target_multiplier
        else:
            stop_distance = current_price * self.default_stop_pct
            target_distance = current_price * self.default_target_pct
        
        if signal == 1:  # Buy
            entry_price = current_price
            stop_loss = current_price - stop_distance
            take_profit = current_price + target_distance
        elif signal == -1:  # Sell
            entry_price = current_price
            stop_loss = current_price + stop_distance
            take_profit = current_price - target_distance
        else:  # Hold
            entry_price = current_price
            stop_loss = current_price
            take_profit = current_price
        
        # Calculate risk/reward ratio
        if signal != 0:
            risk = abs(entry_price - stop_loss)
            reward = abs(take_profit - entry_price)
            risk_reward_ratio = reward / risk if risk > 0 else 0
        else:
            risk_reward_ratio = 0
        
        # Calculate position size based on risk
        position_size_pct = self._calculate_position_size(
            entry_price, stop_loss, confidence
        )
        
        return TradingSignal(
            timestamp=timestamp,
            signal=signal,
            signal_name=signal_name,
            entry_price=round(entry_price, 4),
            stop_loss=round(stop_loss, 4),
            take_profit=round(take_profit, 4),
            confidence=round(confidence, 4),
            risk_reward_ratio=round(risk_reward_ratio, 2),
            position_size_pct=round(position_size_pct, 4)
        )
    
    def _calculate_position_size(
        self,
        entry_price: float,
        stop_loss: float,
        confidence: float
    ) -> float:
        """
        Calculate position size based on risk management.
        
        Uses the formula:
        Position Size = (Account Risk) / (Trade Risk)
        
        Where:
        - Account Risk = Capital * Max Risk Per Trade
        - Trade Risk = |Entry - Stop Loss| / Entry
        """
        trade_risk_pct = abs(entry_price - stop_loss) / entry_price
        
        if trade_risk_pct == 0:
            return 0
        
        # Base position size from risk management
        base_position = self.max_risk_per_trade / trade_risk_pct
        
        # Adjust by confidence
        confidence_adjusted = base_position * confidence
        
        # Cap at maximum position size
        position_size = min(confidence_adjusted, self.max_position_size)
        
        return float(position_size)
    
    def get_signals_summary(
        self,
        signals: List[TradingSignal],
        min_confidence: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Get a summary DataFrame of trading signals.
        
        Args:
            signals: List of TradingSignal objects
            min_confidence: Filter by minimum confidence
        
        Returns:
            DataFrame with signal information
        """
        min_conf = min_confidence or self.min_confidence
        
        filtered = [s for s in signals if s.confidence >= min_conf]
        
        if not filtered:
            return pd.DataFrame()
        
        df = pd.DataFrame([s.to_dict() for s in filtered])
        return df
    
    def analyze_signal_quality(
        self,
        signals: List[TradingSignal],
        actual_returns: pd.Series
    ) -> Dict[str, Any]:
        """
        Analyze the quality of generated signals against actual returns.
        
        Args:
            signals: List of generated signals
            actual_returns: Series of actual forward returns
        
        Returns:
            Dictionary with quality metrics
        """
        if len(signals) != len(actual_returns):
            raise ValueError("signals and actual_returns must have same length")
        
        results = {
            'total_signals': 0,
            'buy_signals': 0,
            'sell_signals': 0,
            'correct_direction': 0,
            'hit_target': 0,
            'hit_stop': 0,
            'avg_confidence': 0,
            'accuracy_by_confidence': {}
        }
        
        confidences = []
        correct = []
        
        for signal, ret in zip(signals, actual_returns):
            if signal.signal == 0:
                continue
            
            results['total_signals'] += 1
            confidences.append(signal.confidence)
            
            if signal.signal == 1:
                results['buy_signals'] += 1
                is_correct = ret > 0
            else:
                results['sell_signals'] += 1
                is_correct = ret < 0
            
            if is_correct:
                results['correct_direction'] += 1
            correct.append(is_correct)
        
        if results['total_signals'] > 0:
            results['avg_confidence'] = np.mean(confidences)
            results['direction_accuracy'] = results['correct_direction'] / results['total_signals']
            
            # Accuracy by confidence bucket
            for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
                mask = [c >= threshold for c in confidences]
                if any(mask):
                    acc = np.mean([c for c, m in zip(correct, mask) if m])
                    results['accuracy_by_confidence'][f'>={threshold}'] = acc
        
        return results


# Example usage
if __name__ == "__main__":
    print("TradingPredictor requires trained model and feature engineer.")
    print("See main.py for full usage example.")
