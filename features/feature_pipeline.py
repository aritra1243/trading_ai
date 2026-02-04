"""
Feature Pipeline Module
Combines all feature engineering into a unified pipeline.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import logging
import joblib
from pathlib import Path

from .technical_features import TechnicalFeatures
from .market_features import MarketStructureFeatures
from .time_features import TimeFeatures
from .labeler import TradeLabelGenerator

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Unified feature engineering pipeline.
    
    Combines:
    - Technical indicators
    - Market structure features
    - Time features
    - Label generation
    - Feature scaling
    - Feature selection
    """
    
    def __init__(
        self,
        # Feature modules
        technical_config: Optional[dict] = None,
        market_config: Optional[dict] = None,
        time_config: Optional[dict] = None,
        label_config: Optional[dict] = None,
        # Scaling
        scaler_type: str = 'robust',  # 'standard' or 'robust'
        # Feature selection
        select_k_best: Optional[int] = None,
        feature_selection_method: str = 'f_classif'
    ):
        """Initialize feature engineering pipeline."""
        # Initialize feature generators
        self.technical = TechnicalFeatures(**(technical_config or {}))
        self.market = MarketStructureFeatures(**(market_config or {}))
        self.time = TimeFeatures(**(time_config or {}))
        self.labeler = TradeLabelGenerator(**(label_config or {}))
        
        # Scaling setup
        self.scaler_type = scaler_type
        self.scaler = None
        
        # Feature selection setup
        self.select_k_best = select_k_best
        self.feature_selection_method = feature_selection_method
        self.feature_selector = None
        
        # Track all feature columns before selection
        self.feature_columns = []
        self.selected_feature_columns = []
        self.exclude_columns = [
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'signal', 'target_price', 'stop_loss', 'bars_to_target',
            'potential_profit', 'potential_loss', 'triple_barrier_label',
            'forward_return_1', 'forward_return_5', 'forward_return_10', 'forward_return_20'
        ]
    
    def fit_transform(
        self,
        df: pd.DataFrame,
        generate_labels: bool = True
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Generate all features, labels, and fit scalers.
        
        Args:
            df: DataFrame with OHLCV data
            generate_labels: Whether to generate trading labels
        
        Returns:
            Tuple of (full_df, X, y) where X is scaled features and y is labels
        """
        logger.info("Starting feature engineering pipeline...")
        
        # Generate all features
        df = self._generate_all_features(df, generate_labels)
        
        # Identify feature columns
        self.feature_columns = [
            col for col in df.columns 
            if col not in self.exclude_columns and df[col].dtype in ['float64', 'int64', 'int32']
        ]
        
        logger.info(f"Generated {len(self.feature_columns)} features")
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Extract features and labels
        X = df[self.feature_columns].values
        y = df['signal'].values if 'signal' in df.columns else None
        
        # Fit and transform scaler
        if self.scaler_type == 'robust':
            self.scaler = RobustScaler()
        else:
            self.scaler = StandardScaler()
        
        X_scaled = self.scaler.fit_transform(X)
        
        # Feature selection (if enabled)
        if self.select_k_best and y is not None:
            if self.feature_selection_method == 'mutual_info':
                score_func = mutual_info_classif
            else:
                score_func = f_classif
            
            k = min(self.select_k_best, X_scaled.shape[1])
            self.feature_selector = SelectKBest(score_func=score_func, k=k)
            X_scaled = self.feature_selector.fit_transform(X_scaled, y)
            
            # Update selected feature columns
            mask = self.feature_selector.get_support()
            self.selected_feature_columns = [
                col for col, selected in zip(self.feature_columns, mask) if selected
            ]
            logger.info(f"Selected top {k} features")
        else:
            self.selected_feature_columns = self.feature_columns
        
        return df, X_scaled, y
    
    def transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Transform new data using fitted scalers.
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            Tuple of (full_df, X_scaled)
        """
        if self.scaler is None:
            raise ValueError("Pipeline not fitted. Call fit_transform first.")
        
        # Generate features (no labels for new data)
        df = self._generate_all_features(df, generate_labels=False)
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Extract and scale features using ALL original columns
        # Check if all required columns exist
        missing_cols = [col for col in self.feature_columns if col not in df.columns]
        if missing_cols:
            logger.warning(f"Missing {len(missing_cols)} feature columns. Filling with 0.")
            for col in missing_cols:
                df[col] = 0
                
        X = df[self.feature_columns].values
        X_scaled = self.scaler.transform(X)
        
        # Apply feature selection if fitted
        if self.feature_selector is not None:
            X_scaled = self.feature_selector.transform(X_scaled)
        
        return df, X_scaled
    
    def _generate_all_features(
        self,
        df: pd.DataFrame,
        generate_labels: bool = True
    ) -> pd.DataFrame:
        """Generate all feature types."""
        df = df.copy()
        
        # Technical features
        logger.info("Adding technical features...")
        df = self.technical.add_all_features(df)
        
        # Market structure features
        logger.info("Adding market structure features...")
        df = self.market.add_all_features(df)
        
        # Time features
        logger.info("Adding time features...")
        df = self.time.add_all_features(df)
        
        # Labels
        if generate_labels:
            logger.info("Generating labels...")
            df = self.labeler.generate_labels(df)
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in features."""
        df = df.copy()
        
        # Count missing values
        missing_counts = df[self.feature_columns].isnull().sum()
        total_missing = missing_counts.sum()
        
        if total_missing > 0:
            logger.info(f"Handling {total_missing} missing values across {(missing_counts > 0).sum()} columns")
            
            # Forward fill first, then backward fill, then fill with 0
            for col in self.feature_columns:
                df[col] = df[col].ffill().bfill().fillna(0)
        
        return df
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores from selector."""
        if self.feature_selector is None:
            logger.warning("Feature selector not fitted")
            return pd.DataFrame()
        
        scores = self.feature_selector.scores_
        pvalues = self.feature_selector.pvalues_ if hasattr(self.feature_selector, 'pvalues_') else [None] * len(scores)
        
        # Get feature columns after selection
        features = self.selected_feature_columns if hasattr(self, 'selected_feature_columns') else self.feature_columns
        
        
        # Ensure lengths match
        n_features = len(features)
        
        
        # DEBUG LOOGING
        logger.info(f"Debug Feature Importance: n_features={n_features}, n_scores={len(scores)}, n_pvalues={len(pvalues) if pvalues is not None else 0}")
        
        # Strict length enforcement
        final_scores = scores[:n_features] if len(scores) >= n_features else np.pad(scores, (0, n_features - len(scores)), constant_values=0)
        
        # Handle p-values safely
        if pvalues is not None and len(pvalues) > 0:
             final_pvalues = pvalues[:n_features] if len(pvalues) >= n_features else list(pvalues) + [1.0] * (n_features - len(pvalues))
        else:
             final_pvalues = [None] * n_features
        
        # Ensure features list matches length
        final_features = features[:n_features]
        
        importance_df = pd.DataFrame({
            'feature': final_features,
            'score': final_scores,
            'p_value': final_pvalues
        }).sort_values('score', ascending=False)
        
        return importance_df
    
    def save(self, path: str):
        """Save fitted pipeline to disk."""
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save scaler
        if self.scaler:
            joblib.dump(self.scaler, save_path / 'scaler.joblib')
        
        # Save feature selector
        if self.feature_selector:
            joblib.dump(self.feature_selector, save_path / 'feature_selector.joblib')
        
        # Save feature columns (all)
        joblib.dump(self.feature_columns, save_path / 'feature_columns.joblib')
        
        # Save selected feature columns
        if self.selected_feature_columns:
            joblib.dump(self.selected_feature_columns, save_path / 'selected_feature_columns.joblib')
        
        logger.info(f"Pipeline saved to {save_path}")
    
    def load(self, path: str):
        """Load fitted pipeline from disk."""
        load_path = Path(path)
        
        # Load scaler
        scaler_path = load_path / 'scaler.joblib'
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
        
        # Load feature selector
        selector_path = load_path / 'feature_selector.joblib'
        if selector_path.exists():
            self.feature_selector = joblib.load(selector_path)
        
        # Load feature columns (all)
        cols_path = load_path / 'feature_columns.joblib'
        if cols_path.exists():
            self.feature_columns = joblib.load(cols_path)
            
        # Load selected feature columns
        sel_cols_path = load_path / 'selected_feature_columns.joblib'
        if sel_cols_path.exists():
            self.selected_feature_columns = joblib.load(sel_cols_path)
        else:
            self.selected_feature_columns = self.feature_columns
    
    def get_feature_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get summary statistics for features."""
        summary = df[self.feature_columns].describe().T
        summary['missing'] = df[self.feature_columns].isnull().sum()
        summary['dtype'] = df[self.feature_columns].dtypes
        return summary


# Example usage
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=500, freq='D')
    np.random.seed(42)
    
    price = 100 + np.random.randn(500).cumsum() * 0.5
    df = pd.DataFrame({
        'timestamp': dates,
        'open': price + np.random.randn(500) * 0.2,
        'high': price + np.abs(np.random.randn(500) * 0.5),
        'low': price - np.abs(np.random.randn(500) * 0.5),
        'close': price,
        'volume': np.random.randint(1000000, 5000000, 500)
    })
    
    # Initialize and run pipeline
    engineer = FeatureEngineer(
        label_config={'profit_threshold': 0.01, 'loss_threshold': 0.005},
        select_k_best=30
    )
    
    full_df, X, y = engineer.fit_transform(df)
    
    print(f"\nDataset shape: {full_df.shape}")
    print(f"Feature matrix shape: {X.shape}")
    print(f"Label distribution: {np.bincount(y[y != 0].astype(int) + 1)}")
    print(f"\nTop features:")
    print(engineer.feature_columns[:10])
