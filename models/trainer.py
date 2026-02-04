"""
Model Trainer Module
Handles training and evaluation of trading models.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging
import joblib

from sklearn.model_selection import (
    train_test_split, 
    TimeSeriesSplit, 
    cross_val_score,
    GridSearchCV
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Handles training and evaluation of trading classification models.
    
    Supports:
    - Logistic Regression
    - Random Forest
    - XGBoost
    - Gradient Boosting
    
    Features:
    - Time-series cross-validation
    - Hyperparameter tuning
    - Model persistence
    - Comprehensive evaluation metrics
    """
    
    SUPPORTED_MODELS = ['logistic', 'random_forest', 'xgboost', 'gradient_boosting']
    
    def __init__(
        self,
        model_type: str = 'xgboost',
        random_state: int = 42,
        n_jobs: int = -1,
        model_params: Optional[Dict] = None
    ):
        """
        Initialize model trainer.
        
        Args:
            model_type: Type of model to train
            random_state: Random seed for reproducibility
            n_jobs: Number of parallel jobs
            model_params: Custom model parameters
        """
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(f"Model type must be one of {self.SUPPORTED_MODELS}")
        
        self.model_type = model_type
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.model_params = model_params or {}
        
        self.model = None
        self.label_encoder = LabelEncoder()
        self.classes_ = None
        self.feature_names = None
        self.training_metrics = {}
    
    def _create_model(self) -> Any:
        """Create model instance based on model_type."""
        if self.model_type == 'logistic':
            default_params = {
                'max_iter': 1000,
                'random_state': self.random_state,
                'class_weight': 'balanced',
                'n_jobs': self.n_jobs
            }
            default_params.update(self.model_params)
            return LogisticRegression(**default_params)
        
        elif self.model_type == 'random_forest':
            default_params = {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': self.random_state,
                'class_weight': 'balanced',
                'n_jobs': self.n_jobs
            }
            default_params.update(self.model_params)
            return RandomForestClassifier(**default_params)
        
        elif self.model_type == 'xgboost':
            try:
                from xgboost import XGBClassifier
            except ImportError:
                raise ImportError("xgboost not installed. Run: pip install xgboost")
            
            default_params = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': self.random_state,
                'n_jobs': self.n_jobs,
                'use_label_encoder': False,
                'eval_metric': 'mlogloss'
            }
            default_params.update(self.model_params)
            return XGBClassifier(**default_params)
        
        elif self.model_type == 'gradient_boosting':
            default_params = {
                'n_estimators': 100,
                'max_depth': 5,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'random_state': self.random_state
            }
            default_params.update(self.model_params)
            return GradientBoostingClassifier(**default_params)
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        test_size: float = 0.2,
        val_size: float = 0.1
    ) -> Dict[str, Any]:
        """
        Train the model with train/val/test split.
        
        Args:
            X: Feature matrix
            y: Labels
            feature_names: Names of features
            test_size: Proportion for test set
            val_size: Proportion for validation set
        
        Returns:
            Dictionary with training metrics
        """
        logger.info(f"Training {self.model_type} model...")
        self.feature_names = feature_names
        
        # Filter out NaN labels
        valid_mask = ~np.isnan(y)
        X = X[valid_mask]
        y = y[valid_mask]
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        self.classes_ = self.label_encoder.classes_
        
        logger.info(f"Dataset size: {len(X)}")
        logger.info(f"Classes: {self.classes_}")
        logger.info(f"Class distribution: {np.bincount(y_encoded)}")
        
        # Split data (preserving time order)
        train_end = int(len(X) * (1 - test_size - val_size))
        val_end = int(len(X) * (1 - test_size))
        
        X_train, y_train = X[:train_end], y_encoded[:train_end]
        X_val, y_val = X[train_end:val_end], y_encoded[train_end:val_end]
        X_test, y_test = X[val_end:], y_encoded[val_end:]
        
        logger.info(f"Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}")
        
        # Create and train model
        self.model = self._create_model()
        self.model.fit(X_train, y_train)
        
        # Evaluate on all sets
        train_metrics = self._evaluate(X_train, y_train, "Train")
        val_metrics = self._evaluate(X_val, y_val, "Validation")
        test_metrics = self._evaluate(X_test, y_test, "Test")
        
        self.training_metrics = {
            'train': train_metrics,
            'validation': val_metrics,
            'test': test_metrics
        }
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.training_metrics['feature_importance'] = dict(zip(
                feature_names or [f'f_{i}' for i in range(X.shape[1])],
                self.model.feature_importances_
            ))
        
        return self.training_metrics
    
    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_splits: int = 5,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Perform time-series cross-validation.
        
        Args:
            X: Feature matrix
            y: Labels
            n_splits: Number of CV splits
            feature_names: Names of features
        
        Returns:
            Dictionary with CV metrics
        """
        logger.info(f"Running {n_splits}-fold time series cross-validation...")
        self.feature_names = feature_names
        
        # Filter out NaN labels
        valid_mask = ~np.isnan(y)
        X = X[valid_mask]
        y = y[valid_mask]
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        self.classes_ = self.label_encoder.classes_
        
        # Time series CV
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        cv_results = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': []
        }
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
            
            model = self._create_model()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            cv_results['accuracy'].append(accuracy_score(y_test, y_pred))
            cv_results['precision'].append(precision_score(y_test, y_pred, average='weighted', zero_division=0))
            cv_results['recall'].append(recall_score(y_test, y_pred, average='weighted', zero_division=0))
            cv_results['f1'].append(f1_score(y_test, y_pred, average='weighted', zero_division=0))
            
            logger.info(f"Fold {fold + 1}: Accuracy={cv_results['accuracy'][-1]:.4f}, "
                       f"F1={cv_results['f1'][-1]:.4f}")
        
        # Compute mean and std
        cv_summary = {}
        for metric, values in cv_results.items():
            cv_summary[f'{metric}_mean'] = np.mean(values)
            cv_summary[f'{metric}_std'] = np.std(values)
        
        logger.info(f"CV Results: Accuracy={cv_summary['accuracy_mean']:.4f}±{cv_summary['accuracy_std']:.4f}")
        
        # Train final model on all data
        self.model = self._create_model()
        self.model.fit(X, y_encoded)
        
        return cv_summary
    
    def _evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        set_name: str
    ) -> Dict[str, float]:
        """Evaluate model on a dataset."""
        y_pred = self.model.predict(X)
        y_prob = self.model.predict_proba(X) if hasattr(self.model, 'predict_proba') else None
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y, y_pred, average='weighted', zero_division=0)
        }
        
        # ROC-AUC (for multi-class)
        if y_prob is not None and len(self.classes_) > 2:
            try:
                metrics['roc_auc'] = roc_auc_score(y, y_prob, multi_class='ovr', average='weighted')
            except ValueError:
                pass
        
        logger.info(f"{set_name} - Accuracy: {metrics['accuracy']:.4f}, "
                   f"Precision: {metrics['precision']:.4f}, "
                   f"Recall: {metrics['recall']:.4f}, "
                   f"F1: {metrics['f1']:.4f}")
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        y_pred_encoded = self.model.predict(X)
        return self.label_encoder.inverse_transform(y_pred_encoded)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance as DataFrame."""
        if not hasattr(self.model, 'feature_importances_'):
            logger.warning("Model does not have feature_importances_")
            return pd.DataFrame()
        
        importances = self.model.feature_importances_
        
        # Handle length mismatch
        feature_names = self.feature_names
        if feature_names is None or len(feature_names) != len(importances):
            logger.warning(f"Feature names length ({len(feature_names) if feature_names else 0}) matches importances ({len(importances)}). Using generic names.")
            feature_names = [f'f_{i}' for i in range(len(importances))]
            
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def tune_hyperparameters(
        self,
        X: np.ndarray,
        y: np.ndarray,
        param_grid: Optional[Dict] = None,
        cv: int = 3
    ) -> Dict[str, Any]:
        """
        Tune hyperparameters using GridSearchCV.
        
        Args:
            X: Feature matrix
            y: Labels
            param_grid: Parameters to search
            cv: Number of CV folds
        
        Returns:
            Best parameters and score
        """
        logger.info("Starting hyperparameter tuning...")
        
        # Filter and encode
        valid_mask = ~np.isnan(y)
        X = X[valid_mask]
        y = y[valid_mask]
        y_encoded = self.label_encoder.fit_transform(y)
        self.classes_ = self.label_encoder.classes_
        
        # Default param grids
        if param_grid is None:
            if self.model_type == 'random_forest':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15],
                    'min_samples_split': [2, 5, 10]
                }
            elif self.model_type == 'xgboost':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.05, 0.1, 0.2]
                }
            else:
                param_grid = {}
        
        base_model = self._create_model()
        tscv = TimeSeriesSplit(n_splits=cv)
        
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=tscv,
            scoring='f1_weighted',
            n_jobs=self.n_jobs,
            verbose=1
        )
        
        grid_search.fit(X, y_encoded)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best score: {grid_search.best_score_:.4f}")
        
        # Update model with best params
        self.model = grid_search.best_estimator_
        self.model_params.update(grid_search.best_params_)
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
    
    def save(self, path: str):
        """Save model to disk."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'label_encoder': self.label_encoder,
            'classes_': self.classes_,
            'feature_names': self.feature_names,
            'training_metrics': self.training_metrics,
            'model_params': self.model_params
        }
        
        joblib.dump(model_data, save_path)
        logger.info(f"Model saved to {save_path}")
    
    def load(self, path: str):
        """Load model from disk."""
        load_path = Path(path)
        if not load_path.exists():
            raise FileNotFoundError(f"Model file not found: {load_path}")
        
        model_data = joblib.load(load_path)
        
        self.model = model_data['model']
        self.model_type = model_data['model_type']
        self.label_encoder = model_data['label_encoder']
        self.classes_ = model_data['classes_']
        self.feature_names = model_data['feature_names']
        self.training_metrics = model_data['training_metrics']
        self.model_params = model_data['model_params']
        
        logger.info(f"Model loaded from {load_path}")


# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.choice([-1, 0, 1], size=n_samples, p=[0.2, 0.6, 0.2])
    
    # Train model
    trainer = ModelTrainer(model_type='random_forest')
    metrics = trainer.train(X, y, feature_names=[f'feature_{i}' for i in range(n_features)])
    
    print("\nTraining Metrics:")
    print(f"Test Accuracy: {metrics['test']['accuracy']:.4f}")
    print(f"Test F1: {metrics['test']['f1']:.4f}")
    
    # Feature importance
    importance = trainer.get_feature_importance()
    print("\nTop 5 Features:")
    print(importance.head())
