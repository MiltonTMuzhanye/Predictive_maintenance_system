"""
XGBoost Model Implementation

Provides XGBoost model wrapper for churn prediction.
"""

import xgboost as xgb
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import json
import pickle
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)
from ..utils.logger import get_logger
from ..utils.exceptions import ModelError
from ..evaluation.metrics import ModelMetrics

logger = get_logger(__name__)

class XGBoostModel:
    """XGBoost wrapper for churn prediction."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize XGBoost model.
        
        Args:
            config: Model configuration parameters
        """
        self.config = config or {}
        self.model = None
        self.feature_names = None
        self.metrics = None
        
    def build_model(self, params: Optional[Dict] = None) -> xgb.XGBClassifier:
        """
        Build XGBoost model with parameters.
        
        Args:
            params: XGBoost parameters
            
        Returns:
            XGBClassifier instance
        """
        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.01,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'scale_pos_weight': 1,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'early_stopping_rounds': 10,
            'random_state': 42
        }
        
        if params is None:
            params = default_params
        else:
            # Merge with defaults
            params = {**default_params, **params}
            
        self.model = xgb.XGBClassifier(**params)
        logger.info(f"Built XGBoost model with params: {params}")
        return self.model
        
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """
        Train XGBoost model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Dictionary with training metrics
        """
        logger.info("Starting XGBoost training...")
        
        if self.model is None:
            self.build_model()
            
        # Store feature names
        self.feature_names = X_train.columns.tolist()
        
        # Prepare evaluation set if provided
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]
            
        # Train model
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False
        )
        
        # Calculate training metrics
        y_pred = self.model.predict(X_train)
        y_proba = self.model.predict_proba(X_train)[:, 1]
        
        metrics = {
            'train_accuracy': accuracy_score(y_train, y_pred),
            'train_precision': precision_score(y_train, y_pred),
            'train_recall': recall_score(y_train, y_pred),
            'train_f1': f1_score(y_train, y_pred),
            'train_auc': roc_auc_score(y_train, y_proba)
        }
        
        # Calculate validation metrics if available
        if X_val is not None and y_val is not None:
            y_val_pred = self.model.predict(X_val)
            y_val_proba = self.model.predict_proba(X_val)[:, 1]
            
            metrics.update({
                'val_accuracy': accuracy_score(y_val, y_val_pred),
                'val_precision': precision_score(y_val, y_val_pred),
                'val_recall': recall_score(y_val, y_val_pred),
                'val_f1': f1_score(y_val, y_val_pred),
                'val_auc': roc_auc_score(y_val, y_val_proba)
            })
            
        self.metrics = metrics
        logger.info(f"Training complete. Metrics: {metrics}")
        return metrics
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Features for prediction
            
        Returns:
            Array of predictions
        """
        if self.model is None:
            raise ModelError("Model not trained or loaded")
            
        try:
            # Ensure features match training data
            if self.feature_names is not None:
                missing = set(self.feature_names) - set(X.columns)
                if missing:
                    logger.warning(f"Missing features: {missing}")
                    for col in missing:
                        X[col] = 0
                        
            predictions = self.model.predict(X)
            logger.info(f"Made {len(predictions)} predictions")
            return predictions
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise ModelError(f"Prediction failed: {str(e)}")
            
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get probability predictions.
        
        Args:
            X: Features for prediction
            
        Returns:
            Array of probabilities
        """
        if self.model is None:
            raise ModelError("Model not trained or loaded")
            
        try:
            if self.feature_names is not None:
                missing = set(self.feature_names) - set(X.columns)
                if missing:
                    for col in missing:
                        X[col] = 0
                        
            probabilities = self.model.predict_proba(X)[:, 1]
            return probabilities
            
        except Exception as e:
            logger.error(f"Probability prediction failed: {str(e)}")
            raise ModelError(f"Probability prediction failed: {str(e)}")
            
    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Evaluating model...")
        
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred)
        }
        
        self.metrics = metrics
        logger.info(f"Evaluation complete: {metrics}")
        return metrics
        
    def get_feature_importance(self, importance_type: str = 'gain') -> pd.DataFrame:
        """
        Get feature importance scores.
        
        Args:
            importance_type: Type of importance ('gain', 'weight', 'cover')
            
        Returns:
            DataFrame with feature importance
        """
        if self.model is None:
            raise ModelError("Model not trained or loaded")
            
        importance = self.model.get_booster().get_score(importance_type=importance_type)
        
        # Convert to DataFrame
        if self.feature_names:
            # Match feature names from model to actual names
            importance_df = pd.DataFrame({
                'Feature': self.feature_names[:len(importance)],
                'Importance': list(importance.values())
            }).sort_values('Importance', ascending=False)
        else:
            importance_df = pd.DataFrame({
                'Feature': list(importance.keys()),
                'Importance': list(importance.values())
            }).sort_values('Importance', ascending=False)
            
        return importance_df
        
    def save_model(self, path: str):
        """
        Save model to disk.
        
        Args:
            path: Path to save model
        """
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save model
            self.model.save_model(f"{path}.json")
            
            # Save metadata
            metadata = {
                'feature_names': self.feature_names,
                'config': self.config,
                'metrics': self.metrics
            }
            
            with open(f"{path}_metadata.json", 'w') as f:
                json.dump(metadata, f)
                
            logger.info(f"Model saved to {path}")
            
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            raise ModelError(f"Failed to save model: {str(e)}")
            
    def load_model(self, path: str):
        """
        Load model from disk.
        
        Args:
            path: Path to load model from
        """
        try:
            # Load model
            self.model = xgb.XGBClassifier()
            self.model.load_model(f"{path}.json")
            
            # Load metadata
            with open(f"{path}_metadata.json", 'r') as f:
                metadata = json.load(f)
                
            self.feature_names = metadata.get('feature_names')
            self.config = metadata.get('config', {})
            self.metrics = metadata.get('metrics', {})
            
            logger.info(f"Model loaded from {path}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise ModelError(f"Failed to load model: {str(e)}")