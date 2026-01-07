"""
Machine Learning model training for predictive maintenance.
"""
import pandas as pd
import numpy as np
import joblib
from typing import Dict, Tuple, Any
import logging

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             classification_report)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

class MLModelTrainer:
    """Train and evaluate machine learning models."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
    
    def prepare_training_data(self, X: pd.DataFrame, y: pd.Series, 
                             test_size: float = 0.2) -> Tuple:
        """Prepare training and testing data with SMOTE."""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, 
            stratify=y
        )
        
        logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        logger.info(f"Original failure rate: {y_train.mean():.4f}")
        
        # Handle class imbalance with SMOTE
        smote = SMOTE(random_state=self.random_state)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        
        logger.info(f"After SMOTE - Class distribution: {pd.Series(y_train_resampled).value_counts().to_dict()}")
        
        return X_train_resampled, X_test, y_train_resampled, y_test
    
    def define_models(self) -> Dict[str, Any]:
        """Define ML models to train."""
        models = {
            'Logistic Regression': LogisticRegression(
                random_state=self.random_state, 
                max_iter=1000,
                class_weight='balanced'
            ),
            'Random Forest': RandomForestClassifier(
                random_state=self.random_state,
                n_estimators=100,
                class_weight='balanced_subsample',
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                random_state=self.random_state,
                n_estimators=100
            ),
            'XGBoost': XGBClassifier(
                random_state=self.random_state,
                eval_metric='logloss',
                use_label_encoder=False,
                scale_pos_weight=(len(y_train_resampled[y_train_resampled==0]) / 
                                 len(y_train_resampled[y_train_resampled==1]))
            )
        }
        return models
    
    def train_models(self, X_train: np.ndarray, X_test: np.ndarray,
                    y_train: pd.Series, y_test: pd.Series,
                    feature_names: list = None) -> Dict:
        """Train and evaluate multiple models."""
        
        models = self.define_models()
        self.results = {}
        
        for name, model in models.items():
            logger.info(f"Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            self.models[name] = model
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            self.results[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_test, y_pred_proba),
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
            }
            
            # Feature importance for tree-based models
            if hasattr(model, 'feature_importances_') and feature_names:
                importances = model.feature_importances_
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importances
                }).sort_values('importance', ascending=False)
                self.results[name]['feature_importance'] = importance_df.to_dict()
        
        # Determine best model based on F1 score
        self.best_model_name = max(self.results.items(), 
                                  key=lambda x: x[1]['f1'])[0]
        self.best_model = self.models[self.best_model_name]
        
        logger.info(f"Best model: {self.best_model_name} "
                   f"(F1: {self.results[self.best_model_name]['f1']:.4f})")
        
        return self.results
    
    def get_best_model(self):
        """Get the best performing model."""
        return self.best_model, self.best_model_name
    
    def save_model(self, model, filepath: str):
        """Save trained model to disk."""
        joblib.dump(model, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model from disk."""
        model = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")
        return model
    
    def cross_validate(self, model, X: np.ndarray, y: pd.Series, 
                      cv: int = 5) -> Dict:
        """Perform cross-validation."""
        cv_scores = cross_val_score(model, X, y, cv=cv, 
                                   scoring='f1', n_jobs=-1)
        
        return {
            'mean_f1': cv_scores.mean(),
            'std_f1': cv_scores.std(),
            'scores': cv_scores.tolist()
        }