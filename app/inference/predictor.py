"""
Prediction Engine

Handles real-time and batch predictions.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import json
import joblib
from datetime import datetime
import logging
from ..utils.logger import get_logger

logger = get_logger(__name__)

class Predictor:
    """Main predictor for maintenance system."""
    
    def __init__(self, model_path: str, config_path: str = "configs/config.yaml"):
        """
        Initialize predictor with trained model.
        
        Args:
            model_path: Path to trained model
            config_path: Path to configuration
        """
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.thresholds = None
        
        self.load_model()
        self.load_scaler()
        self.load_features()
        self.load_thresholds()
        
    def load_model(self):
        """Load trained model."""
        try:
            # Load model based on file extension
            if self.model_path.endswith('.json'):
                import xgboost as xgb
                self.model = xgb.XGBClassifier()
                self.model.load_model(self.model_path)
            elif self.model_path.endswith('.pkl'):
                self.model = joblib.load(self.model_path)
            else:
                # Try to load with default
                import xgboost as xgb
                self.model = xgb.XGBClassifier()
                self.model.load_model(self.model_path)
                
            logger.info(f"Model loaded from {self.model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
            
    def load_scaler(self):
        """Load scaler for feature normalization."""
        try:
            scaler_path = Path(self.model_path).parent.parent / "scalers" / "scaler.pkl"
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
                logger.info("Scaler loaded successfully")
            else:
                logger.warning("Scaler not found, using raw features")
        except Exception as e:
            logger.warning(f"Failed to load scaler: {str(e)}")
            
    def load_features(self):
        """Load feature names."""
        try:
            feature_path = Path(self.model_path).parent.parent / "feature_lists" / "feature_names.json"
            if feature_path.exists():
                with open(feature_path, 'r') as f:
                    self.feature_names = json.load(f)
                logger.info(f"Loaded {len(self.feature_names)} feature names")
            else:
                logger.warning("Feature names not found")
        except Exception as e:
            logger.warning(f"Failed to load feature names: {str(e)}")
            
    def load_thresholds(self):
        """Load prediction thresholds."""
        try:
            threshold_path = Path(self.model_path).parent.parent / "thresholds" / "thresholds.json"
            if threshold_path.exists():
                with open(threshold_path, 'r') as f:
                    self.thresholds = json.load(f)
                logger.info("Thresholds loaded successfully")
            else:
                self.thresholds = {
                    'churn_high': 0.7,
                    'churn_medium': 0.4,
                    'churn_low': 0.2,
                    'health_excellent': 0.8,
                    'health_good': 0.6,
                    'health_fair': 0.4,
                    'health_poor': 0.2
                }
                logger.info("Using default thresholds")
        except Exception as e:
            logger.warning(f"Failed to load thresholds: {str(e)}")
            self.thresholds = {}
            
    def preprocess_features(
        self,
        data: Union[pd.DataFrame, Dict]
    ) -> pd.DataFrame:
        """
        Preprocess input data for prediction.
        
        Args:
            data: Input data (DataFrame or dict)
            
        Returns:
            Preprocessed DataFrame
        """
        # Convert dict to DataFrame if needed
        if isinstance(data, dict):
            data = pd.DataFrame([data])
            
        # Ensure all required features exist
        if self.feature_names:
            for feature in self.feature_names:
                if feature not in data.columns:
                    data[feature] = 0
                    
            # Reorder columns to match training
            data = data[self.feature_names]
            
        # Scale features
        if self.scaler is not None:
            try:
                scaled_data = self.scaler.transform(data)
                data = pd.DataFrame(scaled_data, columns=data.columns)
            except Exception as e:
                logger.warning(f"Scaling failed: {str(e)}")
                
        return data
        
    def predict(self, data: Union[pd.DataFrame, Dict]) -> Dict[str, Any]:
        """
        Make prediction on single instance.
        
        Args:
            data: Input data
            
        Returns:
            Prediction results
        """
        try:
            # Preprocess data
            processed_data = self.preprocess_features(data)
            
            # Make prediction
            if hasattr(self.model, 'predict_proba'):
                probability = self.model.predict_proba(processed_data)[0][1]
                prediction = 1 if probability >= 0.5 else 0
            else:
                prediction = self.model.predict(processed_data)[0]
                probability = float(prediction)
                
            # Get churn risk level
            risk_level = self._get_risk_level(probability)
            
            # Calculate health score
            health_score = 1 - probability
            
            # Get health status
            health_status = self._get_health_status(health_score)
            
            return {
                'prediction': int(prediction),
                'probability': float(probability),
                'risk_level': risk_level,
                'health_score': float(health_score),
                'health_status': health_status,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            
    def predict_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions on batch of data.
        
        Args:
            data: DataFrame with multiple instances
            
        Returns:
            DataFrame with predictions
        """
        try:
            # Preprocess data
            processed_data = self.preprocess_features(data)
            
            # Make predictions
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(processed_data)[:, 1]
                predictions = (probabilities >= 0.5).astype(int)
            else:
                predictions = self.model.predict(processed_data)
                probabilities = predictions.astype(float)
                
            # Create results DataFrame
            results = data.copy()
            results['prediction'] = predictions
            results['probability'] = probabilities
            results['risk_level'] = probabilities.apply(self._get_risk_level)
            results['health_score'] = 1 - probabilities
            results['health_status'] = results['health_score'].apply(self._get_health_status)
            
            return results
            
        except Exception as e:
            logger.error(f"Batch prediction failed: {str(e)}")
            raise
            
    def _get_risk_level(self, probability: float) -> str:
        """Get risk level based on probability."""
        if probability >= self.thresholds.get('churn_high', 0.7):
            return 'high'
        elif probability >= self.thresholds.get('churn_medium', 0.4):
            return 'medium'
        else:
            return 'low'
            
    def _get_health_status(self, health_score: float) -> str:
        """Get health status based on score."""
        if health_score >= self.thresholds.get('health_excellent', 0.8):
            return 'excellent'
        elif health_score >= self.thresholds.get('health_good', 0.6):
            return 'good'
        elif health_score >= self.thresholds.get('health_fair', 0.4):
            return 'fair'
        else:
            return 'poor'