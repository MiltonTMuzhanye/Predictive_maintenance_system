"""
Inference engine for predictive maintenance API.
"""
import pandas as pd
import numpy as np
import joblib
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime
import os

logger = logging.getLogger(__name__)

class PredictionEngine:
    """Prediction engine for equipment failure prediction."""
    
    def __init__(self, model_dir: str = "models/"):
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.threshold = 0.5
        self.model_info = {}
        self.is_ready_flag = False
    
    def load_models(self, model_name: str = "best_model.pkl", 
                   scaler_name: str = "scaler.pkl",
                   metadata_name: str = "model_metadata.json"):
        """Load trained model, scaler, and metadata."""
        try:
            # Load model
            model_path = os.path.join(self.model_dir, model_name)
            self.model = joblib.load(model_path)
            logger.info(f"Model loaded from {model_path}")
            
            # Load scaler
            scaler_path = os.path.join(self.model_dir, scaler_name)
            self.scaler = joblib.load(scaler_path)
            logger.info(f"Scaler loaded from {scaler_path}")
            
            # Load metadata if exists
            metadata_path = os.path.join(self.model_dir, metadata_name)
            if os.path.exists(metadata_path):
                import json
                with open(metadata_path, 'r') as f:
                    self.model_info = json.load(f)
                logger.info(f"Model metadata loaded from {metadata_path}")
            
            # Extract feature names if available
            if hasattr(self.model, 'feature_names_in_'):
                self.feature_names = list(self.model.feature_names_in_)
            elif self.model_info and 'feature_names' in self.model_info:
                self.feature_names = self.model_info['feature_names']
            
            self.is_ready_flag = True
            logger.info("Prediction engine ready")
            
        except Exception as e:
            logger.error(f"Failed to load models: {str(e)}")
            self.is_ready_flag = False
            raise
    
    def is_ready(self) -> bool:
        """Check if prediction engine is ready."""
        return self.is_ready_flag and self.model is not None
    
    def predict(self, features_df: pd.DataFrame, threshold: Optional[float] = None) -> Dict[str, Any]:
        """
        Predict failure probability.
        
        Args:
            features_df: DataFrame with equipment features
            threshold: Optional prediction threshold
            
        Returns:
            Dictionary with prediction results
        """
        if not self.is_ready():
            raise ValueError("Prediction engine not ready. Call load_models() first.")
        
        # Use provided threshold or default
        prediction_threshold = threshold if threshold is not None else self.threshold
        
        try:
            # Ensure correct feature order
            if self.feature_names:
                features_df = features_df[self.feature_names]
            
            # Scale features
            features_scaled = self.scaler.transform(features_df)
            
            # Get prediction probabilities
            failure_probability = self.model.predict_proba(features_scaled)[:, 1]
            
            # Apply threshold
            prediction = (failure_probability >= prediction_threshold).astype(int)
            
            # Get first prediction (assuming single equipment)
            prob = float(failure_probability[0])
            pred = int(prediction[0])
            
            return {
                'failure_probability': prob,
                'prediction': pred,
                'alert_level': self._determine_alert_level(prob),
                'recommended_action': self._get_recommended_action(prob),
                'confidence': self._calculate_confidence(prob)
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise
    
    def predict_batch(self, features_list: List[pd.DataFrame], 
                     threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """Predict failures for a batch of equipment."""
        predictions = []
        for features_df in features_list:
            prediction = self.predict(features_df, threshold)
            predictions.append(prediction)
        return predictions
    
    def _determine_alert_level(self, probability: float) -> str:
        """Determine alert level based on probability."""
        if probability >= 0.8:
            return "CRITICAL"
        elif probability >= 0.6:
            return "HIGH"
        elif probability >= 0.4:
            return "MEDIUM"
        elif probability >= 0.2:
            return "LOW"
        else:
            return "NORMAL"
    
    def _get_recommended_action(self, probability: float) -> str:
        """Get recommended maintenance action."""
        if probability >= 0.8:
            return "SCHEDULE IMMEDIATE MAINTENANCE"
        elif probability >= 0.6:
            return "SCHEDULE MAINTENANCE WITHIN 24 HOURS"
        elif probability >= 0.4:
            return "INCREASE MONITORING FREQUENCY"
        elif probability >= 0.2:
            return "CONTINUE NORMAL MONITORING"
        else:
            return "NO ACTION REQUIRED"
    
    def _calculate_confidence(self, probability: float) -> float:
        """Calculate prediction confidence."""
        # Confidence is highest when probability is near 0 or 1
        return 1 - 2 * abs(probability - 0.5)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if not self.is_ready():
            raise ValueError("Model not loaded")
        
        info = {
            'model_name': self.model_info.get('model_name', 'Random Forest'),
            'model_version': self.model_info.get('version', '1.0.0'),
            'training_date': self.model_info.get('training_date', 'Unknown'),
            'performance_metrics': self.model_info.get('performance_metrics', {}),
            'feature_count': len(self.feature_names) if self.feature_names else 0,
            'threshold': self.threshold,
            'feature_names': self.feature_names
        }
        
        return info
    
    def get_feature_importance(self) -> List[Dict[str, Any]]:
        """Get feature importance from the model."""
        if not self.is_ready():
            raise ValueError("Model not loaded")
        
        if hasattr(self.model, 'feature_importances_') and self.feature_names:
            importances = self.model.feature_importances_
            
            importance_list = []
            for feature, importance in zip(self.feature_names, importances):
                importance_list.append({
                    'feature': feature,
                    'importance': float(importance),
                    'importance_normalized': float(importance / importances.sum())
                })
            
            # Sort by importance
            importance_list.sort(key=lambda x: x['importance'], reverse=True)
            return importance_list
        else:
            return []
    
    def set_threshold(self, threshold: float):
        """Set prediction threshold."""
        if not 0 <= threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")
        
        self.threshold = threshold
        logger.info(f"Prediction threshold set to {threshold}")