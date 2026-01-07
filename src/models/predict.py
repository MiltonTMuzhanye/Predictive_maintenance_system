"""
Prediction module for real-time failure prediction.
"""
import numpy as np
import pandas as pd
import joblib
from typing import Dict, List, Any, Tuple
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class FailurePredictor:
    """Predict equipment failures in real-time."""
    
    def __init__(self, model_path: str = None, scaler_path: str = None):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.threshold = 0.5
        
        if model_path:
            self.load_model(model_path)
        if scaler_path:
            self.load_scaler(scaler_path)
    
    def load_model(self, model_path: str):
        """Load trained model from disk."""
        self.model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")
    
    def load_scaler(self, scaler_path: str):
        """Load fitted scaler from disk."""
        self.scaler = joblib.load(scaler_path)
        logger.info(f"Scaler loaded from {scaler_path}")
    
    def set_threshold(self, threshold: float):
        """Set prediction threshold."""
        self.threshold = threshold
        logger.info(f"Prediction threshold set to {threshold}")
    
    def predict(self, features: pd.DataFrame) -> Dict[str, Any]:
        """
        Predict failure probability for given features.
        
        Args:
            features: DataFrame with required features
            
        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Scale features if scaler is available
        if self.scaler is not None:
            features_scaled = self.scaler.transform(features)
        else:
            features_scaled = features.values
        
        # Get prediction probabilities
        failure_probability = self.model.predict_proba(features_scaled)[:, 1]
        
        # Apply threshold
        prediction = (failure_probability >= self.threshold).astype(int)
        
        # Prepare results
        results = []
        for i, prob in enumerate(failure_probability):
            result = {
                'failure_probability': float(prob),
                'prediction': int(prediction[i]),
                'alert_level': self._determine_alert_level(prob),
                'recommended_action': self._get_recommended_action(prob),
                'confidence': self._calculate_confidence(prob)
            }
            results.append(result)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'predictions': results,
            'summary': self._create_summary(prediction, failure_probability)
        }
    
    def predict_batch(self, features_list: List[pd.DataFrame]) -> List[Dict]:
        """Predict failures for a batch of equipment."""
        predictions = []
        for features in features_list:
            prediction = self.predict(features)
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
    
    def _create_summary(self, predictions: np.ndarray, 
                       probabilities: np.ndarray) -> Dict:
        """Create prediction summary."""
        total_predictions = len(predictions)
        failures_predicted = int(predictions.sum())
        
        return {
            'total_equipment': total_predictions,
            'predicted_failures': failures_predicted,
            'failure_rate_predicted': float(failures_predicted / total_predictions),
            'average_risk_score': float(probabilities.mean()),
            'high_risk_count': int((probabilities >= 0.7).sum()),
            'critical_risk_count': int((probabilities >= 0.9).sum())
        }
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from the model."""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        if hasattr(self.model, 'feature_importances_') and self.feature_names:
            importances = self.model.feature_importances_
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            return importance_df
        else:
            logger.warning("Model doesn't support feature importance or feature names not set")
            return pd.DataFrame()