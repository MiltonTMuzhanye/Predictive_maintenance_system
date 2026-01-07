"""
Feature engineering for predictive maintenance.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Engineer features for predictive maintenance."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = []
    
    def create_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Create engineered features from raw data.
        
        Args:
            df: Cleaned equipment data
            
        Returns:
            Tuple of (features, target)
        """
        data = df.copy()
        
        # Encode categorical variables
        if 'Type' in data.columns:
            data['Type_encoded'] = self.label_encoder.fit_transform(data['Type'])
        
        # Create physics-based features
        data = self._create_physics_features(data)
        
        # Create statistical features
        data = self._create_statistical_features(data)
        
        # Prepare features and target
        features = self._select_features(data)
        target = data['Machine failure']
        
        self.feature_names = features.columns.tolist()
        logger.info(f"Created {len(self.feature_names)} features")
        
        return features, target
    
    def _create_physics_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create physics-based features."""
        data = df.copy()
        
        # Power calculation
        if 'Rotational speed [rpm]' in data.columns and 'Torque [Nm]' in data.columns:
            data['Power'] = data['Rotational speed [rpm]'] * data['Torque [Nm]']
        
        # Temperature ratio
        if 'Process temperature [K]' in data.columns and 'Air temperature [K]' in data.columns:
            data['Temperature_Ratio'] = data['Process temperature [K]'] / data['Air temperature [K]']
        
        # Tool wear rate (adjusted for temperature)
        if 'Tool wear [min]' in data.columns and 'Air temperature [K]' in data.columns:
            data['Tool_Wear_Rate'] = data['Tool wear [min]'] / (data['Air temperature [K]'] - 273.15)
            data['Tool_Wear_Rate'] = data['Tool_Wear_Rate'].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Stress factor
        if 'Torque [Nm]' in data.columns and 'Rotational speed [rpm]' in data.columns:
            data['Stress_Factor'] = data['Torque [Nm]'] * np.sqrt(data['Rotational speed [rpm]'])
        
        return data
    
    def _create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create statistical features (for sequence data)."""
        data = df.copy()
        
        # These would be expanded with rolling windows in production
        # For now, create basic statistics
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        failure_cols = ['Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
        numerical_cols = [col for col in numerical_cols if col not in failure_cols]
        
        return data
    
    def _select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select final feature set."""
        base_features = [
            'Air temperature [K]', 
            'Process temperature [K]',
            'Rotational speed [rpm]', 
            'Torque [Nm]', 
            'Tool wear [min]'
        ]
        
        engineered_features = [
            'Power',
            'Temperature_Ratio',
            'Tool_Wear_Rate',
            'Stress_Factor',
            'Type_encoded'
        ]
        
        # Only include features that exist in the dataframe
        selected_features = []
        for feature in base_features + engineered_features:
            if feature in df.columns:
                selected_features.append(feature)
        
        return df[selected_features]
    
    def scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame = None) -> Tuple:
        """Scale features using StandardScaler."""
        if X_test is not None:
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            return X_train_scaled, X_test_scaled
        else:
            return self.scaler.fit_transform(X_train)