"""
Data cleaning and preprocessing for predictive maintenance.
"""
import pandas as pd
import numpy as np
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)

class DataCleaner:
    """Clean and preprocess equipment sensor data."""
    
    def __init__(self):
        self.features_to_drop = ['UDI', 'Product ID']
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess the data.
        
        Args:
            df: Raw equipment data
            
        Returns:
            Cleaned dataframe
        """
        data = df.copy()
        
        # Remove non-predictive columns
        data = data.drop(columns=self.features_to_drop, errors='ignore')
        
        # Validate data types
        self._validate_data_types(data)
        
        # Check for outliers
        self._detect_outliers(data)
        
        logger.info(f"Preprocessed data shape: {data.shape}")
        return data
    
    def _validate_data_types(self, df: pd.DataFrame):
        """Validate and convert data types."""
        expected_types = {
            'Type': 'object',
            'Air temperature [K]': 'float64',
            'Process temperature [K]': 'float64',
            'Rotational speed [rpm]': 'int64',
            'Torque [Nm]': 'float64',
            'Tool wear [min]': 'int64',
            'Machine failure': 'int64',
            'TWF': 'int64',
            'HDF': 'int64',
            'PWF': 'int64',
            'OSF': 'int64',
            'RNF': 'int64'
        }
        
        for col, expected_type in expected_types.items():
            if col in df.columns:
                actual_type = str(df[col].dtype)
                if expected_type not in actual_type:
                    logger.warning(f"Column {col} has type {actual_type}, expected {expected_type}")
    
    def _detect_outliers(self, df: pd.DataFrame, threshold: float = 3.0):
        """Detect outliers in numerical columns."""
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        numerical_cols = [col for col in numerical_cols if col not in 
                         ['Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']]
        
        for col in numerical_cols:
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            outliers = (z_scores > threshold).sum()
            if outliers > 0:
                logger.info(f"Column {col}: {outliers} outliers detected")