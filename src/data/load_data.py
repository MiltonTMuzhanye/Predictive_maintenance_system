"""
Data loading module for predictive maintenance system.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Tuple, Dict, Optional

logger = logging.getLogger(__name__)

class DataLoader:
    """Load and validate predictive maintenance data."""
    
    def __init__(self, data_path: str = "data/raw/"):
        self.data_path = Path(data_path)
        self.failure_mapping = {
            'TWF': 'Tool Wear Failure',
            'HDF': 'Heat Dissipation Failure',
            'PWF': 'Power Failure',
            'OSF': 'Overstrain Failure',
            'RNF': 'Random Failures'
        }
    
    def load_equipment_data(self, filename: str = "ai4i2020.csv") -> pd.DataFrame:
        """Load main equipment sensor data."""
        file_path = self.data_path / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        df = pd.read_csv(file_path)
        logger.info(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
        
        return df
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict:
        """Generate comprehensive data summary."""
        summary = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'failure_summary': self._analyze_failures(df)
        }
        return summary
    
    def _analyze_failures(self, df: pd.DataFrame) -> Dict:
        """Analyze failure type distributions."""
        failure_columns = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
        
        failure_counts = df[failure_columns].sum().to_dict()
        failure_counts_named = {}
        
        for code, count in failure_counts.items():
            failure_counts_named[self.failure_mapping[code]] = int(count)
        
        total_failures = df['Machine failure'].sum()
        
        return {
            'individual_failures': failure_counts_named,
            'total_individual_occurrences': sum(failure_counts.values()),
            'machine_failures': int(total_failures),
            'failure_rate': float(total_failures / len(df))
        }
    
    def split_data(self, df: pd.DataFrame, test_size: float = 0.2, 
                   random_state: int = 42) -> Tuple:
        """Split data into train and test sets with stratification."""
        from sklearn.model_selection import train_test_split
        
        # Prepare features and target
        X = df.drop('Machine failure', axis=1)
        y = df['Machine failure']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, 
            stratify=y
        )
        
        logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
        logger.info(f"Train failure rate: {y_train.mean():.4f}")
        
        return X_train, X_test, y_train, y_test