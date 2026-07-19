"""
Data Ingestion Module

Handles loading data from various sources with validation and preprocessing.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Union
from datetime import datetime
import logging
from ..utils.logger import get_logger
from ..utils.config import load_config
from ..utils.exceptions import DataIngestionError

logger = get_logger(__name__)

class DataIngestion:
    """Data ingestion handler for maintenance system."""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        Initialize data ingestion with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.data_config = self.config.get('data', {})
        self.source_config = self.data_config.get('source', {})
        self.raw_path = Path(self.config['paths']['data']['raw'])
        self.processed_path = Path(self.config['paths']['data']['processed'])
        
        # Create directories
        self.raw_path.mkdir(parents=True, exist_ok=True)
        self.processed_path.mkdir(parents=True, exist_ok=True)
        
    def load_data(self) -> pd.DataFrame:
        """
        Load data from configured source.
        
        Returns:
            DataFrame with loaded data
            
        Raises:
            DataIngestionError: If data loading fails
        """
        try:
            source_type = self.source_config.get('type', 'excel')
            source_path = self.source_config.get('path', 'data/raw/Online_Retail.xlsx')
            
            logger.info(f"Loading data from {source_path} (type: {source_type})")
            
            if source_type == 'excel':
                df = pd.read_excel(
                    source_path,
                    sheet_name=self.source_config.get('sheet_name', 0)
                )
            elif source_type == 'csv':
                df = pd.read_csv(source_path)
            elif source_type == 'parquet':
                df = pd.read_parquet(source_path)
            elif source_type == 'database':
                df = self._load_from_database()
            else:
                raise DataIngestionError(f"Unsupported source type: {source_type}")
                
            logger.info(f"Successfully loaded {len(df)} records")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
            raise DataIngestionError(f"Data loading failed: {str(e)}")
            
    def _load_from_database(self) -> pd.DataFrame:
        """Load data from database connection."""
        # Implement database loading logic
        pass
        
    def validate_schema(self, df: pd.DataFrame) -> bool:
        """
        Validate data against expected schema.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            True if validation passes
            
        Raises:
            DataIngestionError: If validation fails
        """
        try:
            schema_config = self.data_config.get('validation', {})
            required_cols = schema_config.get('required_columns', [])
            
            # Check required columns
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise DataIngestionError(f"Missing required columns: {missing_cols}")
            
            # Check data types
            dtype_config = schema_config.get('data_types', {})
            for col, expected_dtype in dtype_config.items():
                if col in df.columns:
                    actual_dtype = str(df[col].dtype)
                    if expected_dtype not in actual_dtype:
                        logger.warning(
                            f"Column {col} expected dtype {expected_dtype}, "
                            f"got {actual_dtype}"
                        )
            
            # Check constraints
            constraints = schema_config.get('constraints', {})
            for col, constraint in constraints.items():
                if col in df.columns:
                    if 'min' in constraint:
                        min_val = constraint['min']
                        invalid = df[df[col] < min_val][col].count()
                        if invalid > 0:
                            logger.warning(f"Column {col} has {invalid} values below {min_val}")
                            
                    if 'max' in constraint:
                        max_val = constraint['max']
                        invalid = df[df[col] > max_val][col].count()
                        if invalid > 0:
                            logger.warning(f"Column {col} has {invalid} values above {max_val}")
            
            logger.info("Schema validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Schema validation failed: {str(e)}")
            raise DataIngestionError(f"Schema validation failed: {str(e)}")
            
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply preprocessing steps to data.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        try:
            logger.info("Starting data preprocessing...")
            original_count = len(df)
            
            # Convert date columns
            date_cols = ['InvoiceDate']
            for col in date_cols:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])
            
            # Filter data based on conditions
            filter_conditions = self.data_config.get('preprocessing', {}).get(
                'filter_conditions', []
            )
            for condition in filter_conditions:
                try:
                    df = df.query(condition)
                except Exception as e:
                    logger.warning(f"Could not apply filter condition '{condition}': {str(e)}")
            
            # Create derived columns
            derived_cols = self.data_config.get('preprocessing', {}).get(
                'derived_columns', {}
            )
            for col_name, expression in derived_cols.items():
                try:
                    if 'Quantity' in expression and 'UnitPrice' in expression:
                        df[col_name] = df['Quantity'] * df['UnitPrice']
                    elif 'notna' in expression:
                        df[col_name] = df[col_name.split('[')[1].split(']')[0]]
                    elif 'astype' in expression:
                        dtype = expression.split('(')[1].split(')')[0]
                        col = expression.split('[')[1].split(']')[0]
                        df[col_name] = df[col].astype(dtype)
                except Exception as e:
                    logger.warning(f"Could not create derived column {col_name}: {str(e)}")
            
            # Add date features
            if 'InvoiceDate' in df.columns:
                df['year'] = df['InvoiceDate'].dt.year
                df['month'] = df['InvoiceDate'].dt.month
                df['day'] = df['InvoiceDate'].dt.day
                df['day_of_week'] = df['InvoiceDate'].dt.dayofweek
                df['hour'] = df['InvoiceDate'].dt.hour
                df['quarter'] = df['InvoiceDate'].dt.quarter
            
            # Log preprocessing stats
            final_count = len(df)
            removed_count = original_count - final_count
            logger.info(
                f"Preprocessing complete: {original_count} -> {final_count} "
                f"({removed_count} records removed)"
            )
            
            return df
            
        except Exception as e:
            logger.error(f"Preprocessing failed: {str(e)}")
            raise DataIngestionError(f"Preprocessing failed: {str(e)}")
            
    def save_processed_data(self, df: pd.DataFrame, filename: str = "processed_data.parquet"):
        """Save processed data to disk."""
        try:
            filepath = self.processed_path / filename
            df.to_parquet(filepath, index=False)
            logger.info(f"Saved processed data to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save processed data: {str(e)}")
            raise DataIngestionError(f"Failed to save processed data: {str(e)}")
            
    def load_processed_data(self, filename: str = "processed_data.parquet") -> pd.DataFrame:
        """Load processed data from disk."""
        try:
            filepath = self.processed_path / filename
            if filepath.exists():
                df = pd.read_parquet(filepath)
                logger.info(f"Loaded processed data from {filepath}")
                return df
            else:
                raise FileNotFoundError(f"Processed data file not found: {filepath}")
        except Exception as e:
            logger.error(f"Failed to load processed data: {str(e)}")
            raise DataIngestionError(f"Failed to load processed data: {str(e)}")
            
    def run(self, save_processed: bool = True) -> pd.DataFrame:
        """
        Run the complete ingestion pipeline.
        
        Args:
            save_processed: Whether to save processed data
            
        Returns:
            Processed DataFrame
        """
        logger.info("Starting data ingestion pipeline...")
        
        # Load data
        df = self.load_data()
        
        # Validate schema
        self.validate_schema(df)
        
        # Preprocess data
        df = self.preprocess_data(df)
        
        # Save if requested
        if save_processed:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"processed_data_{timestamp}.parquet"
            self.save_processed_data(df, filename)
            
        logger.info("Data ingestion pipeline completed successfully")
        return df