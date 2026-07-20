"""
Feature Engineering Module

Creates advanced features for predictive maintenance.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from ..utils.logger import get_logger
from ..utils.exceptions import FeatureEngineeringError

logger = get_logger(__name__)

class FeatureEngineer:
    """Feature engineering pipeline for maintenance system."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize feature engineer with configuration."""
        self.config = config or {}
        
    def create_rfm_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create RFM (Recency, Frequency, Monetary) features.
        
        Args:
            df: Transaction data
            
        Returns:
            DataFrame with RFM features
        """
        logger.info("Creating RFM features...")
        
        # Create a copy with only necessary columns
        rfm_df = df[['CustomerID', 'InvoiceDate', 'Amount']].copy()
        
        # Reference date (latest date + 1 day)
        reference_date = rfm_df['InvoiceDate'].max() + timedelta(days=1)
        
        # Group by customer
        rfm = rfm_df.groupby('CustomerID').agg({
            'InvoiceDate': lambda x: (reference_date - x.max()).days,  # Recency
            'InvoiceDate': lambda x: len(x),  # Frequency (number of transactions)
            'Amount': 'sum'  # Monetary
        }).rename(columns={
            'InvoiceDate': 'Recency',
            'InvoiceDate': 'Frequency',
            'Amount': 'Monetary'
        })
        
        # Add RFM scores
        rfm['R_Score'] = pd.qcut(rfm['Recency'], q=5, labels=range(5, 0, -1)).astype(int)
        rfm['F_Score'] = pd.qcut(rfm['Frequency'], q=5, labels=range(1, 6)).astype(int)
        rfm['M_Score'] = pd.qcut(rfm['Monetary'], q=5, labels=range(1, 6)).astype(int)
        
        # RFM total score
        rfm['RFM_Score'] = rfm['R_Score'] * 100 + rfm['F_Score'] * 10 + rfm['M_Score']
        
        # RFM segments
        def get_segment(row):
            if row['RFM_Score'] >= 550:
                return 'Champions'
            elif row['RFM_Score'] >= 450:
                return 'Loyal'
            elif row['RFM_Score'] >= 350:
                return 'Potential'
            elif row['RFM_Score'] >= 250:
                return 'At Risk'
            else:
                return 'Lost'
                
        rfm['RFM_Segment'] = rfm.apply(get_segment, axis=1)
        
        logger.info(f"Created RFM features for {len(rfm)} customers")
        return rfm
        
    def create_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create behavioral features from transaction data.
        
        Args:
            df: Transaction data
            
        Returns:
            DataFrame with behavioral features
        """
        logger.info("Creating behavioral features...")
        
        behavioral = df.groupby('CustomerID').agg({
            'InvoiceNo': ['count', 'nunique'],
            'Amount': ['mean', 'std', 'sum', 'min', 'max']
        }).reset_index()
        
        behavioral.columns = [
            'CustomerID',
            'Total_Transactions',
            'Unique_Invoice_Count',
            'Avg_Order_Value',
            'Std_Order_Value',
            'Total_Amount',
            'Min_Order_Value',
            'Max_Order_Value'
        ]
        
        # Additional behavioral metrics
        behavioral['Avg_Items_Per_Transaction'] = (
            df.groupby('CustomerID')['Quantity'].sum() / 
            df.groupby('CustomerID')['InvoiceNo'].nunique()
        ).values
        
        behavioral['Order_Frequency'] = (
            df.groupby('CustomerID')['InvoiceDate'].count() / 
            (df.groupby('CustomerID')['InvoiceDate'].max() - 
             df.groupby('CustomerID')['InvoiceDate'].min()).dt.days
        ).fillna(0)
        
        # Purchase velocity
        time_diff = df.groupby('CustomerID')['InvoiceDate'].diff().dt.days
        behavioral['Avg_Purchase_Interval'] = time_diff.groupby(
            df['CustomerID']
        ).mean().fillna(0)
        
        logger.info(f"Created behavioral features for {len(behavioral)} customers")
        return behavioral
        
    def create_time_series_features(
        self,
        df: pd.DataFrame,
        window_sizes: List[int] = [7, 14, 30]
    ) -> pd.DataFrame:
        """
        Create time series features using rolling windows.
        
        Args:
            df: Transaction data with InvoiceDate
            window_sizes: List of window sizes in days
            
        Returns:
            DataFrame with time series features
        """
        logger.info(f"Creating time series features with windows: {window_sizes}")
        
        ts_features = []
        
        for customer_id in df['CustomerID'].unique():
            customer_data = df[df['CustomerID'] == customer_id].copy()
            customer_data = customer_data.sort_values('InvoiceDate')
            
            # Calculate rolling features
            features = {'CustomerID': customer_id}
            
            for window in window_sizes:
                window_data = customer_data.tail(window)
                if len(window_data) > 0:
                    features[f'Amount_Window_{window}_Mean'] = window_data['Amount'].mean()
                    features[f'Amount_Window_{window}_Std'] = window_data['Amount'].std()
                    features[f'Amount_Window_{window}_Min'] = window_data['Amount'].min()
                    features[f'Amount_Window_{window}_Max'] = window_data['Amount'].max()
                    features[f'Quantity_Window_{window}_Mean'] = window_data['Quantity'].mean()
                    features[f'Transaction_Count_Window_{window}'] = window_data['InvoiceNo'].nunique()
                    
                    # Trend feature
                    if len(window_data) > 1:
                        x = np.arange(len(window_data))
                        y = window_data['Amount'].values
                        trend = np.polyfit(x, y, 1)[0]
                        features[f'Trend_Window_{window}'] = trend
                    else:
                        features[f'Trend_Window_{window}'] = 0
            
            ts_features.append(features)
            
        ts_df = pd.DataFrame(ts_features)
        logger.info(f"Created time series features for {len(ts_df)} customers")
        return ts_df
        
    def create_customer_lifetime_value(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Customer Lifetime Value (CLV) features.
        
        Args:
            df: Transaction data
            
        Returns:
            DataFrame with CLV features
        """
        logger.info("Creating Customer Lifetime Value features...")
        
        clv = df.groupby('CustomerID').agg({
            'Amount': ['sum', 'mean', 'count'],
            'InvoiceDate': ['min', 'max', 'count']
        }).reset_index()
        
        clv.columns = [
            'CustomerID',
            'Total_Value',
            'Avg_Value',
            'Transaction_Count',
            'First_Purchase_Date',
            'Last_Purchase_Date',
            'Total_Days_Active'
        ]
        
        # Customer lifetime in days
        clv['Lifetime_Days'] = (clv['Last_Purchase_Date'] - clv['First_Purchase_Date']).dt.days
        
        # Average transaction value per month
        clv['Value_Per_Day'] = clv['Total_Value'] / (clv['Lifetime_Days'] + 1)
        clv['Value_Per_Transaction'] = clv['Total_Value'] / clv['Transaction_Count']
        
        # Churn prediction features
        clv['Purchase_Frequency'] = clv['Transaction_Count'] / (clv['Lifetime_Days'] + 1)
        clv['Customer_Age'] = (df['InvoiceDate'].max() - clv['First_Purchase_Date']).dt.days
        
        # Velocity and acceleration
        clv['Value_Velocity'] = clv['Value_Per_Day']
        
        logger.info(f"Created CLV features for {len(clv)} customers")
        return clv
        
    def create_churn_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create churn indicators and target labels.
        
        Args:
            df: Transaction data
            
        Returns:
            DataFrame with churn labels
        """
        logger.info("Creating churn indicators...")
        
        # Get last transaction per customer
        last_transaction = df.groupby('CustomerID')['InvoiceDate'].max()
        
        # Calculate days since last transaction
        reference_date = df['InvoiceDate'].max()
        days_since_last = (reference_date - last_transaction).dt.days
        
        # Create churn labels with different thresholds
        churn_df = pd.DataFrame({
            'CustomerID': days_since_last.index,
            'Days_Since_Last_Purchase': days_since_last.values
        })
        
        # Churn labels for different time horizons
        churn_df['Churn_30'] = (churn_df['Days_Since_Last_Purchase'] >= 30).astype(int)
        churn_df['Churn_60'] = (churn_df['Days_Since_Last_Purchase'] >= 60).astype(int)
        churn_df['Churn_90'] = (churn_df['Days_Since_Last_Purchase'] >= 90).astype(int)
        
        # At-risk labels
        churn_df['At_Risk'] = (
            (churn_df['Days_Since_Last_Purchase'] >= 15) &
            (churn_df['Days_Since_Last_Purchase'] < 30)
        ).astype(int)
        
        # Health score (inverse of churn probability)
        churn_df['Health_Score'] = 1 - (churn_df['Days_Since_Last_Purchase'] / 365)
        churn_df['Health_Score'] = churn_df['Health_Score'].clip(0, 1)
        
        # Churn probability (logistic transformation)
        churn_df['Churn_Probability'] = 1 / (1 + np.exp(-(churn_df['Days_Since_Last_Purchase'] - 45) / 15))
        
        logger.info(f"Created churn indicators for {len(churn_df)} customers")
        return churn_df
        
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all features and combine into single DataFrame.
        
        Args:
            df: Raw transaction data
            
        Returns:
            Combined feature DataFrame
        """
        logger.info("Creating all features...")
        
        # Create individual feature sets
        rfm_features = self.create_rfm_features(df)
        behavioral_features = self.create_behavioral_features(df)
        ts_features = self.create_time_series_features(df)
        clv_features = self.create_customer_lifetime_value(df)
        churn_features = self.create_churn_indicators(df)
        
        # Merge all features
        features = rfm_features.reset_index()
        
        # Merge behavioral features
        features = features.merge(
            behavioral_features,
            on='CustomerID',
            how='left'
        )
        
        # Merge time series features
        features = features.merge(
            ts_features,
            on='CustomerID',
            how='left'
        )
        
        # Merge CLV features
        features = features.merge(
            clv_features,
            on='CustomerID',
            how='left'
        )
        
        # Merge churn indicators
        features = features.merge(
            churn_features,
            on='CustomerID',
            how='left'
        )
        
        # Clean up column names
        features.columns = [col.replace(' ', '_') for col in features.columns]
        
        # Handle missing values
        features = features.fillna(0)
        
        logger.info(f"Created {len(features.columns) - 1} features for {len(features)} customers")
        return features