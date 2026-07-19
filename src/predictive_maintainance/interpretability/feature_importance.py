"""
Feature importance analysis for predictive maintenance.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class FeatureImportanceAnalyzer:
    """Analyze and visualize feature importance."""
    
    def __init__(self):
        plt.style.use('fivethirtyeight')
    
    def analyze_importance(self, model, feature_names: List[str]) -> pd.DataFrame:
        """
        Analyze feature importance from model.
        
        Args:
            model: Trained model with feature_importances_ attribute
            feature_names: List of feature names
            
        Returns:
            DataFrame with feature importance
        """
        if not hasattr(model, 'feature_importances_'):
            logger.error("Model doesn't have feature_importances_ attribute")
            return pd.DataFrame()
        
        importances = model.feature_importances_
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances,
            'importance_normalized': importances / importances.sum()
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def plot_feature_importance(self, importance_df: pd.DataFrame, 
                               top_n: int = 15, 
                               title: str = "Feature Importance"):
        """Plot feature importance."""
        if importance_df.empty:
            logger.warning("No importance data to plot")
            return
        
        # Select top N features
        top_features = importance_df.head(top_n)
        
        plt.figure(figsize=(12, 8))
        bars = plt.barh(top_features['feature'], top_features['importance_normalized'])
        plt.xlabel('Normalized Importance')
        plt.title(title)
        plt.gca().invert_yaxis()
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height()/2,
                    f'{width:.3f}',
                    ha='left', va='center')
        
        plt.tight_layout()
        plt.show()
    
    def plot_correlation_heatmap(self, features: pd.DataFrame, 
                                target: pd.Series = None,
                                figsize: tuple = (14, 12)):
        """Plot correlation heatmap."""
        # Calculate correlations
        if target is not None:
            # Add target to dataframe for correlation
            df_corr = features.copy()
            df_corr['Target'] = target
            corr_matrix = df_corr.corr()
        else:
            corr_matrix = features.corr()
        
        # Plot heatmap
        plt.figure(figsize=figsize)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
                   cmap='coolwarm', center=0, square=True,
                   cbar_kws={"shrink": .8})
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.show()
    
    def analyze_feature_relationships(self, features: pd.DataFrame, 
                                     target: pd.Series,
                                     top_n: int = 10):
        """Analyze relationships between features and target."""
        # Calculate correlations with target
        correlations = features.corrwith(target).abs().sort_values(ascending=False)
        
        print("Top feature correlations with failure:")
        for feature, corr in correlations.head(top_n).items():
            print(f"{feature}: {corr:.4f}")
        
        # Plot top correlations
        plt.figure(figsize=(10, 6))
        correlations.head(top_n).plot(kind='bar', color='orange')
        plt.title('Feature Correlation with Equipment Failure')
        plt.ylabel('Absolute Correlation')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        return correlations