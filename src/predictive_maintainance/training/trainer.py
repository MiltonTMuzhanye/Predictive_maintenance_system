"""
Training Pipeline

Orchestrates model training, validation, and saving.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from ..utils.logger import get_logger
from ..utils.config import load_config
from ..utils.exceptions import TrainingError
from ..features.engineering import FeatureEngineer
from ..models.xgboost_model import XGBoostModel
from ..models.lightgbm_model import LightGBMModel
from ..models.random_forest import RandomForestModel
from ..models.lstm_model import LSTMModel
from ..models.autoencoder import AutoencoderModel
from ..evaluation.metrics import ModelMetrics
from .hyperparameter_tuning import HyperparameterTuner

logger = get_logger(__name__)

class ModelTrainer:
    """Orchestrates model training pipeline."""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize trainer with configuration."""
        self.config = load_config(config_path)
        self.model_config = self.config.get('modeling', {})
        self.artifacts_path = Path(self.config['paths']['artifacts']['models'])
        self.scalers_path = Path(self.config['paths']['artifacts']['scalers'])
        self.features_path = Path(self.config['paths']['artifacts']['features'])
        
        # Create directories
        self.artifacts_path.mkdir(parents=True, exist_ok=True)
        self.scalers_path.mkdir(parents=True, exist_ok=True)
        self.features_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.feature_engineer = FeatureEngineer()
        self.tuner = HyperparameterTuner()
        
    def prepare_data(
        self,
        df: pd.DataFrame,
        target_col: str = 'Churn_30'
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, Dict]:
        """
        Prepare data for training.
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            
        Returns:
            Train/test splits and metadata
        """
        logger.info("Preparing data for training...")
        
        # Create features
        features_df = self.feature_engineer.create_all_features(df)
        
        # Separate features and target
        X = features_df.drop(columns=['CustomerID', target_col], errors='ignore')
        y = features_df[target_col]
        
        # Store feature names
        feature_names = X.columns.tolist()
        
        # Split data (time-based split)
        # Assuming we want to predict future behavior
        # Sort by first purchase date (if available)
        if 'First_Purchase_Date' in X.columns:
            X['First_Purchase_Date'] = pd.to_datetime(X['First_Purchase_Date'])
            sorted_idx = X.sort_values('First_Purchase_Date').index
            X = X.loc[sorted_idx]
            y = y.loc[sorted_idx]
            
        # Split into train and test (80/20)
        split_idx = int(0.8 * len(X))
        X_train = X.iloc[:split_idx].copy()
        X_test = X.iloc[split_idx:].copy()
        y_train = y.iloc[:split_idx].copy()
        y_test = y.iloc[split_idx:].copy()
        
        # Remove date columns from features
        date_cols = ['First_Purchase_Date', 'Last_Purchase_Date']
        X_train = X_train.drop(columns=date_cols, errors='ignore')
        X_test = X_test.drop(columns=date_cols, errors='ignore')
        
        logger.info(f"Data split: Train {len(X_train)}, Test {len(X_test)}")
        
        # Handle class imbalance
        class_dist = y_train.value_counts()
        logger.info(f"Class distribution: {class_dist.to_dict()}")
        
        if class_dist[0] / class_dist[1] > 3:
            logger.info("Applying SMOTE for class imbalance...")
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            logger.info(f"After SMOTE: {len(X_train)} samples")
            
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Convert back to DataFrame with feature names
        X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns)
        
        metadata = {
            'feature_names': feature_names,
            'scaler': scaler,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'class_distribution': class_dist.to_dict()
        }
        
        return X_train, y_train, X_test, y_test, metadata
        
    def train_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, Dict]:
        """
        Train multiple models and return results.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of model results
        """
        logger.info("Training multiple models...")
        
        models = {}
        results = {}
        
        # 1. XGBoost
        try:
            logger.info("Training XGBoost...")
            xgb_model = XGBoostModel()
            xgb_model.build_model(self.model_config.get('models', {}).get('xgboost', {}).get('params', {}))
            
            with mlflow.start_run(run_name="xgboost"):
                xgb_model.train(X_train, y_train, X_test, y_test)
                results['xgboost'] = xgb_model.evaluate(X_test, y_test)
                models['xgboost'] = xgb_model
                
                # Log to MLflow
                for key, value in results['xgboost'].items():
                    if not isinstance(value, (list, dict)):
                        mlflow.log_metric(key, value)
                        
                # Log feature importance
                importance = xgb_model.get_feature_importance()
                importance.to_csv("feature_importance_xgboost.csv")
                mlflow.log_artifact("feature_importance_xgboost.csv")
                
        except Exception as e:
            logger.error(f"XGBoost training failed: {str(e)}")
            results['xgboost'] = {'error': str(e)}
            
        # 2. LightGBM
        try:
            logger.info("Training LightGBM...")
            lgb_model = LightGBMModel()
            lgb_model.build_model(self.model_config.get('models', {}).get('lightgbm', {}).get('params', {}))
            
            with mlflow.start_run(run_name="lightgbm"):
                lgb_model.train(X_train, y_train, X_test, y_test)
                results['lightgbm'] = lgb_model.evaluate(X_test, y_test)
                models['lightgbm'] = lgb_model
                
                # Log to MLflow
                for key, value in results['lightgbm'].items():
                    if not isinstance(value, (list, dict)):
                        mlflow.log_metric(key, value)
                        
        except Exception as e:
            logger.error(f"LightGBM training failed: {str(e)}")
            results['lightgbm'] = {'error': str(e)}
            
        # 3. Random Forest
        try:
            logger.info("Training Random Forest...")
            rf_model = RandomForestModel()
            rf_model.build_model(self.model_config.get('models', {}).get('random_forest', {}).get('params', {}))
            
            with mlflow.start_run(run_name="random_forest"):
                rf_model.train(X_train, y_train, X_test, y_test)
                results['random_forest'] = rf_model.evaluate(X_test, y_test)
                models['random_forest'] = rf_model
                
                # Log to MLflow
                for key, value in results['random_forest'].items():
                    if not isinstance(value, (list, dict)):
                        mlflow.log_metric(key, value)
                        
        except Exception as e:
            logger.error(f"Random Forest training failed: {str(e)}")
            results['random_forest'] = {'error': str(e)}
            
        # 4. LSTM (Deep Learning)
        try:
            logger.info("Training LSTM...")
            lstm_model = LSTMModel()
            lstm_model.build_model(self.model_config.get('models', {}).get('lstm', {}).get('params', {}))
            
            with mlflow.start_run(run_name="lstm"):
                lstm_model.train(X_train.values, y_train.values, X_test.values, y_test.values)
                results['lstm'] = lstm_model.evaluate(X_test.values, y_test.values)
                models['lstm'] = lstm_model
                
        except Exception as e:
            logger.error(f"LSTM training failed: {str(e)}")
            results['lstm'] = {'error': str(e)}
            
        # 5. Autoencoder (Anomaly Detection)
        try:
            logger.info("Training Autoencoder...")
            ae_model = AutoencoderModel()
            ae_model.build_model(self.model_config.get('models', {}).get('autoencoder', {}).get('params', {}))
            
            with mlflow.start_run(run_name="autoencoder"):
                ae_model.train(X_train.values, X_test.values)
                results['autoencoder'] = ae_model.evaluate(X_test.values)
                models['autoencoder'] = ae_model
                
        except Exception as e:
            logger.error(f"Autoencoder training failed: {str(e)}")
            results['autoencoder'] = {'error': str(e)}
            
        # Select best model
        best_model = self._select_best_model(results, models)
        logger.info(f"Best model: {best_model}")
        
        return {
            'models': models,
            'results': results,
            'best_model': best_model
        }
        
    def _select_best_model(
        self,
        results: Dict,
        models: Dict
    ) -> str:
        """
        Select best model based on F1 score.
        
        Args:
            results: Model results
            models: Model instances
            
        Returns:
            Name of best model
        """
        best_score = -1
        best_name = None
        
        for name, metrics in results.items():
            if 'error' not in metrics and 'f1' in metrics:
                if metrics['f1'] > best_score:
                    best_score = metrics['f1']
                    best_name = name
                    
        return best_name
        
    def save_artifacts(
        self,
        models: Dict,
        metadata: Dict,
        results: Dict,
        best_model_name: str
    ):
        """
        Save trained models and metadata.
        
        Args:
            models: Dictionary of trained models
            metadata: Training metadata
            results: Model results
            best_model_name: Name of best model
        """
        logger.info("Saving artifacts...")
        
        # Save best model
        best_model = models.get(best_model_name)
        if best_model:
            model_path = self.artifacts_path / f"{best_model_name}_best"
            best_model.save_model(str(model_path))
            
        # Save all models
        for name, model in models.items():
            if model:
                model_path = self.artifacts_path / name
                model.save_model(str(model_path))
                
        # Save metadata
        metadata_path = self.artifacts_path / "training_metadata.json"
        import json
        with open(metadata_path, 'w') as f:
            json.dump({
                'models': list(models.keys()),
                'best_model': best_model_name,
                'results': {k: v for k, v in results.items() if isinstance(v, dict)},
                'metadata': {k: str(v) if hasattr(v, '__str__') else v for k, v in metadata.items()}
            }, f, default=str)
            
        # Save feature names
        feature_path = self.features_path / "feature_names.json"
        with open(feature_path, 'w') as f:
            json.dump(metadata['feature_names'], f)
            
        # Save scaler
        import joblib
        scaler_path = self.scalers_path / "scaler.pkl"
        joblib.dump(metadata['scaler'], scaler_path)
        
        logger.info("Artifacts saved successfully")
        
    def run_pipeline(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Run complete training pipeline.
        
        Args:
            df: Input transaction data
            
        Returns:
            Pipeline results
        """
        logger.info("Starting training pipeline...")
        
        # Prepare data
        X_train, y_train, X_test, y_test, metadata = self.prepare_data(df)
        
        # Train models
        training_results = self.train_models(X_train, y_train, X_test, y_test)
        
        # Save artifacts
        self.save_artifacts(
            training_results['models'],
            metadata,
            training_results['results'],
            training_results['best_model']
        )
        
        logger.info("Training pipeline completed successfully")
        return {
            'results': training_results['results'],
            'best_model': training_results['best_model'],
            'metadata': metadata
        }