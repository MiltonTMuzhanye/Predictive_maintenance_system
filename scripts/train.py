"""
Training script for predictive maintenance models.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import argparse
from src.predictive_maintenance.data.ingestion import DataIngestion
from src.predictive_maintenance.training.trainer import ModelTrainer
from src.predictive_maintenance.utils.logger import get_logger
from src.predictive_maintenance.utils.config import load_config

logger = get_logger(__name__)

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train predictive maintenance models")
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--data', type=str, default='data/raw/Online_Retail.xlsx',
                       help='Path to data file')
    parser.add_argument('--target', type=str, default='Churn_30',
                       help='Target column name')
    args = parser.parse_args()
    
    logger.info("Starting training pipeline...")
    
    # Load configuration
    config = load_config(args.config)
    
    # Ingest data
    ingestion = DataIngestion(args.config)
    df = ingestion.load_data()
    df = ingestion.preprocess_data(df)
    
    # Train models
    trainer = ModelTrainer(args.config)
    results = trainer.run_pipeline(df)
    
    # Print results
    logger.info("Training completed successfully!")
    logger.info(f"Best model: {results['best_model']}")
    logger.info(f"Results: {results['results'][results['best_model']]}")
    
    return results

if __name__ == "__main__":
    main()