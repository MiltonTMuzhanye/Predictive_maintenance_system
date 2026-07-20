"""
API Routes

Defines all API endpoints for the predictive maintenance system.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict, Any, Optional
import pandas as pd
from datetime import datetime

from .schemas import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthRequest,
    HealthResponse,
    ModelInfoResponse
)
from ..inference.predictor import Predictor
from ..inference.health_engine import HealthEngine
from ..inference.rul_estimator import RULEstimator
from ..utils.logger import get_logger

logger = get_logger(__name__)

# Initialize router
router = APIRouter()

# Initialize components
predictor = Predictor("artifacts/trained_models/xgboost_best")
health_engine = HealthEngine()
rul_estimator = RULEstimator()

@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make a single prediction.
    
    Args:
        request: Prediction request data
        
    Returns:
        Prediction response
    """
    try:
        logger.info(f"Making prediction for customer: {request.customer_id}")
        
        # Convert request to DataFrame
        data = request.dict()
        customer_id = data.pop('customer_id')
        df = pd.DataFrame([data])
        
        # Make prediction
        result = predictor.predict(df)
        result['customer_id'] = customer_id
        
        # Get health assessment
        health = health_engine.assess_health(result)
        result.update(health)
        
        # Estimate RUL
        rul = rul_estimator.estimate(result)
        result.update(rul)
        
        return PredictionResponse(**result)
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    Make predictions for multiple customers.
    
    Args:
        request: Batch prediction request
        
    Returns:
        Batch prediction response
    """
    try:
        logger.info(f"Making batch predictions for {len(request.data)} customers")
        
        # Convert to DataFrame
        df = pd.DataFrame(request.data)
        
        # Make predictions
        results = predictor.predict_batch(df)
        
        # Add health assessments and RUL estimates
        for idx, row in results.iterrows():
            health = health_engine.assess_health(row.to_dict())
            rul = rul_estimator.estimate(row.to_dict())
            for key, value in health.items():
                results.at[idx, key] = value
            for key, value in rul.items():
                results.at[idx, key] = value
                
        # Convert to list of dicts
        predictions = results.to_dict('records')
        
        return BatchPredictionResponse(
            predictions=predictions,
            total=len(predictions),
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/health", response_model=HealthResponse)
async def assess_health(request: HealthRequest):
    """
    Assess customer health score.
    
    Args:
        request: Health assessment request
        
    Returns:
        Health assessment response
    """
    try:
        health = health_engine.assess_health(request.dict())
        return HealthResponse(**health)
        
    except Exception as e:
        logger.error(f"Health assessment failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    """
    Get model information and metadata.
    
    Returns:
        Model information
    """
    try:
        return ModelInfoResponse(
            model_type="xgboost",
            version="1.0.0",
            features=predictor.feature_names,
            thresholds=predictor.thresholds,
            last_trained=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Failed to get model info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health/status")
async def get_system_health():
    """
    Get system health status.
    
    Returns:
        System health status
    """
    return {
        "status": "operational",
        "components": {
            "api": "healthy",
            "model": "healthy",
            "inference": "healthy",
            "database": "healthy"
        },
        "timestamp": datetime.now().isoformat()
    }