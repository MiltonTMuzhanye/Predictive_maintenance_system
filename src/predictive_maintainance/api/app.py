"""
FastAPI application for predictive maintenance API.
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
import logging

from .schemas import PredictionRequest, PredictionResponse, EquipmentData
from .inference import PredictionEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Predictive Maintenance API",
    description="API for predicting equipment failures",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize prediction engine
prediction_engine = PredictionEngine()

class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: str
    model_loaded: bool
    version: str

@app.on_event("startup")
async def startup_event():
    """Initialize resources on startup."""
    try:
        # Load models
        prediction_engine.load_models()
        logger.info("Models loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load models: {str(e)}")

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "message": "Predictive Maintenance API",
        "version": "1.0.0",
        "endpoints": [
            "/health",
            "/predict",
            "/predict-batch",
            "/model-info"
        ]
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        model_loaded=prediction_engine.is_ready(),
        version="1.0.0"
    )

@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict_failure(request: PredictionRequest):
    """
    Predict failure probability for a single equipment.
    
    - **equipment_id**: Unique identifier for the equipment
    - **features**: Sensor readings and equipment parameters
    - **threshold**: Optional prediction threshold (default: 0.5)
    """
    try:
        # Convert request to dataframe
        features_df = pd.DataFrame([request.features.dict()])
        
        # Make prediction
        prediction = prediction_engine.predict(features_df, request.threshold)
        
        return PredictionResponse(
            equipment_id=request.equipment_id,
            timestamp=datetime.now().isoformat(),
            **prediction
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-batch", tags=["Predictions"])
async def predict_batch_failures(requests: List[PredictionRequest]):
    """
    Predict failure probabilities for multiple equipment.
    
    Accepts a list of equipment data for batch processing.
    """
    try:
        predictions = []
        
        for request in requests:
            features_df = pd.DataFrame([request.features.dict()])
            prediction = prediction_engine.predict(features_df, request.threshold)
            
            response = {
                "equipment_id": request.equipment_id,
                "timestamp": datetime.now().isoformat(),
                **prediction
            }
            predictions.append(response)
        
        return {
            "batch_id": f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "timestamp": datetime.now().isoformat(),
            "total_predictions": len(predictions),
            "predictions": predictions
        }
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model-info", tags=["Model"])
async def get_model_info():
    """Get information about the loaded model."""
    if not prediction_engine.is_ready():
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    info = prediction_engine.get_model_info()
    return info

@app.get("/feature-importance", tags=["Model"])
async def get_feature_importance():
    """Get feature importance from the model."""
    if not prediction_engine.is_ready():
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    importance = prediction_engine.get_feature_importance()
    return importance

@app.post("/update-threshold", tags=["Model"])
async def update_threshold(threshold: float = Field(0.5, ge=0.0, le=1.0)):
    """Update prediction threshold."""
    prediction_engine.set_threshold(threshold)
    return {
        "message": f"Threshold updated to {threshold}",
        "threshold": threshold,
        "timestamp": datetime.now().isoformat()
    }

# Background task for model retraining
def retrain_model_task():
    """Background task for model retraining."""
    # This would be implemented to retrain the model periodically
    pass

@app.post("/retrain", tags=["Model"])
async def retrain_model(background_tasks: BackgroundTasks):
    """Trigger model retraining (background task)."""
    background_tasks.add_task(retrain_model_task)
    return {
        "message": "Model retraining started in background",
        "timestamp": datetime.now().isoformat()
    }