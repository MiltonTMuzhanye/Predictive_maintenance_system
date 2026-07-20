"""
Pydantic Schemas for API Validation
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

class PredictionRequest(BaseModel):
    """Single prediction request."""
    
    customer_id: str
    recency: Optional[float] = None
    frequency: Optional[float] = None
    monetary: Optional[float] = None
    avg_order_value: Optional[float] = None
    total_transactions: Optional[int] = None
    days_since_last_purchase: Optional[int] = None
    
    class Config:
        schema_extra = {
            "example": {
                "customer_id": "12345",
                "recency": 5,
                "frequency": 20,
                "monetary": 500,
                "avg_order_value": 25,
                "total_transactions": 15,
                "days_since_last_purchase": 3
            }
        }

class PredictionResponse(BaseModel):
    """Single prediction response."""
    
    customer_id: str
    prediction: int
    probability: float
    risk_level: str
    health_score: float
    health_status: str
    rul_days: Optional[int] = None
    confidence: Optional[float] = None
    timestamp: str

class BatchPredictionRequest(BaseModel):
    """Batch prediction request."""
    
    data: List[Dict[str, Any]]

class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""
    
    predictions: List[Dict[str, Any]]
    total: int
    timestamp: str

class HealthRequest(BaseModel):
    """Health assessment request."""
    
    customer_id: str
    probability: float
    recency: Optional[float] = None
    frequency: Optional[float] = None
    monetary: Optional[float] = None
    
    class Config:
        schema_extra = {
            "example": {
                "customer_id": "12345",
                "probability": 0.3,
                "recency": 5,
                "frequency": 20,
                "monetary": 500
            }
        }

class HealthResponse(BaseModel):
    """Health assessment response."""
    
    health_score: float
    health_status: str
    risk_level: str
    factors: List[str]
    recommendations: List[str]

class ModelInfoResponse(BaseModel):
    """Model information response."""
    
    model_type: str
    version: str
    features: List[str]
    thresholds: Dict[str, float]
    last_trained: str

class AlertRequest(BaseModel):
    """Alert configuration request."""
    
    type: str
    threshold: float
    severity: str
    enabled: bool

class AlertResponse(BaseModel):
    """Alert configuration response."""
    
    id: str
    type: str
    threshold: float
    severity: str
    enabled: bool
    created_at: str
    updated_at: str