"""
Pydantic schemas for API request/response validation.
"""
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
import numpy as np

class EquipmentFeatures(BaseModel):
    """Equipment sensor features for prediction."""
    air_temperature_k: float = Field(..., alias="air_temperature_k", description="Air temperature in Kelvin")
    process_temperature_k: float = Field(..., alias="process_temperature_k", description="Process temperature in Kelvin")
    rotational_speed_rpm: int = Field(..., alias="rotational_speed_rpm", description="Rotational speed in RPM")
    torque_nm: float = Field(..., alias="torque_nm", description="Torque in Newton-meters")
    tool_wear_min: int = Field(..., alias="tool_wear_min", description="Tool wear in minutes")
    power: Optional[float] = Field(None, description="Calculated power")
    temperature_ratio: Optional[float] = Field(None, description="Temperature ratio")
    tool_wear_rate: Optional[float] = Field(None, description="Tool wear rate")
    stress_factor: Optional[float] = Field(None, description="Stress factor")
    type_encoded: Optional[int] = Field(None, description="Encoded equipment type")
    
    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "air_temperature_k": 298.1,
                "process_temperature_k": 308.6,
                "rotational_speed_rpm": 1551,
                "torque_nm": 42.8,
                "tool_wear_min": 0,
                "power": 66366.8,
                "temperature_ratio": 1.0352,
                "tool_wear_rate": 0.0,
                "stress_factor": 534.9,
                "type_encoded": 1
            }
        }

class PredictionRequest(BaseModel):
    """Request model for failure prediction."""
    equipment_id: str = Field(..., description="Unique equipment identifier")
    features: EquipmentFeatures = Field(..., description="Equipment sensor data")
    threshold: Optional[float] = Field(0.5, description="Prediction threshold", ge=0.0, le=1.0)
    
    @validator('threshold')
    def validate_threshold(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Threshold must be between 0 and 1')
        return v

class PredictionResult(BaseModel):
    """Individual prediction result."""
    failure_probability: float = Field(..., description="Probability of failure", ge=0.0, le=1.0)
    prediction: int = Field(..., description="Binary prediction (0=No Failure, 1=Failure)")
    alert_level: str = Field(..., description="Alert level (NORMAL, LOW, MEDIUM, HIGH, CRITICAL)")
    recommended_action: str = Field(..., description="Recommended maintenance action")
    confidence: float = Field(..., description="Prediction confidence", ge=0.0, le=1.0)

class PredictionSummary(BaseModel):
    """Summary of batch predictions."""
    total_equipment: int = Field(..., description="Total number of equipment predicted")
    predicted_failures: int = Field(..., description="Number of predicted failures")
    failure_rate_predicted: float = Field(..., description="Predicted failure rate")
    average_risk_score: float = Field(..., description="Average risk score")
    high_risk_count: int = Field(..., description="Count of high risk equipment")
    critical_risk_count: int = Field(..., description="Count of critical risk equipment")

class PredictionResponse(BaseModel):
    """Response model for failure prediction."""
    equipment_id: str = Field(..., description="Equipment identifier")
    timestamp: str = Field(..., description="Prediction timestamp")
    failure_probability: float = Field(..., description="Probability of failure")
    prediction: int = Field(..., description="Binary prediction")
    alert_level: str = Field(..., description="Alert level")
    recommended_action: str = Field(..., description="Recommended action")
    confidence: float = Field(..., description="Prediction confidence")
    
    class Config:
        json_schema_extra = {
            "example": {
                "equipment_id": "EQP-001",
                "timestamp": "2023-10-01T12:00:00Z",
                "failure_probability": 0.85,
                "prediction": 1,
                "alert_level": "CRITICAL",
                "recommended_action": "SCHEDULE IMMEDIATE MAINTENANCE",
                "confidence": 0.92
            }
        }

class ModelInfo(BaseModel):
    """Model information response."""
    model_name: str = Field(..., description="Model name")
    model_version: str = Field(..., description="Model version")
    training_date: str = Field(..., description="Date when model was trained")
    performance_metrics: Dict[str, float] = Field(..., description="Model performance metrics")
    feature_count: int = Field(..., description="Number of features")
    threshold: float = Field(..., description="Current prediction threshold")

class FeatureImportance(BaseModel):
    """Feature importance response."""
    feature: str = Field(..., description="Feature name")
    importance: float = Field(..., description="Feature importance score")
    importance_normalized: float = Field(..., description="Normalized importance")

class EquipmentData(BaseModel):
    """Complete equipment data for batch processing."""
    equipment_id: str
    timestamp: str
    features: EquipmentFeatures
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)