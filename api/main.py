"""
FastAPI Service for Fake News Detection

This module provides a REST API for the fake news detection model
with endpoints for single and batch predictions.

Features:
- Single text prediction
- Batch text prediction
- Health check endpoint
- Model information endpoint
- Comprehensive error handling
- Request validation
- Response documentation
"""

import os
import sys
import time
from typing import List, Dict, Any, Optional
from fastapi import HTTPException, Security, Depends
from fastapi.security import APIKeyHeader
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from fastapi import FastAPI, HTTPException, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import uvicorn
import logging

from src.predict import FakeNewsPredictor, PredictionResult

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Key Configuration
API_KEY_NAME = "X-API-Key"
API_KEY = os.getenv("API_KEY", "fake_news_detection_key_571")
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

async def get_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API Key"
        )
    return api_key

# Initialize FastAPI app
app = FastAPI(
    title="Fake News Detection API",
    description="A REST API for detecting fake news using deep learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and preprocessor
predictor: Optional[FakeNewsPredictor] = None
model_loaded = False


# Pydantic models for request/response
class TextInput(BaseModel):
    text: str = Field(..., min_length=10, max_length=10000, description="Text to classify")
    
    @validator('text')
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError('Text cannot be empty')
        return v.strip()


class BatchTextInput(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=100, description="List of texts to classify")
    
    @validator('texts')
    def validate_texts(cls, v):
        if not v:
            raise ValueError('Texts list cannot be empty')
        validated_texts = []
        for text in v:
            if text and text.strip():
                validated_texts.append(text.strip())
        if not validated_texts:
            raise ValueError('All texts are empty')
        return validated_texts


class PredictionResponse(BaseModel):
    text: str
    prediction: str  # "FAKE" or "TRUE"
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    category: str
    category_confidence: float = Field(..., ge=0.0, le=1.0, description="Category confidence")
    processing_time: float


class BatchPredictionResponse(BaseModel):
    results: List[PredictionResponse]
    total_texts: int
    fake_count: int
    true_count: int
    average_confidence: float
    processing_time: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    timestamp: str


class ModelInfoResponse(BaseModel):
    model_loaded: bool
    preprocessor_loaded: bool
    model_parameters: int
    model_layers: int
    vocabulary_size: int
    num_subjects: int
    subject_classes: List[str]


class ErrorResponse(BaseModel):
    error: str
    message: str
    timestamp: str


# Startup event to load the model
@app.on_event("startup")
async def startup_event():
    """Load the model and preprocessor on startup."""
    global predictor, model_loaded
    
    model_path = os.environ.get('MODEL_PATH')
    preprocessor_path = os.environ.get('PREPROCESSOR_PATH')
    
    if not model_path:
        logger.warning("MODEL_PATH environment variable not set")
        return
    
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return
    
    try:
        # Determine preprocessor path if not provided
        if not preprocessor_path:
            preprocessor_path = model_path.replace('.h5', '_preprocessor.pkl')
        
        # Load predictor
        predictor = FakeNewsPredictor(model_path, preprocessor_path)
        model_loaded = True
        
        logger.info(f"Model loaded successfully from {model_path}")
        logger.info(f"Preprocessor loaded from {preprocessor_path}")
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        model_loaded = False


# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if the API is healthy and model is loaded."""
    from datetime import datetime
    
    return HealthResponse(
        status="healthy" if model_loaded else "unhealthy",
        model_loaded=model_loaded,
        timestamp=datetime.utcnow().isoformat()
    )


# Model information endpoint
@app.get("/model-info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get information about the loaded model."""
    if not model_loaded or not predictor:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    info = predictor.get_model_info()
    
    return ModelInfoResponse(
        model_loaded=info['model_loaded'],
        preprocessor_loaded=info['preprocessor_loaded'],
        model_parameters=info['model_parameters'],
        model_layers=info['model_layers'],
        vocabulary_size=info.get('vocab_size', 0),
        num_subjects=info.get('num_subjects', 0),
        subject_classes=info.get('subject_classes', [])
    )


# Single prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict_single(text_input: TextInput, api_key: str = Depends(get_api_key)):
    """
    Predict if a single text is fake or real news.
    
    - **text**: The news text to classify (10-10000 characters)
    
    Returns the prediction with confidence scores and category classification.
    """
    if not model_loaded or not predictor:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please check the server status."
        )
    
    try:
        start_time = time.time()
        result = predictor.predict_single(text_input.text)
        processing_time = time.time() - start_time
        
        return PredictionResponse(
            text=result.text,
            prediction=result.prediction,
            confidence=result.confidence,
            category=result.category,
            category_confidence=result.category_confidence,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


# Batch prediction endpoint
@app.post("/predict-batch", response_model=BatchPredictionResponse)
async def predict_batch(batch_input: BatchTextInput, api_key: str = Depends(get_api_key)):
    """
    Predict if multiple texts are fake or real news.
    
    - **texts**: List of news texts to classify (1-100 texts)
    
    Returns predictions with confidence scores and category classifications for all texts.
    """
    if not model_loaded or not predictor:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please check the server status."
        )
    
    try:
        start_time = time.time()
        results = predictor.predict_batch(batch_input.texts)
        total_time = time.time() - start_time
        
        # Calculate statistics
        predictions = [r.prediction for r in results]
        confidences = [r.confidence for r in results]
        
        fake_count = predictions.count('FAKE')
        true_count = predictions.count('TRUE')
        average_confidence = sum(confidences) / len(confidences)
        
        # Convert to response format
        response_results = []
        for result in results:
            response_results.append(PredictionResponse(
                text=result.text,
                prediction=result.prediction,
                confidence=result.confidence,
                category=result.category,
                category_confidence=result.category_confidence,
                processing_time=result.processing_time
            ))
        
        return BatchPredictionResponse(
            results=response_results,
            total_texts=len(results),
            fake_count=fake_count,
            true_count=true_count,
            average_confidence=average_confidence,
            processing_time=total_time
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


# Analysis endpoint
@app.post("/analyze")
async def analyze_texts(batch_input: BatchTextInput, api_key: str = Depends(get_api_key)):
    """
    Analyze a batch of texts and provide detailed statistics.
    
    - **texts**: List of news texts to analyze (1-100 texts)
    
    Returns comprehensive analysis including predictions, confidence distributions,
    and category breakdowns.
    """
    if not model_loaded or not predictor:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please check the server status."
        )
    
    try:
        stats = predictor.analyze_predictions(batch_input.texts)
        return {
            "analysis": stats,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}"
        )


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Fake News Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "model_info": "/model-info",
        "endpoints": {
            "predict": "/predict - Single text prediction",
            "predict_batch": "/predict-batch - Batch text prediction",
            "analyze": "/analyze - Text analysis",
            "health": "/health - Health check",
            "model_info": "/model-info - Model information"
        }
    }


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return {
        "error": exc.status_code,
        "message": exc.detail,
        "timestamp": time.time()
    }


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return {
        "error": 500,
        "message": "Internal server error",
        "timestamp": time.time()
    }


# Background task for model reloading
async def reload_model():
    """Background task to reload the model."""
    global predictor, model_loaded
    
    model_path = os.environ.get('MODEL_PATH')
    if not model_path:
        return
    
    try:
        # Unload current model
        predictor = None
        model_loaded = False
        
        # Load new model
        preprocessor_path = model_path.replace('.h5', '_preprocessor.pkl')
        predictor = FakeNewsPredictor(model_path, preprocessor_path)
        model_loaded = True
        
        logger.info("Model reloaded successfully")
        
    except Exception as e:
        logger.error(f"Error reloading model: {str(e)}")
        model_loaded = False


# Model reload endpoint (for admin use)
@app.post("/reload-model")
async def reload_model_endpoint(background_tasks: BackgroundTasks):
    """
    Reload the model (admin endpoint).
    
    This endpoint triggers a background task to reload the model.
    Useful for updating the model without restarting the server.
    """
    background_tasks.add_task(reload_model)
    
    return {
        "message": "Model reload initiated",
        "status": "processing"
    }


# Configuration endpoint
@app.get("/config")
async def get_config():
    """Get current API configuration."""
    return {
        "model_path": os.environ.get('MODEL_PATH', 'Not set'),
        "preprocessor_path": os.environ.get('PREPROCESSOR_PATH', 'Not set'),
        "model_loaded": model_loaded,
        "max_text_length": 10000,
        "max_batch_size": 100
    }


# Performance monitoring endpoint
@app.get("/performance")
async def get_performance_stats():
    """Get basic performance statistics."""
    if not model_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    # Simple performance test
    test_text = "This is a test news article for performance monitoring."
    
    try:
        start_time = time.time()
        result = predictor.predict_single(test_text)
        response_time = time.time() - start_time
        
        return {
            "single_prediction_time": response_time,
            "model_loaded": model_loaded,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Performance test error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Performance test failed"
        )


if __name__ == "__main__":
    # Run the API server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
