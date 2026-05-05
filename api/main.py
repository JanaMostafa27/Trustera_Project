"""
FastAPI Service for Fake News Detection
"""

import os
import sys
import time
from typing import List, Dict, Any, Optional
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

from fastapi import FastAPI, HTTPException, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import uvicorn
import logging

from src.predict import FakeNewsPredictor, PredictionResult

print("APP STARTING...")
print(sys.version)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Fake News Detection API",
    description="Simple API for detecting fake news",
    version="1.0.0",
    docs_url="/docs"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
predictor = None
model_loaded = False

# Request model
class TextInput(BaseModel):
    text: str

# Response model
class PredictionResponse(BaseModel):
    prediction: str
    category: str

# Health check
@app.get("/health")
async def health():
    return {
        "status": "healthy" if model_loaded else "unhealthy",
        "model_loaded": model_loaded
    }

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Fake News Detection API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict"
        }
    }

# Load model on startup
@app.on_event("startup")
async def startup():
    global predictor, model_loaded
    
    model_path = os.environ.get('MODEL_PATH', 'models/fake_news_model.pt')
    
    if not os.path.exists(model_path):
        logger.error(f"Model not found: {model_path}")
        return
    

    try:
        preprocessor_path = model_path.replace('.pt', '_preprocessor.pkl')

        predictor = FakeNewsPredictor(model_path, preprocessor_path)
        model_loaded = True

        logger.info("Model loaded successfully")

    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        model_loaded = False


# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict(text_input: TextInput):
    if not model_loaded or not predictor:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    try:
        result = predictor.predict_single(text_input.text)
        
        return PredictionResponse(
            prediction=result.prediction,
            category=result.category
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


# ================= RUN =================
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000
    )
