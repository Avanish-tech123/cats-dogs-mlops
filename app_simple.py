import torch
import torch.nn as nn
import torchvision.transforms as transforms
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from PIL import Image
import io
import numpy as np
from typing import Dict, Any, Optional
import logging
import traceback
import time
import json
from collections import defaultdict, deque
from datetime import datetime
import threading
import uuid

# Setup comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Simple metrics storage without middleware
class SimpleMetrics:
    def __init__(self):
        self.request_count = defaultdict(int)
        self.prediction_history = deque(maxlen=100)
        self.start_time = datetime.utcnow()
        self._lock = threading.Lock()
    
    def record_prediction(self, request_id: str, filename: str, prediction: str, 
                         confidence: float, processing_time_ms: float, true_label: Optional[str] = None):
        with self._lock:
            self.request_count['predictions'] += 1
            self.prediction_history.append({
                'request_id': request_id,
                'timestamp': datetime.utcnow().isoformat(),
                'filename': filename,
                'predicted_class': prediction,
                'confidence': confidence,
                'processing_time_ms': processing_time_ms,
                'true_label': true_label
            })
    
    def record_request(self, endpoint: str):
        with self._lock:
            self.request_count[endpoint] += 1
    
    def get_summary(self):
        with self._lock:
            recent_predictions = list(self.prediction_history)
            uptime_seconds = (datetime.utcnow() - self.start_time).total_seconds()
            
            # Calculate metrics from recent predictions
            if recent_predictions:
                avg_confidence = np.mean([p['confidence'] for p in recent_predictions])
                avg_processing_time = np.mean([p['processing_time_ms'] for p in recent_predictions])
                
                # Calculate accuracy if we have labels
                with_labels = [p for p in recent_predictions if p.get('true_label')]
                accuracy = None
                if with_labels:
                    correct = sum(1 for p in with_labels if p['predicted_class'].lower() == p['true_label'].lower())
                    accuracy = correct / len(with_labels)
            else:
                avg_confidence = 0
                avg_processing_time = 0
                accuracy = None
            
            return {
                'uptime_seconds': round(uptime_seconds),
                'total_requests': dict(self.request_count),
                'total_predictions': len(self.prediction_history),
                'avg_confidence': round(avg_confidence, 3) if recent_predictions else 0,
                'avg_processing_time_ms': round(avg_processing_time, 2) if recent_predictions else 0,
                'accuracy': round(accuracy, 3) if accuracy is not None else None,
                'recent_predictions': recent_predictions[-10:]  # Last 10 predictions
            }

# Global metrics
metrics = SimpleMetrics()

# Define model class for loading (same as M1 - compact version)
class CatsDogsCNN(nn.Module):
    def __init__(self):
        super(CatsDogsCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(512 * 7 * 7, 512), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(512, 128), nn.ReLU(inplace=True), nn.Dropout(0.3), nn.Linear(128, 1),
        )
    def forward(self, x):
        return self.classifier(self.features(x)).squeeze(1)

# Global variables for model and preprocessing
model = None
transform = None
class_names = ['Cat', 'Dog']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model():
    """Load the trained model"""
    global model, transform
    
    try:
        # Load the pre-trained model using state dict
        model = CatsDogsCNN()
        state_dict = torch.load('models/cnn_best.pt', map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        logger.info("Loaded model from cnn_best.pt")
        
        # Define preprocessing transform (same as M1 training)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        logger.info("Model loaded successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False

# Initialize FastAPI app
app = FastAPI(
    title="Cat vs Dog Classifier API with Monitoring",
    description="A CNN-based image classifier for cats and dogs with monitoring",
    version="1.1.0"
)

# Load model on startup
@app.on_event("startup")
async def startup_event():
    success = load_model()
    if not success:
        logger.error("Failed to load model on startup!")

def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Preprocess image for model inference"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Apply transforms
    tensor = transform(image)
    # Add batch dimension
    tensor = tensor.unsqueeze(0)
    return tensor.to(device)

def predict_image(image: Image.Image) -> Dict[str, Any]:
    """Make prediction on image"""
    try:
        # Preprocess image
        input_tensor = preprocess_image(image)
        
        # Make prediction - model outputs single logit for binary classification
        with torch.no_grad():
            outputs = model(input_tensor)
            # Apply sigmoid to get probability of Dog (class 1)
            probability_dog = torch.sigmoid(outputs).item()
            probability_cat = 1 - probability_dog  # Probability of Cat (class 0)
            
            # Determine predicted class
            predicted_class = 1 if probability_dog > 0.5 else 0
            confidence = max(probability_dog, probability_cat)
            
        return {
            "predicted_class": class_names[predicted_class],
            "confidence": confidence,
            "probabilities": {
                class_names[0]: probability_cat,
                class_names[1]: probability_dog
            }
        }
        
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/health")
async def health_check():
    """Enhanced health check endpoint with basic metrics"""
    metrics.record_request('health')
    
    try:
        # Check if model is loaded
        model_status = "ready" if model is not None else "not_loaded"
        
        # Check device availability
        device_info = {
            "device": str(device),
            "cuda_available": torch.cuda.is_available()
        }
        
        if torch.cuda.is_available():
            device_info["cuda_device_count"] = torch.cuda.device_count()
            device_info["cuda_current_device"] = torch.cuda.current_device()
        
        # Add basic metrics
        metrics_summary = metrics.get_summary()
        
        return {
            "status": "healthy",
            "model_status": model_status,
            "device_info": device_info,
            "class_names": class_names,
            "version": "1.1.0",
            "uptime_stats": {
                "uptime_seconds": metrics_summary['uptime_seconds'],
                "total_predictions": metrics_summary['total_predictions'],
                "avg_confidence": metrics_summary['avg_confidence']
            }
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy", 
                "error": str(e)
            }
        )

@app.post("/predict")
async def predict(file: UploadFile = File(...), true_label: Optional[str] = None):
    """Prediction endpoint that accepts image uploads with optional true label for performance tracking"""
    
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Check if model is loaded
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Make prediction
        prediction_start = time.time()
        result = predict_image(image)
        processing_time_ms = (time.time() - prediction_start) * 1000
        
        # Add metadata
        result.update({
            "request_id": request_id,
            "filename": file.filename,
            "image_size": image.size,
            "image_mode": image.mode,
            "processing_time_ms": round(processing_time_ms, 2)
        })
        
        # Record prediction for monitoring
        metrics.record_prediction(
            request_id=request_id,
            filename=file.filename or "unknown",
            prediction=result['predicted_class'],
            confidence=result['confidence'],
            processing_time_ms=processing_time_ms,
            true_label=true_label
        )
        
        # Log prediction (excluding image data)
        prediction_log = {
            'request_id': request_id,
            'filename': file.filename,
            'predicted_class': result['predicted_class'],
            'confidence': round(result['confidence'], 3),
            'image_size': image.size,
            'processing_time_ms': round(processing_time_ms, 2),
            'true_label': true_label
        }
        logger.info(f"PREDICTION: {json.dumps(prediction_log)}")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        error_log = {
            'request_id': request_id,
            'filename': file.filename,
            'error': str(e),
            'processing_time_ms': round((time.time() - start_time) * 1000, 2)
        }
        logger.error(f"PREDICTION_ERROR: {json.dumps(error_log)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.get("/metrics")
async def get_metrics():
    """Endpoint to retrieve application metrics"""
    metrics.record_request('metrics')
    
    try:
        summary = metrics.get_summary()
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "status": "active",
            **summary
        }
    except Exception as e:
        logger.error(f"Error retrieving metrics: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving metrics")

@app.get("/performance")
async def get_performance_data():
    """Endpoint to retrieve model performance data for analysis"""
    metrics.record_request('performance')
    
    try:
        summary = metrics.get_summary()
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "total_predictions": summary['total_predictions'],
            "accuracy": summary['accuracy'],
            "performance_summary": {
                "avg_confidence": summary['avg_confidence'],
                "avg_processing_time_ms": summary['avg_processing_time_ms']
            },
            "recent_predictions": summary['recent_predictions']
        }
    except Exception as e:
        logger.error(f"Error retrieving performance data: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving performance data")

@app.get("/")
async def root():
    """Root endpoint with API information including monitoring endpoints"""
    metrics.record_request('root')
    
    return {
        "message": "Cat vs Dog Classifier API with Simple Monitoring",
        "version": "1.1.0",
        "endpoints": {
            "health": "/health - Health check with basic metrics",
            "predict": "/predict - POST with image file (optional true_label param for tracking)",
            "metrics": "/metrics - Application metrics and statistics",
            "performance": "/performance - Model performance data and analysis",
            "docs": "/docs - Interactive API documentation"
        },
        "monitoring_features": [
            "Request counting and tracking",
            "Prediction accuracy tracking (when true labels provided)",
            "Performance metrics collection",
            "Structured logging for all requests"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)