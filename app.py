
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io
import numpy as np
from typing import Dict, Any
import logging
import traceback

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Cat vs Dog Classifier API",
    description="A CNN-based image classifier for cats and dogs",
    version="1.0.0"
)

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
    """Health check endpoint"""
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
        
        return {
            "status": "healthy",
            "model_status": model_status,
            "device_info": device_info,
            "class_names": class_names,
            "version": "1.0.0"
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
async def predict(file: UploadFile = File(...)):
    """Prediction endpoint that accepts image uploads"""
    
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
        result = predict_image(image)
        
        # Add metadata
        result.update({
            "filename": file.filename,
            "image_size": image.size,
            "image_mode": image.mode
        })
        
        logger.info(f"Prediction made for {file.filename}: {result['predicted_class']} ({result['confidence']:.3f})")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing image {file.filename}: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Cat vs Dog Classifier API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST with image file)",
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
