import os
import io
import pickle
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import List
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image

# Paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "disaster_match_modelqwe.keras"
LABEL_ENCODER_PATH = BASE_DIR / "label_encoder.pkl"

# Initialize FastAPI
app = FastAPI(
    title="Disaster Verification API",
    description="Predict if an image matches a declared disaster type.",
)

# Check model file
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

# Check label encoder file
if not LABEL_ENCODER_PATH.exists():
    raise FileNotFoundError(f"Label encoder file not found at: {LABEL_ENCODER_PATH}")

# Load model
try:
    model = tf.keras.models.load_model(str(MODEL_PATH))
    print("✅ Model loaded successfully.")
except Exception as e:
    raise RuntimeError(f"Error loading model: {e}")

# Load label encoder
try:
    with open(LABEL_ENCODER_PATH, "rb") as f:
        label_encoder = pickle.load(f)
    print("✅ Label encoder loaded successfully.")
except Exception as e:
    raise RuntimeError(f"Error loading label encoder: {e}")

# Classes
valid_labels = list(label_encoder.classes_)
num_classes = len(valid_labels)

# Image Preprocessing
def preprocess_image(image: Image.Image, target_size=(224, 224)) -> np.ndarray:
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0).astype(np.float32)

# Label Encoding
def encode_label(declared_label: str) -> np.ndarray:
    if declared_label not in valid_labels:
        raise ValueError(f"Invalid declared label: {declared_label}. Must be one of {valid_labels}")
    label_idx = label_encoder.transform([declared_label])[0]
    one_hot = np.zeros(num_classes, dtype=np.float32)
    one_hot[label_idx] = 1
    return np.expand_dims(one_hot, axis=0)

# Root endpoint
@app.get("/")
async def root():
    return {"message": "✅ Disaster Verification API is running", "valid_labels": valid_labels}

# Predict endpoint
@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    declared_label: str = Form(...)
):
    try:
        # Validate image file
        if not image.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')):
            raise HTTPException(status_code=400, detail="Unsupported image format.")

        # Read and preprocess image
        contents = await image.read()
        image_pil = Image.open(io.BytesIO(contents))
        image_array = preprocess_image(image_pil)

        # Encode declared label
        label_array = encode_label(declared_label)

        # Predict
        prediction = model.predict([image_array, label_array])[0][0]
        result = "Match" if prediction > 0.5 else "No Match"

        return JSONResponse(content={
            "declared_label": declared_label,
            "prediction_score": float(prediction),
            "result": result,
            "valid_labels": valid_labels
        })

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Error: {str(e)}")

# Optional: Run local test
def test_local_prediction(image_path: str, declared_label: str):
    try:
        with open(image_path, 'rb') as f:
            image_pil = Image.open(f)
        img_array = preprocess_image(image_pil)
        label_array = encode_label(declared_label)
        prediction = model.predict([img_array, label_array])[0][0]
        result = "Match" if prediction > 0.5 else "No Match"
        print({
            "declared_label": declared_label,
            "prediction": float(prediction),
            "result": result
        })
    except Exception as e:
        print(f"Local test failed: {e}")

if __name__ == "__main__":
    # Run local test (only when executing directly)
    test_image_path = BASE_DIR / "model_files" / "test_image.jpg"
    test_declared_label = "Flood"
    if test_image_path.exists():
        test_local_prediction(str(test_image_path), test_declared_label)
    else:
        print("Local test image not found.")
    
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
