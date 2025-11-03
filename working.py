from fastapi import FastAPI, File, UploadFile, HTTPException, status, Depends
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont, ExifTags
from datetime import datetime
from typing import Optional, Tuple, Dict
from contextlib import asynccontextmanager
import requests
import shutil
import os
import io
import json
import logging
import hashlib
import numpy as np

# Keras/TensorFlow imports for your ResNet50 model
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image

# Import custom modules
from utils3 import (
    extract_location_from_image,
    get_weather_for_timestamp,
    validate_coordinates,
    get_location_name,
    generate_weather_description
)
from database import (
    init_database,
    save_weather_data,
    get_latest_weather,
    get_weather_history,
    init_connection_pool,
    close_connection_pool
)
from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== WEATHER AI MODEL ====================
# Global variable for your ResNet50 model
WEATHER_AI_MODEL = None
CLASS_NAMES = ["cloudy", "foggy", "rainy", "sunny", "snowy"]

def load_weather_ai_model(model_path: str = "weather.h5"):
    """Load your fine-tuned ResNet50 weather model"""
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"⚠️ {model_path} model file not found in the current directory.")
        
        model = load_model(model_path)
        logger.info(f"✅ Weather AI model loaded successfully from {model_path}")
        logger.info(f"📋 Weather classes: {CLASS_NAMES}")
        return model
    except Exception as e:
        logger.error(f"❌ Error loading weather AI model: {e}")
        return None

def predict_weather_from_image(image: Image.Image) -> Dict:
    """
    Predict weather condition from image using your trained ResNet50 model
    
    Args:
        image: PIL Image
        
    Returns:
        Dictionary with prediction results
    """
    if WEATHER_AI_MODEL is None:
        return {
            "error": "Weather AI model not loaded",
            "predicted_weather": "Unknown",
            "confidence": 0.0
        }
    
    try:
        # Preprocess image
        img = image.convert("RGB")
        img = img.resize((224, 224))
        img_array = keras_image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        
        # Make prediction
        predictions = WEATHER_AI_MODEL.predict(img_array, verbose=0)
        predicted_class_idx = np.argmax(predictions)
        predicted_class = CLASS_NAMES[predicted_class_idx]
        confidence = float(np.max(predictions))
        
        # Get top 3 predictions
        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        top_3_predictions = [
            {
                "class": CLASS_NAMES[idx],
                "confidence": round(float(predictions[0][idx]), 4)
            }
            for idx in top_3_indices
        ]
        
        logger.info(f"🤖 AI predicted: {predicted_class} ({confidence*100:.2f}% confidence)")
        
        return {
            "predicted_weather": predicted_class.capitalize(),
            "confidence": round(confidence, 4),
            "confidence_percentage": round(confidence * 100, 2),
            "top_3_predictions": top_3_predictions
        }
        
    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        return {
            "error": str(e),
            "predicted_weather": "Unknown",
            "confidence": 0.0
        }

def compare_weather_predictions(api_weather: Dict, ai_prediction: Dict) -> Dict:
    """Compare API weather data with AI prediction"""
    api_condition = api_weather.get('weather_condition', 'Unknown').lower()
    ai_predicted = ai_prediction.get('predicted_weather', 'Unknown').lower()
    ai_confidence = ai_prediction.get('confidence', 0.0)
    
    # Simple matching logic
    match = False
    analysis = ""
    
    if api_condition == ai_predicted:
        match = True
        analysis = f"Perfect match! Both API and AI agree on '{ai_predicted}'"
    elif any(word in api_condition for word in ai_predicted.split()) or \
         any(word in ai_predicted for word in api_condition.split()):
        match = True
        analysis = f"Close match: API reports '{api_condition}' and AI predicts '{ai_predicted}'"
    else:
        match = False
        analysis = f"Different predictions: API says '{api_condition}' but AI predicts '{ai_predicted}' ({ai_confidence*100:.1f}% confident)"
    
    return {
        "api_condition": api_condition,
        "ai_prediction": ai_predicted,
        "ai_confidence": ai_confidence,
        "match": match,
        "analysis": analysis
    }

# ==================== LIFESPAN ====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global WEATHER_AI_MODEL
    
    logger.info("🚀 Starting Vehicle Detection & Weather API...")
    
    # Create directories
    os.makedirs(config.UPLOAD_DIR, exist_ok=True)
    IMAGES_DIR.mkdir(exist_ok=True)
    ANNOTATED_DIR.mkdir(exist_ok=True)
    
    # Initialize databases
    init_connection_pool()
    init_database()
    
    # Load your ResNet50 weather AI model
    WEATHER_AI_MODEL = load_weather_ai_model("weather.h5")
    
    logger.info("✅ Application started successfully!")
    yield
    
    logger.info("🛑 Shutting down API...")
    close_connection_pool()
    logger.info("✅ Application shutdown complete!")

# ==================== FASTAPI APP ====================
app = FastAPI(
    title="Vehicle Detection & Weather Analysis API",
    description="Upload an image once for vehicle detection, people/PPE detection, weather analysis, and AI weather prediction",
    version="5.0.0",
    lifespan=lifespan
)

# ==================== DIRECTORIES ====================
IMAGES_DIR = Path("uploaded_images")
ANNOTATED_DIR = Path("images_annotated")
IMAGES_DIR.mkdir(exist_ok=True)
ANNOTATED_DIR.mkdir(exist_ok=True)

# Mount static files
app.mount("/uploaded_images", StaticFiles(directory=IMAGES_DIR), name="uploaded_images")
app.mount("/images_annotated", StaticFiles(directory=ANNOTATED_DIR), name="images_annotated")

# ==================== MODELS ====================
try:
    PPE_MODEL = YOLO("PPE.pt")
    logger.info("✅ PPE model loaded")
except Exception as e:
    logger.warning(f"⚠️ PPE model not found: {e}")
    PPE_MODEL = None

try:
    VEHICLE_MODEL = YOLO("vehicle.pt")
    logger.info("✅ Vehicle model loaded")
except Exception as e:
    logger.error(f"❌ Vehicle model not found: {e}")
    VEHICLE_MODEL = None

# ==================== CONSTANTS ====================
VEHICLE_TYPES = [
    "ambulance", "car", "bus", "motorbike", "truck",
    "ricksaw", "threewheel", "cng", "bicycle", "motorcycle", "otherVehicles"
]

PPE_CLASSES = {
    0: "helmet", 1: "gloves", 2: "vest", 3: "boots", 4: "goggles",
    5: "none", 6: "person", 7: "no_helmet", 8: "no_goggle",
    9: "no_gloves", 10: "no_boots"
}

CLASS_COLORS = {
    "person": (255, 0, 0), "helmet": (0, 255, 0), "gloves": (255, 255, 0),
    "vest": (0, 255, 255), "boots": (255, 165, 0), "goggles": (128, 0, 128),
    "no_helmet": (255, 20, 147), "no_goggle": (255, 105, 180),
    "no_gloves": (255, 69, 0), "no_boots": (255, 0, 255),
    "car": (0, 128, 255), "bus": (255, 128, 0), "truck": (255, 255, 0),
    "motorbike": (0, 255, 128), "ambulance": (255, 0, 128),
    "bicycle": (0, 200, 255), "ricksaw": (200, 100, 255),
    "threewheel": (255, 180, 0)
}

# ==================== GLOBAL STATE ====================
latest_image_path: Optional[Path] = None
latest_uploaded_image_id: Optional[int] = None
cached_weather_data: Optional[Dict] = None
cached_vehicle_detections: Optional[list] = None
cached_people_detections: Optional[list] = None
cached_gps: Optional[Tuple] = None
cached_photo_timestamp: Optional[datetime] = None
cached_location_name: Optional[str] = None
cached_ai_weather_prediction: Optional[Dict] = None  # NEW: AI prediction cache

# ==================== HELPER FUNCTIONS ====================
def calculate_file_hash(file_path: str) -> str:
    """Calculate MD5 hash of a file"""
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()

def detect_vehicles(image: Image.Image) -> list:
    """Detect all vehicles in the image"""
    if VEHICLE_MODEL is None:
        logger.warning("Vehicle model not loaded")
        return []
    
    results = VEHICLE_MODEL.predict(image, conf=0.3)
    detections = []
    
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0].item())
            cls_name = r.names[cls_id].lower()
            conf = float(box.conf[0].item())
            xyxy = box.xyxy[0].tolist()
            
            detections.append({
                "class": cls_name,
                "confidence": round(conf, 3),
                "bbox": {
                    "x1": round(xyxy[0], 2),
                    "y1": round(xyxy[1], 2),
                    "x2": round(xyxy[2], 2),
                    "y2": round(xyxy[3], 2)
                }
            })
    
    return detections

def detect_people_and_ppe(image: Image.Image) -> list:
    """Detect people and PPE in the image"""
    if PPE_MODEL is None:
        logger.warning("PPE model not loaded")
        return []
    
    results = PPE_MODEL.predict(image, conf=0.3)
    detections = []
    
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0].item())
            cls_name = PPE_CLASSES.get(cls_id, "unknown")
            conf = float(box.conf[0].item())
            xyxy = box.xyxy[0].tolist()
            
            detections.append({
                "class": cls_name,
                "confidence": round(conf, 3),
                "bbox": {
                    "x1": round(xyxy[0], 2),
                    "y1": round(xyxy[1], 2),
                    "x2": round(xyxy[2], 2),
                    "y2": round(xyxy[3], 2)
                }
            })
    
    return detections

def create_annotated_image(image_path: Path, vehicle_dets: list, people_dets: list) -> Optional[Path]:
    """Create annotated image with bounding boxes"""
    try:
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        
        # Try to load a font, fallback to default if not available
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        # Draw vehicle detections
        for det in vehicle_dets:
            bbox = det['bbox']
            color = CLASS_COLORS.get(det['class'], (255, 255, 255))
            draw.rectangle([bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']], outline=color, width=3)
            label = f"{det['class']} {det['confidence']:.2f}"
            draw.text((bbox['x1'], max(0, bbox['y1'] - 20)), label, fill=color, font=font)
        
        # Draw people detections
        for det in people_dets:
            bbox = det['bbox']
            color = CLASS_COLORS.get(det['class'], (255, 255, 255))
            draw.rectangle([bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']], outline=color, width=3)
            label = f"{det['class']} {det['confidence']:.2f}"
            draw.text((bbox['x1'], max(0, bbox['y1'] - 20)), label, fill=color, font=font)
        
        # Save annotated image
        annotated_path = ANNOTATED_DIR / f"annotated_{image_path.name}"
        img.save(annotated_path)
        logger.info(f"✅ Annotated image created: {annotated_path}")
        return annotated_path
    except Exception as e:
        logger.error(f"Error creating annotated image: {e}")
        return None

# ==================== MAIN UPLOAD ENDPOINT ====================
@app.post("/upload-image/", status_code=status.HTTP_201_CREATED)
async def upload_image(file: UploadFile = File(...)):
    """
    Upload an image once for:
    - Vehicle detection (11 types)
    - People/PPE detection  
    - Weather analysis from GPS/timestamp
    - AI weather prediction from image
    All processed simultaneously!
    """
    global latest_image_path, latest_uploaded_image_id, cached_weather_data
    global cached_vehicle_detections, cached_people_detections
    global cached_gps, cached_photo_timestamp, cached_location_name
    global cached_ai_weather_prediction
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are allowed")
    
    try:
        # Save file
        file_path = IMAGES_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        latest_image_path = file_path
        file_hash = calculate_file_hash(str(file_path))
        
        logger.info(f"📸 Image uploaded: {file.filename}")
        
        # Load image for processing
        image = Image.open(file_path).convert("RGB")
        
        # ========== AI WEATHER PREDICTION (NEW!) ==========
        ai_weather_prediction = predict_weather_from_image(image)
        cached_ai_weather_prediction = ai_weather_prediction
        
        # ========== GPS & TIMESTAMP EXTRACTION ==========
        lat, lon, photo_timestamp = extract_location_from_image(str(file_path))
        
        if not validate_coordinates(lat, lon):
            raise HTTPException(status_code=400, detail="Invalid GPS coordinates extracted from image")
        
        location_name = get_location_name(lat, lon)
        logger.info(f"📍 Location: {location_name} ({lat}, {lon})")
        
        # ========== FETCH API WEATHER DATA ==========
        weather = get_weather_for_timestamp(lat, lon, photo_timestamp)
        if not weather:
            raise HTTPException(status_code=503, detail="Unable to fetch weather data from APIs")
        
        description = generate_weather_description(weather, location_name, photo_timestamp)
        weather['description'] = description
        
        # Add AI prediction to weather data
        weather['ai_prediction'] = ai_weather_prediction
        
        # Cache weather data
        cached_weather_data = weather
        cached_gps = (lat, lon)
        cached_photo_timestamp = photo_timestamp
        cached_location_name = location_name
        
        # ========== COMPARE API VS AI PREDICTIONS ==========
        weather_comparison = compare_weather_predictions(weather, ai_weather_prediction)
        logger.info(f"📊 {weather_comparison['analysis']}")
        
        # ========== SAVE TO DATABASE ==========
        weather_result = save_weather_data(
            weather_data=weather,
            gps_data={
                "latitude": lat,
                "longitude": lon,
                "location_name": location_name,
                "photo_timestamp": photo_timestamp
            },
            image_filename=file.filename
        )
        
        logger.info(f"✅ Weather data saved to DB (ID: {weather_result['id']})")
        
        # ========== VEHICLE DETECTION ==========
        vehicle_detections = detect_vehicles(image)
        cached_vehicle_detections = vehicle_detections
        logger.info(f"🚗 Detected {len(vehicle_detections)} vehicles")
        
        # ========== PEOPLE/PPE DETECTION ==========
        people_detections = detect_people_and_ppe(image)
        cached_people_detections = people_detections
        logger.info(f"👷 Detected {len(people_detections)} people/PPE items")
        
        # ========== CREATE ANNOTATED IMAGE ==========
        annotated_path = create_annotated_image(file_path, vehicle_detections, people_detections)
        
        # ========== COUNT DETECTIONS ==========
        vehicle_counts = {}
        for det in vehicle_detections:
            vtype = det['class']
            vehicle_counts[vtype] = vehicle_counts.get(vtype, 0) + 1
        
        people_count = sum(1 for d in people_detections if d['class'] == 'person')
        ppe_counts = {}
        for det in people_detections:
            if det['class'] != 'person':
                ppe_counts[det['class']] = ppe_counts.get(det['class'], 0) + 1
        
        # ========== BUILD RESPONSE ==========
        return {
            "message": "✅ Image processed successfully!",
            "file_info": {
                "filename": file.filename,
                "path": str(file_path),
                "hash": file_hash
            },
            "location": {
                "latitude": lat,
                "longitude": lon,
                "name": location_name,
                "photo_taken_at": photo_timestamp.isoformat()
            },
            "weather": {
                "api_data": {
                    "temperature": f"{weather['temperature']}°C",
                    "feels_like": f"{weather['feels_like']}°C",
                    "condition": weather['weather_condition'],
                    "humidity": f"{weather['humidity']}%",
                    "wind_speed": f"{weather['windspeed']} km/h",
                    "pressure": f"{weather.get('pressure', 'N/A')} hPa",
                    "description": description
                },
                "ai_prediction": ai_weather_prediction,
                "comparison": weather_comparison,
                "record_id": weather_result['id']
            },
            "detections": {
                "vehicles": {
                    "total": len(vehicle_detections),
                    "by_type": vehicle_counts,
                    "details": vehicle_detections
                },
                "people": {
                    "total_people": people_count,
                    "total_detections": len(people_detections),
                    "ppe_items": ppe_counts,
                    "details": people_detections
                }
            },
            "annotated_image": f"/images_annotated/annotated_{file.filename}" if annotated_path else None,
            "data_sources": weather.get('data_source', [])
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error processing upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== DETECTION ENDPOINTS ====================
def ensure_image_uploaded():
    """Check if an image has been uploaded"""
    if latest_image_path is None or not latest_image_path.exists():
        raise HTTPException(
            status_code=404,
            detail="No image uploaded yet. Please upload an image first using POST /upload-image/"
        )

def detect_specific_type(image: Image.Image, vtype: str):
    """Detect specific vehicle type"""
    if VEHICLE_MODEL is None:
        return []
    
    results = VEHICLE_MODEL.predict(image, conf=0.3)
    detections = []
    
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0].item())
            cls_name = r.names[cls_id].lower()
            conf = float(box.conf[0].item())
            xyxy = box.xyxy[0].tolist()
            
            if cls_name == vtype.lower():
                detections.append({
                    "class": cls_name,
                    "confidence": round(conf, 3),
                    "bbox": {
                        "x1": round(xyxy[0], 2),
                        "y1": round(xyxy[1], 2),
                        "x2": round(xyxy[2], 2),
                        "y2": round(xyxy[3], 2)
                    }
                })
    
    return detections

# Create dynamic endpoints for each vehicle type
def create_vehicle_endpoint(vtype: str):
    async def detect_vehicle_type():
        ensure_image_uploaded()
        
        if cached_vehicle_detections is None:
            image = Image.open(latest_image_path).convert("RGB")
            detections = detect_specific_type(image, vtype)
        else:
            # Filter cached detections
            detections = [d for d in cached_vehicle_detections if d['class'].lower() == vtype.lower()]
        
        return JSONResponse({
            "vehicle_type": vtype,
            "total_detected": len(detections),
            "detections": detections,
            "image": latest_image_path.name
        })
    return detect_vehicle_type

# Register all vehicle detection endpoints
for vtype in VEHICLE_TYPES:
    route_path = f"/detect/{vtype.lower()}/"
    endpoint_func = create_vehicle_endpoint(vtype)
    app.get(route_path, name=f"Detect {vtype}")(endpoint_func)

@app.get("/detect-all/")
async def detect_all():
    """Get all detections (vehicles + people) from last uploaded image"""
    ensure_image_uploaded()
    
    vehicle_counts = {}
    for det in (cached_vehicle_detections or []):
        vtype = det['class']
        vehicle_counts[vtype] = vehicle_counts.get(vtype, 0) + 1
    
    people_count = sum(1 for d in (cached_people_detections or []) if d['class'] == 'person')
    
    return {
        "image": latest_image_path.name,
        "vehicles": {
            "total": len(cached_vehicle_detections) if cached_vehicle_detections else 0,
            "by_type": vehicle_counts,
            "detections": cached_vehicle_detections or []
        },
        "people": {
            "total_people": people_count,
            "total_detections": len(cached_people_detections) if cached_people_detections else 0,
            "detections": cached_people_detections or []
        }
    }

@app.get("/detect-people/")
async def detect_people_only():
    """Get only people detections from last uploaded image"""
    ensure_image_uploaded()
    
    people = [d for d in (cached_people_detections or []) if d['class'] == 'person']
    
    return {
        "image": latest_image_path.name,
        "total_people": len(people),
        "detections": people
    }

# ==================== WEATHER ENDPOINTS ====================
def ensure_weather_data():
    """Check if weather data exists"""
    if cached_weather_data is None:
        raise HTTPException(
            status_code=404,
            detail="No weather data found. Please upload an image first using POST /upload-image/"
        )

@app.get("/get-temperature/")
async def get_temperature():
    """Get temperature data with human-readable description"""
    ensure_weather_data()
    
    weather = cached_weather_data
    temp = weather['temperature']
    feels = weather.get('feels_like')
    
    if feels and temp:
        diff = feels - temp
        if abs(diff) < 2:
            desc = f"The temperature is {temp}°C and it feels about the same."
        elif diff > 0:
            desc = f"The temperature is {temp}°C, but it feels warmer at {feels}°C due to humidity."
        else:
            desc = f"The temperature is {temp}°C, but it feels cooler at {feels}°C due to wind."
    else:
        desc = f"The temperature is {temp}°C."
    
    return {
        "temperature": f"{temp}°C",
        "feels_like": f"{feels}°C" if feels else None,
        "description": desc,
        "photo_taken_at": cached_photo_timestamp.isoformat() if cached_photo_timestamp else None
    }

@app.get("/get-humidity/")
async def get_humidity():
    """Get humidity data with human-readable description"""
    ensure_weather_data()
    
    weather = cached_weather_data
    humidity = weather.get('humidity')
    
    if humidity:
        if humidity < 30:
            desc = f"The humidity is {humidity}%, which is quite dry."
        elif humidity < 60:
            desc = f"The humidity is {humidity}%, which is comfortable."
        elif humidity < 80:
            desc = f"The humidity is {humidity}%, which is moderately humid."
        else:
            desc = f"The humidity is {humidity}%, which is very humid and muggy."
    else:
        desc = "Humidity data not available."
    
    return {
        "humidity": f"{humidity}%" if humidity else "N/A",
        "description": desc,
        "photo_taken_at": cached_photo_timestamp.isoformat() if cached_photo_timestamp else None
    }

@app.get("/get-windspeed/")
async def get_windspeed():
    """Get wind speed data with human-readable description"""
    ensure_weather_data()
    
    weather = cached_weather_data
    wind = weather.get('wind_speed')
    
    if wind:
        if wind < 5:
            desc = f"The wind speed is {wind} km/h, which is calm."
        elif wind < 15:
            desc = f"The wind speed is {wind} km/h, creating a gentle breeze."
        elif wind < 30:
            desc = f"The wind speed is {wind} km/h, with moderate winds."
        else:
            desc = f"The wind speed is {wind} km/h, with strong winds."
    else:
        desc = "Wind speed data not available."
    
    return {
        "wind_speed": f"{wind} km/h" if wind else "N/A",
        "description": desc,
        "photo_taken_at": cached_photo_timestamp.isoformat() if cached_photo_timestamp else None
    }

@app.get("/get-weather-condition/")
async def get_condition():
    """Get weather condition with human-readable description"""
    ensure_weather_data()
    
    weather = cached_weather_data
    condition = weather.get('weather_condition', 'Unknown')
    clouds = weather.get('cloud_coverage')
    
    if clouds is not None:
        desc = f"The weather condition is {condition.lower()} with {clouds}% cloud coverage. "
        if clouds < 20:
            desc += "The sky is mostly clear."
        elif clouds < 50:
            desc += "There are some clouds but plenty of sunshine."
        elif clouds < 80:
            desc += "The sky is quite cloudy with limited sunshine."
        else:
            desc += "The sky is heavily overcast."
    else:
        desc = f"The weather condition is {condition.lower()}."
    
    return {
        "condition": condition,
        "cloud_coverage": f"{clouds}%" if clouds is not None else "N/A",
        "description": desc,
        "photo_taken_at": cached_photo_timestamp.isoformat() if cached_photo_timestamp else None
    }

@app.get("/get-pressure/")
async def get_pressure():
    """Get atmospheric pressure with human-readable description"""
    ensure_weather_data()
    
    weather = cached_weather_data
    pressure = weather.get('pressure')
    
    if pressure:
        if pressure < 1000:
            desc = f"The atmospheric pressure is {pressure} hPa, which is low."
        elif pressure < 1013:
            desc = f"The atmospheric pressure is {pressure} hPa, slightly below normal."
        elif pressure < 1020:
            desc = f"The atmospheric pressure is {pressure} hPa, normal."
        else:
            desc = f"The atmospheric pressure is {pressure} hPa, high, usually clear weather."
    else:
        desc = "Pressure data not available."
    
    return {
        "pressure": f"{pressure} hPa" if pressure else "N/A",
        "description": desc,
        "photo_taken_at": cached_photo_timestamp.isoformat() if cached_photo_timestamp else None
    }

@app.get("/get-ai-weather-prediction/")
async def get_ai_weather():
    """Get AI-based weather prediction from the uploaded image (NEW!)"""
    ensure_image_uploaded()
    
    if cached_ai_weather_prediction is None:
        return {
            "status": "not_available",
            "message": "AI weather prediction not available. Model may not be loaded."
        }
    
    return {
        "image": latest_image_path.name,
        "ai_prediction": cached_ai_weather_prediction,
        "photo_taken_at": cached_photo_timestamp.isoformat() if cached_photo_timestamp else None
    }

@app.get("/compare-weather/")
async def compare_weather():
    """Compare API weather data with AI prediction (NEW!)"""
    ensure_weather_data()
    
    if cached_ai_weather_prediction is None:
        raise HTTPException(
            status_code=404,
            detail="AI weather prediction not available"
        )
    
    comparison = compare_weather_predictions(cached_weather_data, cached_ai_weather_prediction)
    
    return {
        "image": latest_image_path.name if latest_image_path else None,
        "location": cached_location_name,
        "api_weather": {
            "condition": cached_weather_data.get('weather_condition'),
            "temperature": cached_weather_data.get('temperature'),
            "sources": cached_weather_data.get('data_source', [])
        },
        "ai_prediction": cached_ai_weather_prediction,
        "comparison": comparison,
        "photo_taken_at": cached_photo_timestamp.isoformat() if cached_photo_timestamp else None
    }

@app.get("/get-all-weather/")
async def get_all_weather():
    """Get complete weather data with AI prediction"""
    ensure_weather_data()
    
    weather = cached_weather_data
    
    return {
        "location": {
            "latitude": cached_gps[0],
            "longitude": cached_gps[1],
            "name": cached_location_name
        },
        "weather": {
            "condition": weather.get('weather_condition', "Unknown"),
            "temperature": f"{weather.get('temperature', 'N/A')}°C",
            "feels_like": f"{weather.get('feels_like', 'N/A')}°C",
            "humidity": f"{weather.get('humidity', 'N/A')}%",
            "wind_speed": f"{weather.get('wind_speed', 'N/A')} km/h",
            "pressure": f"{weather.get('pressure', 'N/A')} hPa",
            "cloud_coverage": f"{weather.get('cloud_coverage', 'N/A')}%"
        },
        "ai_prediction": cached_ai_weather_prediction if cached_ai_weather_prediction else {
            "status": "not_available"
        },
        "description": weather.get('description', 'No description available'),
        "photo_taken_at": cached_photo_timestamp.isoformat() if cached_photo_timestamp else None,
        "data_sources": weather.get('data_source', []),
        "image": latest_image_path.name if latest_image_path else None
    }

@app.get("/weather-history/")
async def get_history(hours: int = 24):
    """Get weather history for the last N hours from database"""
    try:
        history = get_weather_history(hours)
        return {
            "count": len(history),
            "hours": hours,
            "records": history
        }
    except Exception as e:
        logger.error(f"❌ Error fetching history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== UTILITY ENDPOINTS ====================
@app.get("/health/")
def health_check():
    """API health check endpoint"""
    return {
        "status": "healthy",
        "message": "Vehicle Detection & Weather API is running",
        "version": "5.0.0",
        "models_loaded": {
            "vehicle_model": VEHICLE_MODEL is not None,
            "ppe_model": PPE_MODEL is not None,
            "weather_ai_model": WEATHER_AI_MODEL is not None
        },
        "ai_model_info": {
            "type": "ResNet50 (Keras/TensorFlow)",
            "classes": CLASS_NAMES,
            "num_classes": len(CLASS_NAMES)
        } if WEATHER_AI_MODEL is not None else None,
        "latest_upload": {
            "has_image": latest_image_path is not None,
            "has_weather": cached_weather_data is not None,
            "has_ai_prediction": cached_ai_weather_prediction is not None,
            "vehicle_detections": len(cached_vehicle_detections) if cached_vehicle_detections else 0,
            "people_detections": len(cached_people_detections) if cached_people_detections else 0
        }
    }

@app.get("/")
def root():
    """Root endpoint with API information"""
    return {
        "message": "🚗🌤️👷🤖 Welcome to Vehicle Detection & Weather Analysis API",
        "version": "5.0.0",
        "features": [
            "Vehicle Detection (11 types: car, bus, motorbike, truck, ambulance, etc.)",
            "People & PPE Detection (helmet, gloves, vest, boots, goggles)",
            "Weather Analysis from GPS & Timestamp",
            "AI Weather Prediction from Image (ResNet50)",
            "Weather Comparison (API vs AI)",
            "Annotated Images with Bounding Boxes",
            "Historical Weather Data Tracking"
        ],
        "workflow": {
            "step_1": "POST /upload-image/ - Upload an image (processes everything at once)",
            "step_2": "Use any detection or weather endpoint to get specific data",
            "note": "Image must be uploaded first before using other endpoints"
        },
        "endpoints": {
            "upload": {
                "upload_image": "POST /upload-image/ - Upload image for complete analysis"
            },
            "vehicle_detection": {
                "specific_vehicle": "GET /detect/{vehicle_type}/ - Detect specific vehicle (car, bus, motorbike, etc.)",
                "all_detections": "GET /detect-all/ - Get all vehicle and people detections",
                "people_only": "GET /detect-people/ - Get only people detections"
            },
            "weather": {
                "temperature": "GET /get-temperature/ - Temperature with description",
                "humidity": "GET /get-humidity/ - Humidity with description",
                "windspeed": "GET /get-windspeed/ - Wind speed with description",
                "condition": "GET /get-weather-condition/ - Weather condition with description",
                "pressure": "GET /get-pressure/ - Atmospheric pressure with description",
                "ai_prediction": "GET /get-ai-weather-prediction/ - AI weather prediction (NEW!)",
                "all_weather": "GET /get-all-weather/ - Complete weather data with AI",
                "compare": "GET /compare-weather/ - Compare API vs AI predictions (NEW!)",
                "history": "GET /weather-history/?hours=24 - Weather history"
            },
            "utility": {
                "health": "GET /health/ - API health check",
                "docs": "GET /docs - Interactive API documentation",
                "redoc": "GET /redoc - Alternative API documentation"
            }
        },
        "supported_vehicles": VEHICLE_TYPES,
        "ai_weather_classes": CLASS_NAMES,
        "documentation": "/docs"
    }
