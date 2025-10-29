from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from ultralytics import YOLO
from PIL import Image
import io
import os
import shutil
import logging
from contextlib import asynccontextmanager

import main3  # your vehicle detection utils
from utils import (
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

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting Vehicle & Weather API...")
    os.makedirs(config.UPLOAD_DIR, exist_ok=True)
    IMAGES_DIR.mkdir(exist_ok=True)
    init_connection_pool()
    init_database()
    logger.info("✅ Application started successfully!")
    yield
    logger.info("🛑 Shutting down Vehicle & Weather API...")
    close_connection_pool()
    logger.info("✅ Application shutdown complete!")

app = FastAPI(
    title="Vehicle Detection & Weather API",
    description="Upload an image once for both vehicle detection and weather data extraction.",
    version="3.0.0",
    lifespan=lifespan
)

IMAGES_DIR = Path("uploaded_images")
MODEL_PATH = "best_vehicle1.pt"
model = YOLO(MODEL_PATH)

VEHICLE_TYPES = [
    "ambulance", "car", "bus", "motorbike", "truck",
    "ricksaw", "threewheel", "cng", "bicycle", "motorcycle", "otherVehicles"
]

# Global state to remember last upload info and cache weather data
latest_image_path = None
cached_weather_data = None
cached_gps = None
cached_photo_timestamp = None
cached_location_name = None

app.mount("/uploaded_images", StaticFiles(directory=IMAGES_DIR), name="uploaded_images")

@app.post("/upload-image/", status_code=status.HTTP_201_CREATED)
async def upload_image(file: UploadFile = File(...)):
    global latest_image_path, cached_weather_data, cached_gps, cached_photo_timestamp, cached_location_name
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are allowed")
    
    file_path = IMAGES_DIR / file.filename
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        latest_image_path = file_path
        
        # Extract GPS and timestamp once here and cache
        lat, lon, photo_timestamp = extract_location_from_image(str(file_path))
        if not validate_coordinates(lat, lon):
            raise HTTPException(status_code=400, detail="Invalid GPS coordinates extracted")
        location_name = get_location_name(lat, lon)
        
        weather = get_weather_for_timestamp(lat, lon, photo_timestamp)
        if not weather:
            raise HTTPException(status_code=503, detail="Unable to fetch weather data from APIs")
        description = generate_weather_description(weather, location_name, photo_timestamp)
        weather['description'] = description
        
        # Save cache info for reuse
        cached_weather_data = weather
        cached_gps = (lat, lon)
        cached_photo_timestamp = photo_timestamp
        cached_location_name = location_name
        
        # Save to database
        db_result = save_weather_data(
            weather_data=weather,
            gps_data={
                "latitude": lat,
                "longitude": lon,
                "location_name": location_name,
                "photo_timestamp": photo_timestamp
            },
            image_filename=file.filename
        )
        
        logger.info(f"Image uploaded and processed: {file.filename}")
        
        return {
            "message": "✅ Image uploaded successfully, weather data extracted and stored.",
            "file_path": str(file_path),
            "record_id": db_result["id"],
            "photo_taken_at": photo_timestamp.isoformat(),
            "location": {
                "latitude": lat,
                "longitude": lon,
                "name": location_name
            },
            "weather_summary": {
                "temperature": f"{weather['temperature']}°C",
                "feels_like": f"{weather['feels_like']}°C",
                "condition": weather['weather_condition'],
                "humidity": f"{weather['humidity']}%",
                "wind_speed": f"{weather['windspeed']} km/h",
                "description": description
            }
        }
    except Exception as e:
        logger.error(f"Error during image upload processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Reuse latest_image_path for vehicle detection
def detect_specific_type(image: Image.Image, vtype: str):
    results = model.predict(image, conf=0.3)
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
                        "x1": xyxy[0],
                        "y1": xyxy[1],
                        "x2": xyxy[2],
                        "y2": xyxy[3]
                    }
                })
    return detections

def create_vehicle_endpoint(vtype: str):
    async def detect_vehicle_type():
        if latest_image_path is None or not latest_image_path.exists():
            raise HTTPException(status_code=404, detail="No image uploaded yet. Use /upload-image/ first.")
        image = Image.open(latest_image_path).convert("RGB")
        detections = detect_specific_type(image, vtype)
        return JSONResponse({
            "vehicle_type": vtype,
            "total_detected": len(detections),
            "detections": detections
        })
    return detect_vehicle_type

for vtype in VEHICLE_TYPES:
    route_path = f"/detect/{vtype.lower()}/"
    endpoint_func = create_vehicle_endpoint(vtype)
    app.get(route_path, name=f"Detect {vtype}")(endpoint_func)

# Weather data endpoints reuse cached_weather_data and cached_photo_timestamp
def ensure_weather_data():
    if cached_weather_data is None:
        raise HTTPException(status_code=404, detail="No weather data found. Please upload an image first.")

@app.get("/get-temperature/")
def get_temperature():
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
def get_humidity():
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
def get_windspeed():
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
def get_condition():
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
def get_pressure():
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

@app.get("/get-all-weather/")
def get_all_weather():
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
        "description": weather.get('description', 'No description available'),
        "photo_taken_at": cached_photo_timestamp.isoformat() if cached_photo_timestamp else None,
        "data_sources": weather.get('data_source', []),
        "image": latest_image_path.name if latest_image_path else None
    }

@app.get("/weather-history/")
def get_history(hours: int = 24):
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

@app.get("/health/")
def health_check():
    return {
        "status": "healthy",
        "message": "Vehicle & Weather API is running",
        "version": "3.0.0"
    }

@app.get("/")
def root():
    return {
        "message": "🚗🌤️ Welcome to Vehicle Detection & Smart Weather API",
        "version": "3.0.0",
        "endpoints": {
            "upload": "POST /upload-image/ - Upload image once for both vehicle and weather data",
            "vehicle_detection": "GET /detect/{vehicle_type}/ - Detect specific vehicle types from last uploaded image",
            "temperature": "GET /get-temperature/ - Temperature with description",
            "humidity": "GET /get-humidity/ - Humidity with description",
            "windspeed": "GET /get-windspeed/ - Wind speed with description",
            "condition": "GET /get-weather-condition/ - Weather condition with description",
            "pressure": "GET /get-pressure/ - Atmospheric pressure with description",
            "all_weather": "GET /get-all-weather/ - Complete weather data with description",
            "weather_history": "GET /weather-history/?hours=24 - Weather history",
            "health": "GET /health/ - API health check"
        },
        "documentation": "/docs"
    }
