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

# Imports from main3 (your vehicle detection logic)
import main3

# Imports from utils.py
from utils import (
    extract_location_from_image,
    get_weather_for_timestamp,
    validate_coordinates,
    get_location_name,
    generate_weather_description
)

# Imports from database.py
from database import (
    init_database,
    save_weather_data,
    get_latest_weather,
    get_weather_history,
    init_connection_pool,
    close_connection_pool
)

# Imports from config.py
from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============== Lifespan Context Manager (From main.py) ==============
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting Weather API...")
    os.makedirs(config.UPLOAD_DIR, exist_ok=True)
    init_connection_pool()
    init_database()
    logger.info("✅ Application started successfully!")
    yield
    logger.info("🛑 Shutting down Weather API...")
    close_connection_pool()
    logger.info("✅ Application shutdown complete!")

app = FastAPI(
    title="Vehicle Detection & Weather API",
    description="Upload images for vehicle detection and weather data extraction with human-readable descriptions.",
    version="2.0.0",
    lifespan=lifespan
)

# ============== Vehicle Detection Setup (From app.py) ==============
IMAGES_DIR = Path("uploaded_images")
IMAGES_DIR.mkdir(exist_ok=True)
app.mount("/uploaded_images", StaticFiles(directory=IMAGES_DIR), name="uploaded_images")

MODEL_PATH = "best_vehicle1.pt"
model = YOLO(MODEL_PATH)

VEHICLE_TYPES = [
    "ambulance", "car", "bus", "motorbike", "truck",
    "ricksaw", "threewheel", "cng", "bicycle", "motorcycle", "otherVehicles"
]

latest_image_path = None

# --- Vehicle Image Upload Endpoint (From app.py) ---
@app.post("/vehicle-upload-image/")
async def vehicle_upload_image(file: UploadFile = File(...)):
    global latest_image_path
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    image_bytes = await file.read()
    file_path = IMAGES_DIR / file.filename
    with open(file_path, "wb") as f:
        f.write(image_bytes)
    latest_image_path = file_path
    return JSONResponse({
        "message": "Image uploaded successfully.",
        "file_path": str(file_path)
    })

# --- Helper for Detection (From app.py) ---
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

# --- Dynamic Vehicle Endpoints (From app.py) ---
def create_vehicle_endpoint(vtype: str):
    async def detect_vehicle_type():
        if latest_image_path is None or not Path(latest_image_path).exists():
            raise HTTPException(status_code=404, detail="No image uploaded yet. Use /vehicle-upload-image/ first.")
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

# ==================== MAIN UPLOAD ENDPOINT (From main.py) ====================
@app.post("/upload-image/", status_code=status.HTTP_201_CREATED)
async def upload_image(file: UploadFile = File(...)):
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail="Only image files are allowed"
            )
        file_path = os.path.join(config.UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"📸 Image uploaded: {file.filename}")
        lat, lon, photo_timestamp = extract_location_from_image(file_path)
        if not validate_coordinates(lat, lon):
            raise HTTPException(
                status_code=400,
                detail="Invalid GPS coordinates extracted"
            )
        logger.info(f"📍 Location: ({lat}, {lon})")
        logger.info(f"📅 Photo taken at: {photo_timestamp}")
        location_name = get_location_name(lat, lon)
        weather = get_weather_for_timestamp(lat, lon, photo_timestamp)
        if not weather:
            raise HTTPException(
                status_code=503,
                detail="Unable to fetch weather data from APIs"
            )
        description = generate_weather_description(weather, location_name, photo_timestamp)
        weather['description'] = description
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
        return {
            "message": "✅ Image uploaded and weather data stored successfully",
            "record_id": db_result["id"],
            "photo_taken_at": photo_timestamp.isoformat(),
            "data_type": "historical" if weather.get("is_historical") else "current",
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
            },
            "data_sources": weather.get("source", [])
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error processing image: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

# ==================== Other Weather Endpoints (From main.py) ====================
@app.get("/get-temperature/")
def get_temperature():
    try:
        weather = get_latest_weather()
        if not weather:
            raise HTTPException(
                status_code=404,
                detail="No weather data found. Please upload an image first."
            )
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
            "photo_taken_at": weather['photo_timestamp'].isoformat() if weather.get('photo_timestamp') else None
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error fetching temperature: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get-humidity/")
def get_humidity():
    try:
        weather = get_latest_weather()
        if not weather:
            raise HTTPException(
                status_code=404,
                detail="No weather data found. Please upload an image first."
            )
        humidity = weather.get('humidity')
        if humidity:
            if humidity < 30:
                desc = f"The humidity is {humidity}%, which is quite dry. You might notice dry skin or static electricity."
            elif humidity < 60:
                desc = f"The humidity is {humidity}%, which is comfortable. It's a pleasant level that most people enjoy."
            elif humidity < 80:
                desc = f"The humidity is {humidity}%, which is moderately humid. You might feel a bit sticky."
            else:
                desc = f"The humidity is {humidity}%, which is very humid. It can feel quite muggy and uncomfortable."
        else:
            desc = "Humidity data is not available."
        return {
            "humidity": f"{humidity}%" if humidity else "N/A",
            "description": desc,
            "photo_taken_at": weather['photo_timestamp'].isoformat() if weather.get('photo_timestamp') else None
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error fetching humidity: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get-windspeed/")
def get_windspeed():
    try:
        weather = get_latest_weather()
        if not weather:
            raise HTTPException(
                status_code=404,
                detail="No weather data found. Please upload an image first."
            )
        wind = weather.get('wind_speed')
        if wind:
            if wind < 5:
                desc = f"The wind speed is {wind} km/h, which is calm. You'll barely notice any breeze."
            elif wind < 15:
                desc = f"The wind speed is {wind} km/h, creating a gentle breeze. It's pleasant and refreshing."
            elif wind < 30:
                desc = f"The wind speed is {wind} km/h, with moderate winds. You'll definitely feel the breeze."
            else:
                desc = f"The wind speed is {wind} km/h, with strong winds. It might be difficult to use an umbrella!"
        else:
            desc = "Wind speed data is not available."
        return {
            "wind_speed": f"{wind} km/h" if wind else "N/A",
            "description": desc,
            "photo_taken_at": weather['photo_timestamp'].isoformat() if weather.get('photo_timestamp') else None
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error fetching wind speed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get-weather-condition/")
def get_condition():
    try:
        weather = get_latest_weather()
        if not weather:
            raise HTTPException(
                status_code=404,
                detail="No weather data found. Please upload an image first."
            )
        condition = weather.get('weather_condition', 'Unknown')
        clouds = weather.get('cloud_coverage')
        if clouds is not None:
            desc = f"The weather condition is {condition.lower()} with {clouds}% cloud coverage. "
            if clouds < 20:
                desc += "The sky is mostly clear with beautiful visibility."
            elif clouds < 50:
                desc += "There are some clouds but plenty of sunshine too."
            elif clouds < 80:
                desc += "The sky is quite cloudy with limited sunshine."
            else:
                desc += "The sky is heavily overcast with very little sunshine."
        else:
            desc = f"The weather condition is {condition.lower()}."
        return {
            "condition": condition,
            "cloud_coverage": f"{clouds}%" if clouds is not None else "N/A",
            "description": desc,
            "photo_taken_at": weather['photo_timestamp'].isoformat() if weather.get('photo_timestamp') else None
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error fetching condition: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get-pressure/")
def get_pressure():
    try:
        weather = get_latest_weather()
        if not weather:
            raise HTTPException(
                status_code=404,
                detail="No weather data found. Please upload an image first."
            )
        pressure = weather.get('pressure')
        if pressure:
            if pressure < 1000:
                desc = f"The atmospheric pressure is {pressure} hPa, which is low. This typically indicates stormy or unsettled weather."
            elif pressure < 1013:
                desc = f"The atmospheric pressure is {pressure} hPa, which is slightly below normal. Weather might be changing."
            elif pressure < 1020:
                desc = f"The atmospheric pressure is {pressure} hPa, which is normal. This indicates stable weather conditions."
            else:
                desc = f"The atmospheric pressure is {pressure} hPa, which is high. This usually means clear and settled weather."
        else:
            desc = "Pressure data is not available."
        return {
            "pressure": f"{pressure} hPa" if pressure else "N/A",
            "description": desc,
            "photo_taken_at": weather['photo_timestamp'].isoformat() if weather.get('photo_timestamp') else None
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error fetching pressure: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get-all-weather/")
def get_all_weather():
    try:
        weather = get_latest_weather()
        if not weather:
            raise HTTPException(
                status_code=404,
                detail="No weather data found. Please upload an image first."
            )
        return {
            "location": {
                "latitude": weather['latitude'],
                "longitude": weather['longitude'],
                "name": weather['location_name']
            },
            "weather": {
                "condition": weather['weather_condition'] if weather.get('weather_condition') else "Unknown",
                "temperature": f"{weather['temperature']}°C" if weather.get('temperature') else "N/A",
                "feels_like": f"{weather['feels_like']}°C" if weather.get('feels_like') else "N/A",
                "humidity": f"{weather['humidity']}%" if weather.get('humidity') else "N/A",
                "wind_speed": f"{weather['wind_speed']} km/h" if weather.get('wind_speed') else "N/A",
                "pressure": f"{weather['pressure']} hPa" if weather.get('pressure') else "N/A",
                "cloud_coverage": f"{weather['cloud_coverage']}%" if weather.get('cloud_coverage') else "N/A"
            },
            "description": weather.get('description', 'No description available'),
            "photo_taken_at": weather['photo_timestamp'].isoformat() if weather.get('photo_timestamp') else None,
            "data_sources": weather['data_source'] if weather.get('data_source') else [],
            "image": weather['image_filename']
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error fetching all weather: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== Utility Endpoints (From main.py) ====================
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
        "message": "Weather API is running",
        "version": "2.0.0"
    }

@app.get("/")
def root():
    return {
        "message": "🌤️ Welcome to Smart Weather & Vehicle Detection API",
        "version": "2.0.0",
        "endpoints": {
            "vehicle_upload": "POST /vehicle-upload-image/ - Upload image for vehicle detection",
            "vehicle_detect": "GET /detect/{vehicle_type}/ - Detect specific vehicle types",
            "weather_upload": "POST /upload-image/ - Upload image to extract weather",
            "temperature": "GET /get-temperature/ - Get temperature with description",
            "humidity": "GET /get-humidity/ - Get humidity with description",
            "windspeed": "GET /get-windspeed/ - Get wind speed with description",
            "condition": "GET /get-weather-condition/ - Get weather condition with description",
            "pressure": "GET /get-pressure/ - Get pressure with description",
            "all": "GET /get-all-weather/ - Get complete weather data with description",
            "history": "GET /weather-history/?hours=24 - Get weather history",
            "health": "GET /health/ - API health check"
        },
        "documentation": "/docs"
    }
