from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse
import os
import shutil
from utils import (
    extract_location_from_image,
    get_weather_for_timestamp,
    validate_coordinates,
    get_location_name
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
import logging
from contextlib import asynccontextmanager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🚀 Starting Weather API...")
    os.makedirs(config.UPLOAD_DIR, exist_ok=True)
    init_connection_pool()
    init_database()
    logger.info("✅ Application started successfully!")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Weather API...")
    close_connection_pool()
    logger.info("✅ Application shutdown complete!")

app = FastAPI(
    title="Smart Weather API",
    description="Upload images and get weather data automatically",
    version="2.0.0",
    lifespan=lifespan
)

# ==================== MAIN UPLOAD ENDPOINT ====================

@app.post("/upload-image/", status_code=status.HTTP_201_CREATED)
async def upload_image(file: UploadFile = File(...)):
    """
    Upload an image with GPS and timestamp data.
    Weather data is automatically fetched and stored.
    """
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail="Only image files are allowed"
            )
        
        # Save file
        file_path = os.path.join(config.UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"📸 Image uploaded: {file.filename}")
        
        # Extract GPS and timestamp
        lat, lon, photo_timestamp = extract_location_from_image(file_path)
        
        if not validate_coordinates(lat, lon):
            raise HTTPException(
                status_code=400,
                detail="Invalid GPS coordinates extracted"
            )
        
        logger.info(f"📍 Location: ({lat}, {lon})")
        logger.info(f"📅 Photo taken at: {photo_timestamp}")
        
        # Get actual location name
        location_name = get_location_name(lat, lon)
        
        # Fetch weather based on timestamp
        weather = get_weather_for_timestamp(lat, lon, photo_timestamp)
        
        if not weather:
            raise HTTPException(
                status_code=503,
                detail="Unable to fetch weather data from APIs"
            )
        
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
        
        return {
            "message": "✅ Image uploaded and weather data stored successfully",
            "record_id": db_result["id"],
            "uploaded_at": db_result["timestamp"],
            "photo_taken_at": photo_timestamp.isoformat(),
            "data_type": "historical" if weather.get("is_historical") else "current",
            "location": {
                "latitude": lat,
                "longitude": lon,
                "name": location_name
            },
            "weather_summary": {
                "temperature": f"{weather['temperature']}°C",
                "condition": weather['weather_condition'],
                "humidity": f"{weather['humidity']}%",
                "wind_speed": f"{weather['windspeed']} km/h"
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

# ==================== AUTO-FETCH ENDPOINTS (NO PARAMS NEEDED) ====================

@app.get("/get-temperature/")
def get_temperature():
    """Get temperature from the last uploaded image"""
    try:
        weather = get_latest_weather()
        if not weather:
            raise HTTPException(
                status_code=404,
                detail="No weather data found. Please upload an image first."
            )
        
        return {
            "temperature": f"{weather['temperature']}°C" if weather['temperature'] else "N/A",
            "feels_like": f"{weather['feels_like']}°C" if weather.get('feels_like') else "N/A",
            "recorded_at": weather['timestamp'].isoformat() if weather.get('timestamp') else None,
            "photo_taken_at": weather['photo_timestamp'].isoformat() if weather.get('photo_timestamp') else None
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error fetching temperature: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get-humidity/")
def get_humidity():
    """Get humidity from the last uploaded image"""
    try:
        weather = get_latest_weather()
        if not weather:
            raise HTTPException(
                status_code=404,
                detail="No weather data found. Please upload an image first."
            )
        
        return {
            "humidity": f"{weather['humidity']}%" if weather.get('humidity') else "N/A",
            "recorded_at": weather['timestamp'].isoformat() if weather.get('timestamp') else None,
            "photo_taken_at": weather['photo_timestamp'].isoformat() if weather.get('photo_timestamp') else None
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error fetching humidity: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get-windspeed/")
def get_windspeed():
    """Get wind speed from the last uploaded image"""
    try:
        weather = get_latest_weather()
        if not weather:
            raise HTTPException(
                status_code=404,
                detail="No weather data found. Please upload an image first."
            )
        
        return {
            "wind_speed": f"{weather['wind_speed']} km/h" if weather.get('wind_speed') else "N/A",
            "recorded_at": weather['timestamp'].isoformat() if weather.get('timestamp') else None,
            "photo_taken_at": weather['photo_timestamp'].isoformat() if weather.get('photo_timestamp') else None
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error fetching wind speed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get-weather-condition/")
def get_condition():
    """Get weather condition from the last uploaded image"""
    try:
        weather = get_latest_weather()
        if not weather:
            raise HTTPException(
                status_code=404,
                detail="No weather data found. Please upload an image first."
            )
        
        return {
            "condition": weather['weather_condition'] if weather.get('weather_condition') else "Unknown",
            "cloud_coverage": f"{weather['cloud_coverage']}%" if weather.get('cloud_coverage') else "N/A",
            "recorded_at": weather['timestamp'].isoformat() if weather.get('timestamp') else None,
            "photo_taken_at": weather['photo_timestamp'].isoformat() if weather.get('photo_timestamp') else None
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error fetching condition: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get-pressure/")
def get_pressure():
    """Get atmospheric pressure from the last uploaded image"""
    try:
        weather = get_latest_weather()
        if not weather:
            raise HTTPException(
                status_code=404,
                detail="No weather data found. Please upload an image first."
            )
        
        return {
            "pressure": f"{weather['pressure']} hPa" if weather.get('pressure') else "N/A",
            "recorded_at": weather['timestamp'].isoformat() if weather.get('timestamp') else None,
            "photo_taken_at": weather['photo_timestamp'].isoformat() if weather.get('photo_timestamp') else None
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error fetching pressure: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get-all-weather/")
def get_all_weather():
    """Get complete weather data from the last uploaded image"""
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
            "timestamps": {
                "photo_taken_at": weather['photo_timestamp'].isoformat() if weather.get('photo_timestamp') else None,
                "data_recorded_at": weather['timestamp'].isoformat() if weather.get('timestamp') else None
            },
            "data_sources": weather['data_source'] if weather.get('data_source') else [],
            "image": weather['image_filename']
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error fetching all weather: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== UTILITY ENDPOINTS ====================

@app.get("/weather-history/")
def get_history(hours: int = 24):
    """Get weather history for the last N hours"""
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
    """API health check endpoint"""
    return {
        "status": "healthy",
        "message": "Weather API is running",
        "version": "2.0.0"
    }

@app.get("/")
def root():
    """Root endpoint with API information"""
    return {
        "message": "🌤️ Welcome to Smart Weather API",
        "version": "2.0.0",
        "endpoints": {
            "upload": "POST /upload-image/ - Upload image to extract weather",
            "temperature": "GET /get-temperature/ - Get temperature from last upload",
            "humidity": "GET /get-humidity/ - Get humidity from last upload",
            "windspeed": "GET /get-windspeed/ - Get wind speed from last upload",
            "condition": "GET /get-weather-condition/ - Get weather condition from last upload",
            "pressure": "GET /get-pressure/ - Get pressure from last upload",
            "all": "GET /get-all-weather/ - Get complete weather data from last upload",
            "history": "GET /weather-history/?hours=24 - Get weather history",
            "health": "GET /health/ - API health check"
        },
        "documentation": "/docs"
    }
