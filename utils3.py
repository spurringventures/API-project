import requests
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict
from config import config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== IMAGE PROCESSING ====================

def extract_gps_from_image(image_path: str) -> Optional[Tuple[float, float]]:
    """Extract GPS coordinates from image EXIF data"""
    try:
        image = Image.open(image_path)
        exif_data = image._getexif()
        
        if not exif_data:
            logger.warning("No EXIF data found in image")
            return None
        
        gps_info = {}
        for tag, value in exif_data.items():
            tag_name = TAGS.get(tag, tag)
            if tag_name == "GPSInfo":
                for gps_tag in value:
                    gps_tag_name = GPSTAGS.get(gps_tag, gps_tag)
                    gps_info[gps_tag_name] = value[gps_tag]
        
        if not gps_info:
            logger.warning("No GPS data found in EXIF")
            return None
        
        # Convert GPS coordinates
        lat = _convert_to_degrees(gps_info.get("GPSLatitude"))
        lon = _convert_to_degrees(gps_info.get("GPSLongitude"))
        
        if lat and lon:
            # Check hemisphere
            if gps_info.get("GPSLatitudeRef") == "S":
                lat = -lat
            if gps_info.get("GPSLongitudeRef") == "W":
                lon = -lon
            
            logger.info(f"✅ Extracted GPS: {lat}, {lon}")
            return lat, lon
        
        return None
    except Exception as e:
        logger.error(f"❌ Error extracting GPS: {e}")
        return None

def _convert_to_degrees(value):
    """Convert GPS coordinates to degrees"""
    if not value:
        return None
    d, m, s = value
    return float(d) + (float(m) / 60.0) + (float(s) / 3600.0)

def extract_timestamp_from_image(image_path: str) -> Optional[datetime]:
    """Extract timestamp when photo was taken from EXIF"""
    try:
        image = Image.open(image_path)
        exif_data = image._getexif()
        
        if not exif_data:
            logger.warning("⚠️ No EXIF data for timestamp")
            return None
        
        # Try multiple EXIF timestamp fields
        timestamp_fields = ["DateTimeOriginal", "DateTime", "DateTimeDigitized"]
        
        for tag, value in exif_data.items():
            tag_name = TAGS.get(tag, tag)
            if tag_name in timestamp_fields:
                try:
                    timestamp = datetime.strptime(value, "%Y:%m:%d %H:%M:%S")
                    logger.info(f"✅ Extracted timestamp: {timestamp}")
                    return timestamp
                except ValueError:
                    continue
        
        logger.warning("⚠️ No valid timestamp found in EXIF")
        return None
    except Exception as e:
        logger.error(f"❌ Error extracting timestamp: {e}")
        return None

def extract_location_from_image(image_path: str) -> Tuple[float, float, Optional[datetime]]:
    """Extract GPS and timestamp from image, fallback to defaults"""
    gps = extract_gps_from_image(image_path)
    timestamp = extract_timestamp_from_image(image_path)
    
    if not gps:
        logger.warning("⚠️ No GPS in image - Using default coordinates (Bengaluru)")
        gps = (12.9716, 77.5946)  # Bengaluru fallback
    
    if not timestamp:
        logger.warning("⚠️ No timestamp in image - Using current time")
        timestamp = datetime.now()
    
    return gps[0], gps[1], timestamp

# ==================== LOCATION RESOLUTION ====================

def get_location_name(lat: float, lon: float) -> str:
    """Convert coordinates to location name using reverse geocoding"""
    try:
        # Using OpenStreetMap Nominatim (free, no API key needed)
        url = "https://nominatim.openstreetmap.org/reverse"
        params = {
            "lat": lat,
            "lon": lon,
            "format": "json",
            "zoom": 10
        }
        headers = {
            "User-Agent": "WeatherAPI/2.0"
        }
        
        response = requests.get(url, params=params, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            address = data.get("address", {})
            
            # Build location name from available fields
            city = address.get("city") or address.get("town") or address.get("village")
            state = address.get("state")
            country = address.get("country")
            
            location_parts = [p for p in [city, state, country] if p]
            location_name = ", ".join(location_parts) if location_parts else "Unknown Location"
            
            logger.info(f"✅ Resolved location: {location_name}")
            return location_name
        else:
            logger.warning(f"⚠️ Geocoding failed with status {response.status_code}")
            return f"Location ({lat:.4f}, {lon:.4f})"
    except Exception as e:
        logger.error(f"❌ Error resolving location: {e}")
        return f"Location ({lat:.4f}, {lon:.4f})"

# ==================== WEATHER FETCHING ====================

def get_current_weather(lat: float, lon: float) -> Optional[Dict]:
    """Fetch current weather from Open-Meteo and OpenWeatherMap"""
    try:
        logger.info(f"🌤️ Fetching CURRENT weather for ({lat}, {lon})")
        
        meteo_params = {
            "latitude": lat,
            "longitude": lon,
            "current_weather": True,
            "hourly": "relative_humidity_2m,pressure_msl,cloudcover"
        }
        meteo_resp = requests.get(config.OPEN_METEO_URL, params=meteo_params, timeout=10)
        meteo_data = meteo_resp.json()

        owm_params = {
            "lat": lat,
            "lon": lon,
            "appid": config.OPENWEATHERMAP_API_KEY,
            "units": "metric"
        }
        owm_resp = requests.get(config.OPENWEATHERMAP_URL, params=owm_params, timeout=10)
        owm_data = owm_resp.json()

        result = {
            "temperature": meteo_data["current_weather"]["temperature"],
            "windspeed": meteo_data["current_weather"]["windspeed"],
            "weather_condition": owm_data["weather"][0]["description"].capitalize(),
            "humidity": owm_data["main"]["humidity"],
            "pressure": owm_data["main"]["pressure"],
            "feels_like": owm_data["main"]["feels_like"],
            "cloud_coverage": owm_data["clouds"]["all"],
            "timestamp": datetime.now(),
            "is_historical": False,
            "source": ["Open-Meteo", "OpenWeatherMap"]
        }
        logger.info(f"✅ Current weather fetched successfully")
        return result
    except Exception as e:
        logger.error(f"❌ Error fetching current weather: {e}")
        return None

def get_historical_weather(lat: float, lon: float, target_date: datetime) -> Optional[Dict]:
    """Fetch historical weather from Open-Meteo Archive API"""
    try:
        logger.info(f"📅 Fetching HISTORICAL weather for {target_date.strftime('%Y-%m-%d %H:%M')}")
        
        date_str = target_date.strftime("%Y-%m-%d")
        
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": date_str,
            "end_date": date_str,
            "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,cloudcover,surface_pressure,weathercode,apparent_temperature",
            "timezone": "auto"
        }
        
        resp = requests.get(config.OPEN_METEO_HISTORICAL_URL, params=params, timeout=15)
        
        if resp.status_code != 200:
            logger.error(f"❌ Historical API returned status {resp.status_code}")
            return None
            
        data = resp.json()
        
        if "hourly" not in data:
            logger.error("❌ No historical data available in response")
            return None
        
        # Get the closest hour to target time
        target_hour = target_date.hour
        hourly = data["hourly"]
        
        # Ensure we have data for that hour
        if not hourly.get("temperature_2m") or len(hourly["temperature_2m"]) <= target_hour:
            logger.warning(f"⚠️ Not enough hourly data, using hour 0")
            target_hour = 0
        
        # Get weather code description
        weather_code = hourly["weathercode"][target_hour] if hourly.get("weathercode") and len(hourly["weathercode"]) > target_hour else 0
        weather_condition = _get_weather_description(weather_code)
        
        # Get temperature values
        temperature = hourly["temperature_2m"][target_hour] if hourly.get("temperature_2m") and len(hourly["temperature_2m"]) > target_hour else None
        
        # Use apparent_temperature if available, otherwise calculate
        feels_like_temp = temperature
        if hourly.get("apparent_temperature") and len(hourly["apparent_temperature"]) > target_hour:
            feels_like_temp = hourly["apparent_temperature"][target_hour]
        elif temperature is not None:
            # Calculate feels like based on temperature, humidity, and wind speed
            humidity = hourly["relative_humidity_2m"][target_hour] if hourly.get("relative_humidity_2m") and len(hourly["relative_humidity_2m"]) > target_hour else 50
            wind_speed = hourly["wind_speed_10m"][target_hour] if hourly.get("wind_speed_10m") and len(hourly["wind_speed_10m"]) > target_hour else 0
            feels_like_temp = calculate_feels_like(temperature, humidity, wind_speed)
        
        result = {
            "temperature": temperature,
            "humidity": hourly["relative_humidity_2m"][target_hour] if hourly.get("relative_humidity_2m") and len(hourly["relative_humidity_2m"]) > target_hour else None,
            "windspeed": hourly["wind_speed_10m"][target_hour] if hourly.get("wind_speed_10m") and len(hourly["wind_speed_10m"]) > target_hour else None,
            "cloud_coverage": hourly["cloudcover"][target_hour] if hourly.get("cloudcover") and len(hourly["cloudcover"]) > target_hour else None,
            "pressure": hourly["surface_pressure"][target_hour] if hourly.get("surface_pressure") and len(hourly["surface_pressure"]) > target_hour else None,
            "weather_condition": weather_condition,
            "feels_like": feels_like_temp,
            "timestamp": target_date,
            "is_historical": True,
            "source": ["Open-Meteo Historical Archive"]
        }
        
        logger.info(f"✅ Historical weather fetched: {weather_condition}, {result['temperature']}°C, feels like {feels_like_temp}°C")
        return result
    except Exception as e:
        logger.error(f"❌ Error fetching historical weather: {e}")
        return None

def calculate_feels_like(temp: float, humidity: float, wind_speed: float) -> float:
    """Calculate feels like temperature using heat index and wind chill"""
    # For temperatures above 27°C, use heat index
    if temp > 27:
        # Simplified heat index formula
        hi = temp + 0.5555 * ((6.11 * pow(10, (7.5 * temp / (237.7 + temp))) * humidity / 100) - 10)
        return round(hi, 1)
    # For temperatures below 10°C with wind, use wind chill
    elif temp < 10 and wind_speed > 4.8:
        # Wind chill formula
        wc = 13.12 + 0.6215 * temp - 11.37 * pow(wind_speed, 0.16) + 0.3965 * temp * pow(wind_speed, 0.16)
        return round(wc, 1)
    else:
        # Temperature is comfortable, feels like actual temp
        return temp

def _get_weather_description(code: int) -> str:
    """Convert WMO weather code to description"""
    weather_codes = {
        0: "Clear sky",
        1: "Mainly clear",
        2: "Partly cloudy",
        3: "Overcast",
        45: "Foggy",
        48: "Depositing rime fog",
        51: "Light drizzle",
        53: "Moderate drizzle",
        55: "Dense drizzle",
        61: "Slight rain",
        63: "Moderate rain",
        65: "Heavy rain",
        71: "Slight snow",
        73: "Moderate snow",
        75: "Heavy snow",
        77: "Snow grains",
        80: "Slight rain showers",
        81: "Moderate rain showers",
        82: "Violent rain showers",
        85: "Slight snow showers",
        86: "Heavy snow showers",
        95: "Thunderstorm",
        96: "Thunderstorm with slight hail",
        99: "Thunderstorm with heavy hail"
    }
    return weather_codes.get(code, "Unknown")

def get_weather_for_timestamp(lat: float, lon: float, timestamp: datetime) -> Optional[Dict]:
    """Smart weather fetcher - uses current or historical API based on timestamp"""
    now = datetime.now()
    time_diff = now - timestamp
    
    logger.info(f"📸 Photo timestamp: {timestamp}")
    logger.info(f"⏰ Current time: {now}")
    logger.info(f"⏳ Time difference: {time_diff.days} days, {time_diff.seconds//3600} hours")
    
    # If photo is older than 6 hours, use historical API
    if time_diff.total_seconds() > 21600:  # 6 hours in seconds
        logger.info("🔄 Photo is old - Using HISTORICAL weather API")
        weather = get_historical_weather(lat, lon, timestamp)
        if weather:
            return weather
        else:
            logger.warning("⚠️ Historical data unavailable, falling back to current weather")
            return get_current_weather(lat, lon)
    else:
        logger.info("🔄 Photo is recent - Using CURRENT weather API")
        return get_current_weather(lat, lon)

def generate_weather_description(weather: dict, location_name: str, photo_timestamp: datetime) -> str:
    """Generate human-readable weather description"""
    try:
        temp = weather.get('temperature')
        feels_like = weather.get('feels_like')
        condition = weather.get('weather_condition', 'unknown')
        humidity = weather.get('humidity')
        wind_speed = weather.get('windspeed')
        
        # Time context
        time_ago = datetime.now() - photo_timestamp
        if time_ago.days == 0 and time_ago.seconds < 3600:
            time_context = "right now"
        elif time_ago.days == 0:
            hours_ago = time_ago.seconds // 3600
            time_context = f"{hours_ago} hour{'s' if hours_ago > 1 else ''} ago"
        elif time_ago.days == 1:
            time_context = "yesterday"
        else:
            time_context = f"{time_ago.days} days ago"
        
        # Temperature description
        if temp < 10:
            temp_desc = "quite cold"
        elif temp < 18:
            temp_desc = "cool"
        elif temp < 25:
            temp_desc = "pleasant"
        elif temp < 30:
            temp_desc = "warm"
        elif temp < 35:
            temp_desc = "hot"
        else:
            temp_desc = "very hot"
        
        # Feels like difference
        if feels_like and temp:
            diff = feels_like - temp
            if abs(diff) < 2:
                feels_desc = ""
            elif diff > 0:
                feels_desc = f", though it feels {abs(diff):.1f}°C warmer due to humidity"
            else:
                feels_desc = f", though it feels {abs(diff):.1f}°C cooler due to wind"
        else:
            feels_desc = ""
        
        # Humidity description
        if humidity:
            if humidity < 30:
                humid_desc = "The air is quite dry"
            elif humidity < 60:
                humid_desc = "Humidity levels are comfortable"
            elif humidity < 80:
                humid_desc = "It's moderately humid"
            else:
                humid_desc = "It's very humid"
        else:
            humid_desc = ""
        
        # Wind description
        if wind_speed:
            if wind_speed < 5:
                wind_desc = "with calm winds"
            elif wind_speed < 15:
                wind_desc = "with a gentle breeze"
            elif wind_speed < 30:
                wind_desc = "with moderate winds"
            else:
                wind_desc = "with strong winds"
        else:
            wind_desc = ""
        
        # Build complete description
        description = f"In {location_name} {time_context}, the weather was {condition.lower()} with {temp_desc} conditions at {temp}°C{feels_desc}. {humid_desc} {wind_desc}."
        
        return description.strip()
    except Exception as e:
        logger.error(f"Error generating description: {e}")
        return "Weather information is available but description could not be generated."

def validate_coordinates(lat: float, lon: float) -> bool:
    """Validate latitude and longitude"""
    is_valid = -90 <= lat <= 90 and -180 <= lon <= 180
    if not is_valid:
        logger.error(f"❌ Invalid coordinates: lat={lat}, lon={lon}")
    return is_valid
