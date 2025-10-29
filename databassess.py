import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager
from config import config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Connection Pool
connection_pool = None

def init_connection_pool():
    """Initialize PostgreSQL connection pool"""
    global connection_pool
    try:
        connection_pool = psycopg2.pool.SimpleConnectionPool(
            1, 10,
            host=config.DB_HOST,
            database=config.DB_NAME,
            user=config.DB_USER,
            password=config.DB_PASSWORD,
            port=config.DB_PORT
        )
        logger.info("✅ Database connection pool created")
    except Exception as e:
        logger.error(f"❌ Error creating connection pool: {e}")
        raise e

@contextmanager
def get_db_connection():
    """Context manager for database connections from pool"""
    if connection_pool is None:
        init_connection_pool()
    
    conn = connection_pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"Database error: {e}")
        raise e
    finally:
        connection_pool.putconn(conn)

def init_database():
    """Initialize database tables"""
    create_table_query = """
    CREATE TABLE IF NOT EXISTS weather_analysis (
        id SERIAL PRIMARY KEY,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        photo_timestamp TIMESTAMP,
        latitude DOUBLE PRECISION,
        longitude DOUBLE PRECISION,
        location_name VARCHAR(255),
        weather_condition VARCHAR(100),
        temperature FLOAT,
        humidity FLOAT,
        wind_speed FLOAT,
        pressure FLOAT,
        feels_like FLOAT,
        cloud_coverage FLOAT,
        air_quality_index VARCHAR(100),
        description TEXT,
        image_filename VARCHAR(255),
        data_source TEXT[]
    );
    """
    create_idx_timestamp = "CREATE INDEX IF NOT EXISTS idx_timestamp ON weather_analysis(timestamp DESC);"
    create_idx_location = "CREATE INDEX IF NOT EXISTS idx_location ON weather_analysis(latitude, longitude);"
    create_idx_photo_timestamp = "CREATE INDEX IF NOT EXISTS idx_photo_timestamp ON weather_analysis(photo_timestamp DESC);"
    
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(create_table_query)
            cur.execute(create_idx_timestamp)
            cur.execute(create_idx_location)
            cur.execute(create_idx_photo_timestamp)
    
    logger.info("✅ Database initialized successfully!")

def save_weather_data(weather_data: dict, gps_data: dict, image_filename: str) -> dict:
    """Save weather analysis data to PostgreSQL"""
    insert_query = """
    INSERT INTO weather_analysis (
        latitude, longitude, location_name, photo_timestamp,
        weather_condition, temperature, humidity, wind_speed, pressure,
        feels_like, cloud_coverage, air_quality_index, 
        description, image_filename, data_source
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    RETURNING id, timestamp;
    """
    
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(insert_query, (
                    gps_data.get("latitude"),
                    gps_data.get("longitude"),
                    gps_data.get("location_name", "Unknown"),
                    gps_data.get("photo_timestamp"),
                    weather_data.get("weather_condition"),
                    weather_data.get("temperature"),
                    weather_data.get("humidity"),
                    weather_data.get("windspeed"),
                    weather_data.get("pressure"),
                    weather_data.get("feels_like"),
                    weather_data.get("cloud_coverage"),
                    weather_data.get("air_quality_index", "N/A"),
                    weather_data.get("description", ""),
                    image_filename,
                    weather_data.get("source", [])
                ))
                result = cur.fetchone()
                logger.info(f"✅ Weather data saved with ID: {result['id']}")
                return {"id": result["id"], "timestamp": result["timestamp"]}
    except Exception as e:
        logger.error(f"❌ Error saving to database: {e}")
        raise e

def get_latest_weather() -> dict:
    """Get the most recent weather data"""
    query = "SELECT * FROM weather_analysis ORDER BY timestamp DESC LIMIT 1;"
    
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query)
                result = cur.fetchone()
                if result:
                    logger.info(f"✅ Retrieved latest weather record ID: {result['id']}")
                else:
                    logger.warning("⚠️ No weather data found in database")
                return result
    except Exception as e:
        logger.error(f"❌ Error fetching latest weather: {e}")
        raise e

def get_weather_history(hours: int = 24) -> list:
    """Get weather history for the last N hours"""
    query = """
    SELECT * FROM weather_analysis 
    WHERE timestamp > NOW() - INTERVAL '%s hours'
    ORDER BY timestamp DESC;
    """
    
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, (hours,))
                results = cur.fetchall()
                logger.info(f"✅ Retrieved {len(results)} historical records")
                return results
    except Exception as e:
        logger.error(f"❌ Error fetching weather history: {e}")
        raise e

def get_weather_by_id(record_id: int) -> dict:
    """Get weather data by record ID"""
    query = "SELECT * FROM weather_analysis WHERE id = %s;"
    
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, (record_id,))
                result = cur.fetchone()
                if result:
                    logger.info(f"✅ Retrieved weather record ID: {record_id}")
                else:
                    logger.warning(f"⚠️ No record found with ID: {record_id}")
                return result
    except Exception as e:
        logger.error(f"❌ Error fetching weather by ID: {e}")
        raise e

def close_connection_pool():
    """Close all connections in the pool"""
    global connection_pool
    if connection_pool:
        connection_pool.closeall()
        logger.info("✅ Connection pool closed")
 
