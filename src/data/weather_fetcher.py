import requests
from datetime import datetime
from typing import Dict, Optional
from .weather_data import AtmosphericData
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class OpenMeteoFetcher:
    def __init__(self):
        self.latitude = 12.9716
        self.longitude = 77.5946
        self.base_url = "https://api.open-meteo.com/v1/forecast"
        logger.debug("Initialized OpenMeteoFetcher")
        
    def fetch_weather_data(self) -> Optional[AtmosphericData]:
        try:
            logger.debug("Attempting to fetch weather data")
            
            params = {
                "latitude": self.latitude,
                "longitude": self.longitude,
                "current": ["temperature_2m", "relative_humidity_2m", "pressure_msl", 
                          "wind_speed_10m", "wind_direction_10m"],
                "timezone": "auto"
            }
            
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            current = data["current"]
            
            return AtmosphericData(
                timestamp=datetime.fromisoformat(current["time"]),
                location={
                    "latitude": self.latitude,
                    "longitude": self.longitude
                },
                temperature=current["temperature_2m"],
                humidity=current["relative_humidity_2m"],
                pressure=current["pressure_msl"],
                wind_speed=current["wind_speed_10m"],
                wind_direction=current["wind_direction_10m"],
                confidence=0.95
            )
            
        except Exception as e:
            logger.error(f"Error fetching weather data: {e}")
            return None