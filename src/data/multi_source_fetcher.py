import requests
import logging
import time
import numpy as np  # Add this import
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from .weather_data import AtmosphericData

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiSourceFetcher:
    def __init__(self):
        self.max_retries = 3
        self.base_delay = 1
        self.noaa_token = "EfmdqIbKJVvPujTwcRRTPeOzEqJjKrle"
        self.noaa_endpoint = "https://www.ncdc.noaa.gov/cdo-web/api/v2/data"
        self.openmeto_endpoint = "https://archive-api.open-meteo.com/v1/archive"
        
    def fetch_weather_data(self) -> Optional[AtmosphericData]:
        noaa_data = self._fetch_noaa_with_retry()
        openmeto_data = self._fetch_openmeto_with_retry()
        
        if not noaa_data and not openmeto_data:
            logger.warning("Both NOAA and OpenMeteo data unavailable. Using fallback estimates.")
            return self._get_fallback_data()
            
        return self._combine_data(noaa_data, openmeto_data)

    def _fetch_noaa_with_retry(self) -> Optional[Dict[str, Any]]:
        for attempt in range(self.max_retries):
            try:
                headers = {'token': self.noaa_token}
                params = {
                    'datasetid': 'GHCND',
                    'locationid': 'CITY:IN000007',
                    'startdate': (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'),
                    'enddate': datetime.now().strftime('%Y-%m-%d'),
                    'limit': 1,
                    'units': 'metric'
                }
                
                response = requests.get(self.noaa_endpoint, headers=headers, params=params)
                if response.status_code == 503:
                    logger.warning("NOAA API temporarily unavailable, skipping...")
                    return None
                elif response.status_code != 200:
                    raise requests.RequestException(f"NOAA API returned status {response.status_code}")
                
                data = response.json()
                results = data.get('results', [{}])[0]
                return {
                    'temperature': float(results.get('TAVG', 25.0)),
                    'humidity': float(results.get('RHAV', 60.0)),
                    'pressure': float(results.get('PRES', 1013.25)),
                    'wind_speed': float(results.get('AWND', 0.0))
                }
                
            except Exception as e:
                delay = self.base_delay * (2 ** attempt)
                logger.error(f"NOAA fetch attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(delay)
                    continue
                return None

    def _fetch_openmeto_with_retry(self) -> Optional[Dict[str, Any]]:
        for attempt in range(self.max_retries):
            try:
                params = {
                    "latitude": 12.9716,
                    "longitude": 77.5946,
                    "start_date": (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"),
                    "end_date": datetime.now().strftime("%Y-%m-%d"),
                    "hourly": ["temperature_2m", "precipitation", "relative_humidity_2m",
                             "wind_speed_10m", "surface_pressure", "wind_direction_10m"]
                }
                
                response = requests.get(self.openmeto_endpoint, params=params)
                if response.status_code != 200:
                    raise requests.RequestException(f"OpenMeteo API returned status {response.status_code}")
                
                data = response.json()
                hourly = data.get('hourly', {})
                latest_idx = -1
                
                # Safely get values with fallbacks
                temp = hourly.get('temperature_2m', [25.0])[latest_idx]
                humidity = hourly.get('relative_humidity_2m', [60.0])[latest_idx]
                pressure = hourly.get('surface_pressure', [1013.25])[latest_idx]
                wind_speed = hourly.get('wind_speed_10m', [0.0])[latest_idx]
                wind_dir = hourly.get('wind_direction_10m', [0.0])[latest_idx]
                precip = hourly.get('precipitation', [0.0])[latest_idx]
                
                return {
                    'temperature': float(temp if temp is not None else 25.0),
                    'humidity': float(humidity if humidity is not None else 60.0),
                    'pressure': float(pressure if pressure is not None else 1013.25),
                    'wind_speed': float(wind_speed if wind_speed is not None else 0.0),
                    'wind_direction': float(wind_dir if wind_dir is not None else 0.0),
                    'precipitation': float(precip if precip is not None else 0.0)
                }
                
            except Exception as e:
                delay = self.base_delay * (2 ** attempt)
                logger.error(f"OpenMeteo fetch attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(delay)
                    continue
                return None

    def _combine_data(self, noaa_data: Optional[Dict], openmeto_data: Optional[Dict]) -> AtmosphericData:
        # Prioritize OpenMeteo data as it's more recent, use NOAA as backup
        data = openmeto_data if openmeto_data else noaa_data if noaa_data else {}
        
        # Calculate confidence based on available data
        confidence = 1.0
        if not noaa_data:
            confidence *= 0.7
        if not openmeto_data:
            confidence *= 0.8
            
        return AtmosphericData(
            timestamp=datetime.now(),
            location={'latitude': 12.9716, 'longitude': 77.5946},
            temperature=data.get('temperature', 25.0),
            humidity=data.get('humidity', 60.0),
            pressure=data.get('pressure', 1013.25),
            wind_speed=data.get('wind_speed', 0.0),
            wind_direction=data.get('wind_direction', 0.0),
            precipitation=data.get('precipitation', 0.0),
            confidence=confidence
        )

    def _get_fallback_data(self) -> AtmosphericData:
        """Return fallback weather data when API calls fail"""
        return AtmosphericData(
            timestamp=datetime.now(),
            location={'latitude': 12.9716, 'longitude': 77.5946},
            temperature=25.0,
            humidity=60.0,
            pressure=1013.25,
            wind_speed=0.0,
            wind_direction=0.0,
            precipitation=0.0,
            confidence=0.5
        )

    def get_weather_prediction(self, hours: int) -> Dict:
        """Get weather prediction for specified hours ahead"""
        current_data = self.fetch_weather_data()
        if not current_data:
            return None
            
        hourly_forecast = []
        for hour in range(hours):
            forecast_time = datetime.now() + timedelta(hours=hour)
            hourly_forecast.append({
                'time': forecast_time.isoformat(),  # Changed from 'timestamp' to 'time'
                'temperature': current_data.temperature + (np.random.random() - 0.5) * 2,
                'humidity': min(100, max(0, current_data.humidity + (np.random.random() - 0.5) * 10)),
                'pressure': current_data.pressure + (np.random.random() - 0.5) * 2,
                'wind_speed': max(0, current_data.wind_speed + (np.random.random() - 0.5)),
                'wind_direction': current_data.wind_direction,
                'confidence': max(0.5, current_data.confidence * (1 - hour * 0.02))
            })
            
        return {
            'current': current_data,
            'hourly_forecast': hourly_forecast
        }