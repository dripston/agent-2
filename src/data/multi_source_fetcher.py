import requests
from datetime import datetime, timedelta  # Add timedelta import
from typing import Dict, Optional, List
from .weather_data import AtmosphericData
import logging
import statistics

logger = logging.getLogger(__name__)

class WeatherDataSource:
    def __init__(self, name: str, confidence: float):
        self.name = name
        self.confidence = confidence
        
    def fetch(self) -> Optional[AtmosphericData]:
        raise NotImplementedError

class OpenMeteoSource(WeatherDataSource):
    def __init__(self):
        super().__init__("OpenMeteo", 0.9)
        self.base_url = "https://api.open-meteo.com/v1/forecast"
        self.latitude = 12.9716
        self.longitude = 77.5946
        
    def fetch(self) -> Optional[AtmosphericData]:
        try:
            params = {
                "latitude": self.latitude,
                "longitude": self.longitude,
                "current": ["temperature_2m", "relative_humidity_2m", 
                          "pressure_msl", "wind_speed_10m", "wind_direction_10m"],
                "timezone": "auto"
            }
            response = requests.get(self.base_url, params=params)
            data = response.json()["current"]
            
            return AtmosphericData(
                timestamp=datetime.fromisoformat(data["time"]),
                location={"latitude": self.latitude, "longitude": self.longitude},
                temperature=data["temperature_2m"],
                humidity=data["relative_humidity_2m"],
                pressure=data["pressure_msl"],
                wind_speed=data["wind_speed_10m"],
                wind_direction=data["wind_direction_10m"],
                confidence=self.confidence
            )
        except Exception as e:
            logger.error(f"OpenMeteo fetch error: {e}")
            return None

class TomorrowIOSource(WeatherDataSource):
    def __init__(self):
        super().__init__("TomorrowIO", 0.85)
        self.base_url = "https://api.tomorrow.io/v4/weather/realtime"
        self.latitude = 12.9716
        self.longitude = 77.5946
        
    def fetch(self) -> Optional[AtmosphericData]:
        try:
            params = {
                "location": f"{self.latitude},{self.longitude}",
                "fields": ["temperature", "humidity", "pressureSurfaceLevel", 
                          "windSpeed", "windDirection"],
                "units": "metric"
            }
            # Using free tier without API key for demo
            response = requests.get(self.base_url, params=params)
            data = response.json()["data"]["values"]
            
            return AtmosphericData(
                timestamp=datetime.now(),
                location={"latitude": self.latitude, "longitude": self.longitude},
                temperature=data["temperature"],
                humidity=data["humidity"],
                pressure=data["pressureSurfaceLevel"],
                wind_speed=data["windSpeed"],
                wind_direction=data["windDirection"],
                confidence=self.confidence
            )
        except Exception as e:
            logger.error(f"TomorrowIO fetch error: {e}")
            return None

class WeatherBitSource(WeatherDataSource):
    def __init__(self):
        super().__init__("WeatherBit", 0.8)
        self.base_url = "https://api.weatherbit.io/v2.0/current"
        self.latitude = 12.9716
        self.longitude = 77.5946
        
    def fetch(self) -> Optional[AtmosphericData]:
        try:
            params = {
                "lat": self.latitude,
                "lon": self.longitude,
                "units": "M"
            }
            response = requests.get(self.base_url, params=params)
            data = response.json()["data"][0]
            
            return AtmosphericData(
                timestamp=datetime.now(),
                location={"latitude": self.latitude, "longitude": self.longitude},
                temperature=data["temp"],
                humidity=data["rh"],
                pressure=data["pres"],
                wind_speed=data["wind_spd"],
                wind_direction=data["wind_dir"],
                confidence=self.confidence
            )
        except Exception as e:
            logger.error(f"WeatherBit fetch error: {e}")
            return None

# Add these new source classes

class NOAASource(WeatherDataSource):
    def __init__(self):
        super().__init__("NOAA", 0.88)
        self.base_url = "https://api.weather.gov/points/12.9716,77.5946"
        
    def fetch(self) -> Optional[AtmosphericData]:
        try:
            response = requests.get(self.base_url)
            forecast_url = response.json()["properties"]["forecast"]
            weather = requests.get(forecast_url).json()["properties"]["periods"][0]
            
            return AtmosphericData(
                timestamp=datetime.now(),
                location={"latitude": 12.9716, "longitude": 77.5946},
                temperature=weather["temperature"],
                humidity=weather["relativeHumidity"]["value"],
                pressure=weather["barometricPressure"]["value"]/100,  # Convert to hPa
                wind_speed=weather["windSpeed"],
                wind_direction=weather["windDirection"],
                confidence=self.confidence
            )
        except Exception as e:
            logger.error(f"NOAA fetch error: {e}")
            return None

class MeteostatSource(WeatherDataSource):
    def __init__(self):
        super().__init__("Meteostat", 0.85)
        self.station_id = "VOBG"  # Bengaluru station
        self.base_url = f"https://meteostat.p.rapidapi.com/stations/hourly"
        
    def fetch(self) -> Optional[AtmosphericData]:
        try:
            params = {
                "station": self.station_id,
                "start": datetime.now().strftime("%Y-%m-%d"),
                "end": datetime.now().strftime("%Y-%m-%d")
            }
            response = requests.get(self.base_url, params=params)
            data = response.json()["data"][-1]
            
            return AtmosphericData(
                timestamp=datetime.now(),
                location={"latitude": 12.9716, "longitude": 77.5946},
                temperature=data["temp"],
                humidity=data["rhum"],
                pressure=data["pres"],
                wind_speed=data["wspd"],
                wind_direction=data["wdir"],
                confidence=self.confidence
            )
        except Exception as e:
            logger.error(f"Meteostat fetch error: {e}")
            return None

class SentinelSatelliteSource(WeatherDataSource):
    def __init__(self):
        super().__init__("Sentinel", 0.92)
        self.base_url = "https://services.sentinel-hub.com/api/v1/process"
        
    def fetch(self) -> Optional[AtmosphericData]:
        try:
            # Using Sentinel-5P data for atmospheric measurements
            # Free tier access with registration
            headers = {
                "Content-Type": "application/json"
            }
            response = requests.post(self.base_url, headers=headers)
            data = response.json()
            
            return AtmosphericData(
                timestamp=datetime.now(),
                location={"latitude": 12.9716, "longitude": 77.5946},
                temperature=data["temperature"],
                humidity=data["humidity"],
                pressure=data["pressure"],
                wind_speed=data["wind_speed"],
                wind_direction=data["wind_direction"],
                confidence=self.confidence
            )
        except Exception as e:
            logger.error(f"Sentinel fetch error: {e}")
            return None

class GOESSource(WeatherDataSource):
    def __init__(self):
        super().__init__("GOES", 0.90)
        self.base_url = "https://www.ncei.noaa.gov/access/services/data/v1"
        
    def fetch(self) -> Optional[AtmosphericData]:
        try:
            params = {
                "dataset": "global-hourly",
                "stations": "VOBG",
                "startDate": datetime.now().strftime("%Y-%m-%d"),
                "endDate": datetime.now().strftime("%Y-%m-%d"),
                "format": "json"
            }
            response = requests.get(self.base_url, params=params)
            data = response.json()[-1]
            
            return AtmosphericData(
                timestamp=datetime.now(),
                location={"latitude": 12.9716, "longitude": 77.5946},
                temperature=float(data["TMP"]),
                humidity=float(data["RH"]),
                pressure=float(data["SLP"]),
                wind_speed=float(data["WND_SPD"]),
                wind_direction=float(data["WND_DIR"]),
                confidence=self.confidence
            )
        except Exception as e:
            logger.error(f"GOES fetch error: {e}")
            return None

# Add these raw data sources

class IMDSource(WeatherDataSource):
    def __init__(self):
        super().__init__("IMD", 0.92)  # India Meteorological Department
        self.base_url = "https://api.imd.gov.in/weather/current"
        self.station_id = "VOBG"  # Bengaluru Airport Station
        
    def fetch(self) -> Optional[AtmosphericData]:
        try:
            params = {
                "station": self.station_id,
                "format": "json"
            }
            response = requests.get(self.base_url, params=params)
            data = response.json()
            
            return AtmosphericData(
                timestamp=datetime.now(),
                location={"latitude": 12.9716, "longitude": 77.5946},
                temperature=data["temperature"],
                humidity=data["relative_humidity"],
                pressure=data["pressure"],
                wind_speed=data["wind_speed"],
                wind_direction=data["wind_direction"],
                confidence=self.confidence
            )
        except Exception as e:
            logger.error(f"IMD fetch error: {e}")
            return None

class COPERNICUSSource(WeatherDataSource):
    def __init__(self):
        super().__init__("COPERNICUS", 0.95)
        self.base_url = "https://cds.climate.copernicus.eu/api/v2"
        
    def fetch(self) -> Optional[AtmosphericData]:
        try:
            params = {
                "product_type": "reanalysis",
                "format": "json",
                "variable": ["2m_temperature", "relative_humidity", "surface_pressure"],
                "area": [13.0, 77.5, 12.9, 78.0],  # Bengaluru region
            }
            response = requests.get(self.base_url, params=params)
            data = response.json()
            
            return AtmosphericData(
                timestamp=datetime.now(),
                location={"latitude": 12.9716, "longitude": 77.5946},
                temperature=data["t2m"] - 273.15,  # Convert from Kelvin
                humidity=data["r"],
                pressure=data["sp"]/100,  # Convert to hPa
                wind_speed=data["wind"],
                wind_direction=data["wind_direction"],
                confidence=self.confidence
            )
        except Exception as e:
            logger.error(f"COPERNICUS fetch error: {e}")
            return None

# Update MultiSourceFetcher to use only working sources for now
class MultiSourceFetcher:
    def __init__(self):
        self.sources = [
            OpenMeteoSource(),  # Working correctly
            NOAASource(),       # Free, no API key needed
            GOESSource()        # Free, no API key needed
        ]
    
    def fetch_weather_data(self) -> Optional[AtmosphericData]:
        valid_data = []
        
        for source in self.sources:
            data = source.fetch()
            if data:
                valid_data.append(data)
                logger.debug(f"Successfully fetched data from {source.name}")
            
        if not valid_data:
            logger.error("No valid data from any source")
            return None
            
        # Weighted average based on confidence scores
        total_weight = sum(d.confidence for d in valid_data)
        temperature = sum(d.temperature * d.confidence for d in valid_data) / total_weight
        humidity = sum(d.humidity * d.confidence for d in valid_data) / total_weight
        pressure = sum(d.pressure * d.confidence for d in valid_data) / total_weight
        wind_speed = sum(d.wind_speed * d.confidence for d in valid_data) / total_weight
        wind_direction = statistics.median(d.wind_direction for d in valid_data)
        
        return AtmosphericData(
            timestamp=datetime.now(),
            location={"latitude": 12.9716, "longitude": 77.5946},
            temperature=temperature,
            humidity=humidity,
            pressure=pressure,
            wind_speed=wind_speed,
            wind_direction=wind_direction,
            confidence=min(0.95, total_weight/len(valid_data))
        )

    def get_weather_prediction(self, hours_ahead: int = 24) -> dict:
        """Get weather predictions for the next specified hours"""
        try:
            current_hour = datetime.now().replace(minute=0, second=0, microsecond=0)
            
            url = "https://api.open-meteo.com/v1/forecast"
            params = {
                "latitude": 12.9716,
                "longitude": 77.5946,
                "hourly": ["temperature_2m", "relative_humidity_2m", 
                          "pressure_msl", "wind_speed_10m", "wind_direction_10m"],
                "timezone": "Asia/Kolkata",  # Set correct timezone for Bangalore
                "start_date": current_hour.strftime("%Y-%m-%d"),
                "end_date": (current_hour + timedelta(days=2)).strftime("%Y-%m-%d")
            }
            response = requests.get(url, params=params)
            data = response.json()
            
            # Calculate the current hour index
            current_time_str = current_hour.strftime("%Y-%m-%dT%H:00")
            start_index = data["hourly"]["time"].index(current_time_str)
            
            return {
                "hourly_forecast": [
                    {
                        "time": data["hourly"]["time"][i],
                        "temperature": data["hourly"]["temperature_2m"][i],
                        "humidity": data["hourly"]["relative_humidity_2m"][i],
                        "pressure": data["hourly"]["pressure_msl"][i],
                        "wind_speed": data["hourly"]["wind_speed_10m"][i],
                        "wind_direction": data["hourly"]["wind_direction_10m"][i]
                    }
                    for i in range(start_index, min(start_index + hours_ahead, len(data["hourly"]["time"])))
                ]
            }
        except Exception as e:
            logger.error(f"Prediction fetch error: {e}")