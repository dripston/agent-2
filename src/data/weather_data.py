from dataclasses import dataclass
from datetime import datetime
from typing import Dict

@dataclass
class AtmosphericData:
    timestamp: datetime
    location: Dict[str, float]
    temperature: float
    humidity: float
    pressure: float
    wind_speed: float
    wind_direction: float
    precipitation: float = 0.0  # Added with default value of 0.0
    confidence: float = 1.0

    def to_dict(self) -> dict:
        return {
            "temperature": self.temperature,
            "humidity": self.humidity,
            "pressure": self.pressure,
            "wind_speed": self.wind_speed,
            "wind_direction": self.wind_direction,
            "precipitation": self.precipitation
        }

@dataclass
class WeatherPrediction:
    location: Dict[str, float]
    forecast_time: datetime
    prediction_data: AtmosphericData
    confidence_score: float
    source_agent: str