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
    confidence: float = 1.0

@dataclass
class WeatherPrediction:
    location: Dict[str, float]
    forecast_time: datetime
    prediction_data: AtmosphericData
    confidence_score: float
    source_agent: str