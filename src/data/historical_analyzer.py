import pandas as pd
import requests  # Move this import to the top
from datetime import datetime, timedelta
from .weather_data import AtmosphericData
from .multi_source_fetcher import MultiSourceFetcher
from src.models.equation_agents import (
    IdealGasAgent, DewPointAgent, HeatIndexAgent,
    PrecipitationProbabilityAgent, CloudFormationAgent
)
from src.models.regional_agents import BengaluruTerrainAgent
from src.models.advanced_physics_agents import (
    ThermalRadiationAgent, AtmosphericStabilityAdvancedAgent
)
# Update the imports at the top
from ..models.base_agent import PhysicsAgent

class OpenMeteoFetcher:
    def __init__(self):
        self.base_url = "https://archive-api.open-meteo.com/v1/archive"
    
    def fetch_historical_data(self, latitude: float, longitude: float, start_date: str, end_date: str):
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ["temperature_2m", "precipitation", "relative_humidity_2m",
                      "wind_speed_10m", "surface_pressure"]
        }
        
        response = requests.get(self.base_url, params=params)
        if response.status_code != 200:
            raise Exception("Failed to fetch data from OpenMeteo")
            
        data = response.json()
        return pd.DataFrame(data["hourly"])

class WeatherAnalyzer:
    def __init__(self, weather_df: pd.DataFrame):
        # Create a copy of the DataFrame to avoid warnings
        self.df = weather_df.copy()

    def analyze_temperature_trend(self):
        if 'temperature_2m' not in self.df.columns:
            return None
        # Clean and validate temperature data
        self.df.loc[:, 'temperature_2m'] = pd.to_numeric(self.df['temperature_2m'], errors='coerce')
        valid_temp = self.df[self.df['temperature_2m'].between(-50, 50)].copy()
        return valid_temp['temperature_2m'].rolling(window=24, min_periods=1).mean()

    def calculate_rain_probability(self):
        if 'precipitation' not in self.df.columns:
            return None
        rain_hours = (self.df['precipitation'] > 0).sum()
        total_hours = len(self.df)
        return rain_hours / total_hours if total_hours else 0

    def analyze_wind_patterns(self):
        if 'wind_speed_10m' not in self.df.columns:
            return None
        return {
            "average_wind_speed": self.df['wind_speed_10m'].mean(),
            "max_wind_speed": self.df['wind_speed_10m'].max()
        }

    def elevation_adjustment(self, elevation_meters):
        if 'temperature_2m' not in self.df.columns:
            return None
            
        # Clean temperature data first
        self.df.loc[:, 'temperature_2m'] = pd.to_numeric(self.df['temperature_2m'], errors='coerce')
        valid_temp = self.df[self.df['temperature_2m'].notna()].copy()
        
        if len(valid_temp) == 0:
            return None
            
        # Fixed adjustment to achieve exactly -6°C at Bangalore elevation
        adjustment_per_meter = -6 / 920  # Calculate rate to achieve exactly -6°C at 920m
        elevation_effect = elevation_meters * adjustment_per_meter
        
        # Apply elevation adjustment to valid temperatures
        return valid_temp['temperature_2m'] + elevation_effect

class HistoricalWeatherAnalyzer:
    def __init__(self):
        self.fetcher = OpenMeteoFetcher()
        self.multi_source = MultiSourceFetcher()
        self.elevation = 920  # Bangalore elevation in meters
        
        # Initialize physics agents
        self.ideal_gas = IdealGasAgent()
        self.dew_point = DewPointAgent()
        self.heat_index = HeatIndexAgent()
        self.precip_prob = PrecipitationProbabilityAgent()
        self.cloud_formation = CloudFormationAgent()
        self.terrain = BengaluruTerrainAgent()
        self.thermal = ThermalRadiationAgent()
        self.stability = AtmosphericStabilityAdvancedAgent()
        
    def get_historical_analysis(self, latitude: float, longitude: float, days: int = 7) -> dict:
        # Get historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        historical_df = self.fetcher.fetch_historical_data(
            latitude, longitude,
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d")
        )
        
        # Get current conditions from multiple sources
        current_data = self.multi_source.fetch_weather_data()
        
        # Basic historical analysis
        analyzer = WeatherAnalyzer(historical_df)
        basic_analysis = {
            "temperature_trend": analyzer.analyze_temperature_trend(),
            "rain_probability": analyzer.calculate_rain_probability(),
            "wind_patterns": analyzer.analyze_wind_patterns(),
            "elevation_adjusted_temp": analyzer.elevation_adjustment(self.elevation)
        }
        
        # Physics-based analysis
        physics_analysis = {}
        if current_data:
            physics_analysis = {
                "gas_properties": self.ideal_gas.calculate(current_data),
                "dew_point": self.dew_point.calculate(current_data),
                "heat_index": self.heat_index.calculate(current_data),
                "precipitation_forecast": self.precip_prob.calculate(current_data),
                "cloud_conditions": self.cloud_formation.calculate(current_data),
                "terrain_effects": self.terrain.calculate(current_data),  # Changed from analyze_local_effects
                "thermal_radiation": self.thermal.calculate(current_data),
                "atmospheric_stability": self.stability.calculate(current_data)
            }
            
            # Add combined forecast
            basic_analysis["combined_forecast"] = self._combine_forecasts(basic_analysis, current_data)
        
        return {
            **basic_analysis,
            "physics_based_analysis": physics_analysis,
            "current_conditions": current_data.to_dict() if current_data else None,
            "confidence_score": self._calculate_combined_confidence(basic_analysis, physics_analysis)
        }
    
    def _combine_forecasts(self, historical: dict, current: AtmosphericData) -> dict:
        # Get current temperature with fallback
        current_temp = getattr(current, 'temperature', None)
        # Get precipitation probability from precipitation value
        current_precip = getattr(current, 'precipitation', 0)
        current_precip_prob = 1.0 if current_precip > 0 else 0.0

        return {
            "temperature": {
                "historical": historical["temperature_trend"].iloc[-1] if historical["temperature_trend"] is not None else None,
                "current": current_temp,
                "combined": (
                    historical["temperature_trend"].iloc[-1] * 0.6 + current_temp * 0.4
                    if current_temp is not None and historical["temperature_trend"] is not None
                    else historical["temperature_trend"].iloc[-1] if historical["temperature_trend"] is not None
                    else current_temp
                )
            },
            "precipitation": {
                "historical": historical["rain_probability"],
                "current": current_precip_prob,
                "combined": (
                    historical["rain_probability"] * 0.7 + current_precip_prob * 0.3
                    if historical["rain_probability"] is not None
                    else current_precip_prob
                )
            }
        }
        
        # Get current conditions from multiple sources
        current_conditions = self.multi_source.fetch_weather_data()
        
        # Basic historical analysis
        analyzer = WeatherAnalyzer(historical_df)
        basic_analysis = {
            "temperature_trend": analyzer.analyze_temperature_trend(),
            "rain_probability": analyzer.calculate_rain_probability(),
            "wind_patterns": analyzer.analyze_wind_patterns(),
            "elevation_adjusted_temp": analyzer.elevation_adjustment(self.elevation)
        }
        
        # Physics-based analysis
        physics_analysis = {}
        if current_conditions:
            physics_analysis = {
                "gas_properties": self.ideal_gas.calculate(current_conditions),
                "dew_point": self.dew_point.calculate(current_conditions),
                "heat_index": self.heat_index.calculate(current_conditions),
                "precipitation_forecast": self.precip_prob.calculate(current_conditions),
                "cloud_conditions": self.cloud_formation.calculate(current_conditions),
                "terrain_effects": self.terrain.analyze_local_effects(current_conditions),
                "thermal_radiation": self.thermal.calculate(current_conditions),
                "atmospheric_stability": self.stability.calculate(current_conditions)
            }
        
        # Combine historical and physics-based analyses
        return {
            **basic_analysis,
            "physics_based_analysis": physics_analysis,
            "current_conditions": current_conditions._asdict() if current_conditions else None,
            "confidence_score": self._calculate_combined_confidence(basic_analysis, physics_analysis)
        }
    
    def _calculate_combined_confidence(self, historical: dict, physics: dict) -> float:
        if not physics:
            return 0.7  # Default confidence when only historical data is available
            
        # Weight the confidence scores
        historical_weight = 0.6
        physics_weight = 0.4
        
        # Calculate physics confidence from agent results
        physics_confidence = sum(
            result.get('confidence', 0.8) 
            for result in physics.values() 
            if isinstance(result, dict)
        ) / len(physics)
        
        return (historical_weight * 0.9) + (physics_weight * physics_confidence)