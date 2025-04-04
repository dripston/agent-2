from datetime import datetime
from typing import List, Dict
from ..data.weather_data import AtmosphericData, WeatherPrediction
from .equation_agents import (
    IdealGasAgent, 
    AdibaticProcessAgent, 
    DewPointAgent,
    VaporPressureAgent,
    HeatIndexAgent,
    WindShearAgent,
    AtmosphericStabilityAgent,
    PrecipitationProbabilityAgent,
    SolarRadiationAgent,
    CloudFormationAgent,
    TurbulenceAgent
)

class AgentOrchestrator:
    def __init__(self):
        self.agents = [
            IdealGasAgent(),
            AdibaticProcessAgent(),
            DewPointAgent(),
            VaporPressureAgent(),
            HeatIndexAgent(),
            WindShearAgent(),
            AtmosphericStabilityAgent(),
            PrecipitationProbabilityAgent(),
            SolarRadiationAgent(),
            CloudFormationAgent(),
            TurbulenceAgent()
        ]
        
    def collect_predictions(self, data: AtmosphericData) -> Dict[str, List[float]]:
        predictions = {
            "temperature": [],
            "pressure": [],
            "humidity": [],
            "wind_speed": [],
            "density": [],
            "precipitation_prob": []
        }
        
        parameter_mapping = {
            "temperature": ["temperature", "dew_point", "feels_like_temperature"],
            "pressure": ["pressure", "vapor_pressure"],
            "density": ["air_density"],
            "wind_speed": ["wind_speed"],
            "precipitation_prob": ["precipitation_prob"]
        }
        
        for agent in self.agents:
            result = agent.calculate(data)
            param_type = result["parameter"]
            
            # Find the actual value in the result
            for key, value in result.items():
                if key != "parameter":
                    # Map the result to the correct prediction category
                    for pred_key, possible_params in parameter_mapping.items():
                        if key in possible_params:
                            predictions[pred_key].append(value)
        
        return predictions
    
    def generate_forecast(self, data: AtmosphericData) -> WeatherPrediction:
        predictions = self.collect_predictions(data)
        
        # Calculate average for each parameter if predictions exist
        temp = sum(predictions["temperature"]) / len(predictions["temperature"]) if predictions["temperature"] else data.temperature
        pressure = sum(predictions["pressure"]) / len(predictions["pressure"]) if predictions["pressure"] else data.pressure
        
        # Ensure pressure stays within reasonable bounds
        pressure = max(900, min(1100, pressure))
        
        prediction_data = AtmosphericData(
            timestamp=datetime.now(),
            location=data.location,
            temperature=temp,
            humidity=data.humidity,
            pressure=pressure,
            wind_speed=data.wind_speed,
            wind_direction=data.wind_direction,
            confidence=0.85
        )
        
        return WeatherPrediction(
            location=data.location,
            forecast_time=datetime.now(),
            prediction_data=prediction_data,
            confidence_score=0.85,
            source_agent="physics_multi_agent"
        )