from typing import Dict, Optional
import numpy as np
from .base_agent import PhysicsAgent
from ..data.weather_data import AtmosphericData

class BengaluruTerrainAgent(PhysicsAgent):
    def __init__(self):
        super().__init__("BengaluruTerrain")
        self.elevation = 920  # Bangalore elevation in meters
        self.urban_heat_factor = 1.5  # Urban heat island effect factor
        
    def calculate(self, data: AtmosphericData) -> Dict[str, float]:
        # Terrain-based temperature adjustment
        elevation_effect = -0.0065 * self.elevation  # standard lapse rate
        urban_effect = self.urban_heat_factor * np.log(data.temperature/20)
        adjusted_temp = data.temperature + elevation_effect + urban_effect
        return {
            "temperature": adjusted_temp,
            "parameter": "temperature",
            "confidence": 0.85
        }

class MonsoonDynamicsAgent(PhysicsAgent):
    def __init__(self):
        super().__init__("monsoon_dynamics")
        
    def calculate(self, data: AtmosphericData) -> Dict[str, float]:
        # Monsoon prediction based on pressure gradients
        pressure_gradient = (data.pressure - 1013.25) / 1013.25
        humidity_factor = data.humidity / 100.0
        
        # Calculate monsoon intensity
        monsoon_strength = pressure_gradient * humidity_factor * \
                          np.exp(-0.1 * abs(data.wind_direction - 225))
        
        return {
            "monsoon_intensity": monsoon_strength,
            "precipitation_modifier": 1 + monsoon_strength,
            "parameter": "precipitation"
        }

class CoriolisEffectAgent(PhysicsAgent):
    def __init__(self):
        super().__init__("coriolis_effect")
        self.latitude = 12.9716  # Bengaluru's latitude
        
    def calculate(self, data: AtmosphericData) -> Dict[str, float]:
        # Coriolis force calculation
        omega = 7.2921e-5  # Earth's angular velocity
        f = 2 * omega * np.sin(np.radians(self.latitude))
        coriolis_wind = f * data.wind_speed
        
        return {
            "wind_deviation": coriolis_wind,
            "parameter": "wind_speed"
        }