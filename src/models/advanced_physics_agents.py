from typing import Dict
import numpy as np
from .base_agent import PhysicsAgent
from ..data.weather_data import AtmosphericData

class ThermalRadiationAgent(PhysicsAgent):
    def __init__(self):
        super().__init__("thermal_radiation")
        self.stefan_boltzmann = 5.67e-8
        
    def calculate(self, data: AtmosphericData) -> Dict[str, float]:
        # Convert temperature to Kelvin
        T = data.temperature + 273.15
        
        # Calculate net radiation (simplified model)
        net_radiation = self.stefan_boltzmann * (T**4)
        surface_heating = net_radiation * 0.7  # 70% absorption
        
        return {
            'net_radiation': net_radiation,
            'surface_heating': surface_heating,
            'confidence': 0.85
        }

class AtmosphericStabilityAdvancedAgent(PhysicsAgent):
    def __init__(self):
        super().__init__("atmospheric_stability")
        self.elevation = 920  # Bangalore elevation in meters
        
    def calculate(self, data: AtmosphericData) -> Dict[str, float]:
        # Calculate stability using temperature, pressure and wind
        temp_gradient = -0.0065  # Standard atmosphere lapse rate
        stability_index = (
            temp_gradient * self.elevation / 1000 +
            data.wind_speed / 10 +
            (data.pressure - 1013.25) / 1013.25
        )
        
        # Calculate static stability and potential temperature
        static_stability = -temp_gradient / (data.temperature + 273.15)
        potential_temp = (data.temperature + 273.15) * (1000/data.pressure)**0.286
        
        return {
            "stability_index": stability_index,
            "stability_class": self._get_stability_class(stability_index),
            "static_stability": static_stability,
            "potential_temperature": potential_temp,
            "confidence": 0.8
        }
        
    def _get_stability_class(self, index: float) -> str:
        if index < -0.5:
            return "unstable"
        elif index > 0.5:
            return "stable"
        return "neutral"


class BoundaryLayerAgent(PhysicsAgent):
    def __init__(self):
        super().__init__("boundary_layer")
        self.von_karman = 0.41  # von Karman constant
        self.surface_roughness = 0.1  # typical urban roughness length in meters
        
    def calculate(self, data: AtmosphericData) -> Dict[str, float]:
        # Calculate mixing height using a simplified model
        mixing_height = 1000.0  # Base mixing height in meters
        if data.wind_speed > 0:
            mixing_height *= (1 + data.wind_speed / 5)  # Adjust for wind speed
            
        # Calculate friction velocity using log-law
        friction_velocity = (data.wind_speed * self.von_karman) / \
                          np.log(10 / self.surface_roughness)  # 10m is measurement height
                          
        return {
            'mixing_height': mixing_height,
            'friction_velocity': friction_velocity,
            'confidence': 0.75
        }

class OrographicLiftAgent(PhysicsAgent):
    def __init__(self):
        super().__init__("orographic_lift")
        self.terrain_gradient = 0.05  # Average slope for Bangalore
        self.specific_heat = 1005.0  # Specific heat of air (J/kg·K)
        
    def calculate(self, data: AtmosphericData) -> Dict[str, float]:
        # Calculate vertical velocity due to terrain
        vertical_velocity = data.wind_speed * self.terrain_gradient
        
        # Calculate temperature change due to lifting
        temperature_change = -(vertical_velocity * self.specific_heat) / 9.81
        
        return {
            'vertical_velocity': vertical_velocity,
            'temperature_change': temperature_change,
            'confidence': 0.7
        }


class ConvectiveIndexAgent(PhysicsAgent):
    def __init__(self):
        super().__init__("convective_index")
        self.g = 9.81  # gravitational acceleration
        self.cp = 1005.0  # specific heat at constant pressure
        
    def calculate(self, data: AtmosphericData) -> Dict[str, float]:
        # Calculate convective potential using temperature and humidity
        T = data.temperature + 273.15  # Convert to Kelvin
        
        # Simple CAPE-like index based on temperature and humidity
        convective_energy = (
            (T - 273.15) * 0.1 +  # Temperature contribution
            (data.humidity / 100) * 0.2 +  # Humidity contribution
            (1013.25 - data.pressure) / 100  # Pressure contribution
        )
        
        # Normalize to 0-1 scale
        convective_potential = max(0, min(1, convective_energy / 10))
        
        return {
            'convective_potential': convective_potential,
            'convective_energy': convective_energy,
            'confidence': 0.75
        }