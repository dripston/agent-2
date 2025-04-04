import numpy as np
from .base_agent import PhysicsAgent
from ..data.weather_data import AtmosphericData

class ThermalRadiationAgent(PhysicsAgent):
    def __init__(self):
        super().__init__("thermal_radiation")
        self.stefan_boltzmann = 5.67e-8
        self.albedo = 0.3  # Bengaluru's average albedo
        
    def calculate(self, data: AtmosphericData) -> Dict[str, float]:
        temp_k = data.temperature + 273.15
        cloud_factor = min(data.humidity / 100, 0.9)
        net_radiation = (1 - self.albedo) * self.stefan_boltzmann * (temp_k**4)
        net_radiation *= (1 - 0.75 * cloud_factor)  # Cloud effect
        return {
            "net_radiation": net_radiation,
            "surface_heating": net_radiation * 0.3,
            "parameter": "radiation"
        }

class BoundaryLayerAgent(PhysicsAgent):
    def __init__(self):
        super().__init__("boundary_layer")
        self.von_karman = 0.41
        self.roughness_length = 0.1  # Urban area
        
    def calculate(self, data: AtmosphericData) -> Dict[str, float]:
        height = 100  # meters
        u_star = self.von_karman * data.wind_speed / np.log(height/self.roughness_length)
        mixing_height = 0.4 * u_star / (7.2921e-5 * np.sin(np.radians(12.9716)))
        return {
            "friction_velocity": u_star,
            "mixing_height": mixing_height,
            "parameter": "boundary_layer"
        }

class OrographicLiftAgent(PhysicsAgent):
    def __init__(self):
        super().__init__("orographic_lift")
        self.terrain_gradient = 0.05  # Bengaluru's average slope
        
    def calculate(self, data: AtmosphericData) -> Dict[str, float]:
        vertical_velocity = data.wind_speed * self.terrain_gradient
        temp_change = -0.0098 * vertical_velocity  # Adiabatic lapse rate
        return {
            "vertical_velocity": vertical_velocity,
            "temperature_change": temp_change,
            "parameter": "vertical_motion"
        }

class AtmosphericStabilityAdvancedAgent(PhysicsAgent):
    def __init__(self):
        super().__init__("atmospheric_stability_advanced")
        
    def calculate(self, data: AtmosphericData) -> Dict[str, float]:
        temp_k = data.temperature + 273.15
        potential_temp = temp_k * (1000/data.pressure)**0.286
        brunt_vaisala = np.sqrt(9.81/temp_k * 0.0098)  # Static stability
        return {
            "potential_temperature": potential_temp,
            "static_stability": brunt_vaisala,
            "parameter": "stability"
        }

class ConvectiveIndexAgent(PhysicsAgent):
    def __init__(self):
        super().__init__("convective_index")
        
    def calculate(self, data: AtmosphericData) -> Dict[str, float]:
        temp_k = data.temperature + 273.15
        e_sat = 6.112 * np.exp(17.67 * data.temperature/(data.temperature + 243.5))
        theta_e = temp_k + 2.5e6 * (data.humidity/100 * e_sat)/(1005 * temp_k)
        return {
            "equivalent_potential_temp": theta_e,
            "convective_potential": max(0, (theta_e - temp_k)/temp_k),
            "parameter": "convection"
        }