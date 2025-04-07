from src.data.multi_source_fetcher import MultiSourceFetcher
from src.models.equation_agents import (
    IdealGasAgent, DewPointAgent, HeatIndexAgent,
    PrecipitationProbabilityAgent, CloudFormationAgent
)
from src.models.regional_agents import BengaluruTerrainAgent, MonsoonDynamicsAgent, CoriolisEffectAgent
from src.models.advanced_physics_agents import (
    ThermalRadiationAgent, BoundaryLayerAgent,
    AtmosphericStabilityAdvancedAgent, OrographicLiftAgent,
    ConvectiveIndexAgent  # Added ConvectiveIndexAgent here
)
from datetime import datetime, timedelta

def test_weather_system():
    # 1. Fetch current weather
    # Replace the existing fetcher initialization with:
    fetcher = MultiSourceFetcher()
    weather_data = fetcher.fetch_weather_data()
    
    if not weather_data:
        print("Failed to fetch weather data")
        return
    
    current_time = weather_data.timestamp
    
    # Print current conditions
    print(f"\nCurrent Weather in Bengaluru (as of {current_time.strftime('%Y-%m-%d %H:%M')})")
    print(f"Temperature: {weather_data.temperature}°C")
    print(f"Humidity: {weather_data.humidity}%")
    print(f"Pressure: {weather_data.pressure} hPa")
    print(f"Wind Speed: {weather_data.wind_speed} m/s")
    print(f"Wind Direction: {weather_data.wind_direction}°")
    
    # 2. Run physics-based predictions
    print("\nPhysics-Based Predictions:")
    
    # Immediate conditions (next hour)
    next_hour = current_time + timedelta(hours=1)
    print(f"\nPredictions for {next_hour.strftime('%H:%M')}:")
    
    gas_agent = IdealGasAgent()
    density_result = gas_agent.calculate(weather_data)
    print(f"Air Density: {density_result['air_density']:.2f} kg/m³")
    
    dew_agent = DewPointAgent()
    dew_result = dew_agent.calculate(weather_data)
    print(f"Dew Point: {dew_result['dew_point']:.1f}°C")
    
    heat_agent = HeatIndexAgent()
    heat_result = heat_agent.calculate(weather_data)
    print(f"Feels Like: {heat_result['feels_like_temperature']:.1f}°C")
    
    # Short-term predictions (next 3 hours)
    next_three_hours = current_time + timedelta(hours=3)
    print(f"\nPredictions for {next_three_hours.strftime('%H:%M')}:")
    
    precip_agent = PrecipitationProbabilityAgent()
    precip_result = precip_agent.calculate(weather_data)
    print(f"Precipitation Probability: {precip_result['precipitation_prob']*100:.1f}%")
    
    cloud_agent = CloudFormationAgent()
    cloud_result = cloud_agent.calculate(weather_data)
    print(f"Cloud Formation Pressure: {cloud_result['pressure']:.1f} hPa")

    # Add after existing agents
    terrain_agent = BengaluruTerrainAgent()
    terrain_result = terrain_agent.calculate(weather_data)
    print(f"Terrain-Adjusted Temperature: {terrain_result['temperature']:.1f}°C")
    
    monsoon_agent = MonsoonDynamicsAgent()
    monsoon_result = monsoon_agent.calculate(weather_data)
    print(f"Monsoon Intensity: {monsoon_result['monsoon_intensity']:.2f}")
    print(f"Precipitation Modified by: {monsoon_result['precipitation_modifier']:.2f}x")
    
    coriolis_agent = CoriolisEffectAgent()
    coriolis_result = coriolis_agent.calculate(weather_data)
    print(f"Wind Deviation due to Coriolis: {coriolis_result['wind_deviation']:.2f} m/s")

    # Add after existing predictions
    print("\nAdvanced Physics Calculations:")
    
    radiation_agent = ThermalRadiationAgent()
    rad_result = radiation_agent.calculate(weather_data)
    print(f"Net Radiation: {rad_result['net_radiation']/1000:.2f} kW/m²")
    print(f"Surface Heating: {rad_result['surface_heating']/1000:.2f} kW/m²")
    
    boundary_agent = BoundaryLayerAgent()
    boundary_result = boundary_agent.calculate(weather_data)
    print(f"Mixing Height: {boundary_result['mixing_height']:.0f}m")
    print(f"Friction Velocity: {boundary_result['friction_velocity']:.2f} m/s")
    
    oro_agent = OrographicLiftAgent()
    oro_result = oro_agent.calculate(weather_data)
    print(f"Vertical Motion: {oro_result['vertical_velocity']:.2f} m/s")
    print(f"Orographic Temperature Effect: {oro_result['temperature_change']:.2f}°C")
    
    stability_agent = AtmosphericStabilityAdvancedAgent()
    stab_result = stability_agent.calculate(weather_data)
    print(f"Static Stability: {stab_result['static_stability']:.4f} /s")
    print(f"Potential Temperature: {stab_result['potential_temperature']:.1f}K")
    
    convective_agent = ConvectiveIndexAgent()
    conv_result = convective_agent.calculate(weather_data)
    print(f"Convective Potential: {conv_result['convective_potential']:.3f}")

if __name__ == "__main__":
    test_weather_system()