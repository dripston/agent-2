import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
from src.models.bangalore_weather_system import BangaloreWeatherSystem

async def test_bangalore_weather():
    print("Initializing Bangalore Weather System...")
    weather_system = BangaloreWeatherSystem()
    
    # Test locations (Indiranagar and Koramangala)
    test_locations = [
        (12.9719, 77.6412, "Indiranagar"),
        (12.9279, 77.6271, "Koramangala")
    ]
    
    for lat, lon, area in test_locations:
        print(f"\nGenerating forecast for {area}...")
        forecast = await weather_system.get_forecast(lat, lon)
        
        print("\n=== Weather Forecast ===")
        print(f"Location: {forecast.location.ward_name}")
        print(f"Temperature: {forecast.temperature:.1f}°C")
        print(f"Humidity: {forecast.humidity:.1f}%")
        print(f"Precipitation Probability: {forecast.precipitation_prob*100:.1f}%")
        print(f"Wind Speed: {forecast.wind_speed:.1f} m/s")
        print(f"Confidence: {forecast.confidence*100:.1f}%")
        print(f"Explanation: {forecast.explanation}")

if __name__ == "__main__":
    asyncio.run(test_bangalore_weather())