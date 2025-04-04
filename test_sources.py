from src.data.multi_source_fetcher import MultiSourceFetcher
import logging

# Set up logging to see debug messages
logging.basicConfig(level=logging.DEBUG)

def test_sources():
    fetcher = MultiSourceFetcher()
    data = fetcher.fetch_weather_data()
    
    if data:
        print("\nSuccessfully fetched weather data:")
        print(f"Temperature: {data.temperature:.1f}°C")
        print(f"Humidity: {data.humidity:.1f}%")
        print(f"Pressure: {data.pressure:.1f} hPa")
        print(f"Wind Speed: {data.wind_speed:.1f} m/s")
        print(f"Wind Direction: {data.wind_direction:.1f}°")
        print(f"Confidence: {data.confidence:.2f}")
    else:
        print("Failed to fetch weather data")

if __name__ == "__main__":
    test_sources()