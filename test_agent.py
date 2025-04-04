from src.data.multi_source_fetcher import MultiSourceFetcher
from datetime import datetime, timedelta
import logging

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_weather_agent():
    fetcher = MultiSourceFetcher()
    print("\nFetching current weather data...")
    
    result = fetcher.fetch_weather_data()
    
    if result:
        print("\nCurrent Weather:")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} IST")
        print(f"Temperature: {result.temperature:.1f}°C")
        print(f"Humidity: {result.humidity:.1f}%")
        print(f"Pressure: {result.pressure:.1f} hPa")
        print(f"Wind Speed: {result.wind_speed:.1f} m/s")
        print(f"Wind Direction: {result.wind_direction:.1f}°")
        print(f"Confidence: {result.confidence:.2f}")
        
        print("\nFetching 24-hour prediction...")
        prediction = fetcher.get_weather_prediction(24)
        if prediction:
            print("\nHourly Predictions (next 6 hours):")
            for hour in prediction["hourly_forecast"][:6]:
                local_time = datetime.fromisoformat(hour['time']).strftime('%Y-%m-%d %H:%M IST')
                print(f"\nTime: {local_time}")
                print(f"Temperature: {hour['temperature']:.1f}°C")
                print(f"Humidity: {hour['humidity']:.1f}%")
                print(f"Wind Speed: {hour['wind_speed']:.1f} m/s")
    else:
        print("Failed to fetch weather data")

if __name__ == "__main__":
    test_weather_agent()