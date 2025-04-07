from src.data.historical_analyzer import HistoricalWeatherAnalyzer
from datetime import datetime

def test_integrated_weather():
    analyzer = HistoricalWeatherAnalyzer()
    
    # Bangalore coordinates
    location = {
        'latitude': 12.9716,
        'longitude': 77.5946
    }
    
    print("\nFetching integrated weather analysis...")
    analysis = analyzer.get_historical_analysis(
        location['latitude'],
        location['longitude'],
        days=7
    )
    
    print("\n=== Weather Analysis Results ===")
    print("\n1. Historical Analysis:")
    print(f"- Average Temperature Trend: {analysis['temperature_trend'].mean():.1f}°C")
    print(f"- Rain Probability: {analysis['rain_probability']*100:.1f}%")
    print(f"- Wind Patterns: {analysis['wind_patterns']}")
    print(f"- Elevation Adjusted Temp: {analysis['elevation_adjusted_temp'].mean():.1f}°C")
    
    # Print physics-based analysis results
    if "physics_based_analysis" in analysis:
        physics = analysis["physics_based_analysis"]
        print("\n2. Physics-Based Analysis:")
        
        # Handle dew point
        dew_point = physics.get('dew_point', {}).get('dew_point', 'N/A')
        print(f"- Dew Point: {dew_point if isinstance(dew_point, str) else f'{dew_point:.1f}°C'}")
        
        # Handle heat index
        heat_index = physics.get('heat_index', {}).get('heat_index', 'N/A')
        print(f"- Heat Index: {heat_index if isinstance(heat_index, str) else f'{heat_index:.1f}°C'}")
        
        # Handle precipitation forecast
        precip_prob = physics.get('precipitation_forecast', {}).get('probability', 'N/A')
        print(f"- Precipitation Forecast: {precip_prob if isinstance(precip_prob, str) else f'{precip_prob*100:.1f}%'}")
        
        # Handle cloud formation probability
        cloud_prob = physics.get('cloud_conditions', {}).get('formation_probability', 'N/A')
        print(f"- Cloud Formation Probability: {cloud_prob if isinstance(cloud_prob, str) else f'{cloud_prob*100:.1f}%'}")
        
        print(f"- Atmospheric Stability: {physics['atmospheric_stability'].get('stability_index', 'N/A')}")
    
    print(f"\nOverall Confidence Score: {analysis['confidence_score']:.2f}")
    
    if analysis.get('current_conditions'):
        print("\n3. Current Conditions:")
        current = analysis['current_conditions']
        print(f"- Temperature: {current['temperature']:.1f}°C")
        print(f"- Humidity: {current['humidity']:.1f}%")
        print(f"- Pressure: {current['pressure']:.1f} hPa")
        print(f"- Wind Speed: {current['wind_speed']:.1f} m/s")

if __name__ == "__main__":
    test_integrated_weather()