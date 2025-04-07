import numpy as np
from src.models.ml_agents import BangaloreMicroZoneAgent
from src.data.weather_data import AtmosphericData
from datetime import datetime, timedelta

def test_ml_predictions():
    ml_agent = BangaloreMicroZoneAgent()
    
    # Test zone-specific predictions
    current_data = AtmosphericData(
        timestamp=datetime.now(),
        location={'latitude': 12.9716, 'longitude': 77.5946},
        temperature=25.0,
        humidity=60.0,
        pressure=1013.25,
        wind_speed=0.0,
        wind_direction=0.0,
        confidence=0.95
    )
    
    # Generate mock historical data
    history = []
    for i in range(24):
        history.append(AtmosphericData(
            timestamp=datetime.now() - timedelta(hours=i),
            location={'latitude': 12.9716, 'longitude': 77.5946},
            temperature=25.0 + np.random.normal(0, 1),
            humidity=60.0 + np.random.normal(0, 5),
            pressure=1013.25 + np.random.normal(0, 0.5),
            wind_speed=np.random.normal(0, 1),
            wind_direction=np.random.normal(0, 45),
            confidence=0.9
        ))
    
    # Test predictions for different zones
    zones = ['indiranagar', 'whitefield', 'koramangala']
    print("\nML-Based Zone Predictions:")
    
    for zone in zones:
        prediction = ml_agent.predict_zone_weather(zone, current_data, history)
        print(f"\n{zone.title()} Forecast:")
        print(f"Temperature: {prediction['temperature']:.1f}°C")
        print(f"Humidity: {prediction['humidity']:.1f}%")
        print(f"Confidence: {prediction['confidence']:.2f}")
        
        # Analyze patterns
        patterns = ml_agent.analyze_zone_patterns(history)
        print(f"Pattern Clusters: {len(set(patterns['clusters']))}")
        print(f"Anomalies Detected: {patterns['anomaly_count']}")

if __name__ == "__main__":
    test_ml_predictions()