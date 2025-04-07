import unittest
from datetime import datetime, timedelta
from src.data.weather_data import AtmosphericData
from src.models.ml_agents import BangaloreMicroZoneAgent
from src.models.equation_agents import ZoneAwareDewPointAgent
from src.data.historical_analyzer import HistoricalWeatherAnalyzer  # Add historical analyzer

class TestZoneWeather(unittest.TestCase):
    def setUp(self):
        self.predictor = BangaloreMicroZoneAgent()
        self.historical_analyzer = HistoricalWeatherAnalyzer()  # Initialize historical analyzer
        
        # Create current test data
        self.test_data = AtmosphericData(
            timestamp=datetime.now(),
            location={'latitude': 12.9716, 'longitude': 77.5946},
            temperature=28.0,
            humidity=65.0,
            pressure=1013.0,
            wind_speed=5.0,
            wind_direction=180.0
        )
        
        # Create historical data for last 24 hours
        self.history = []
        base_time = datetime.now() - timedelta(days=1)
        for i in range(24):
            self.history.append(AtmosphericData(
                timestamp=base_time + timedelta(hours=i),
                location={'latitude': 12.9716, 'longitude': 77.5946},
                temperature=28.0,
                humidity=65.0,
                pressure=1013.0,
                wind_speed=5.0,
                wind_direction=180.0
            ))
        
    def test_zone_predictions(self):
        # Test only zones that have complete metadata
        test_zones = ['indiranagar', 'whitefield', 'koramangala']
        
        for zone in test_zones:
            # Verify zone exists
            self.assertIn(zone, self.predictor.zones)
            self.assertIn(zone, self.predictor.zone_metadata)
            
            # Get zone data
            zone_coords = self.predictor.zones[zone]
            zone_meta = self.predictor.zone_metadata[zone]
            
            # Add elevation to zone_meta from zone_coords
            zone_meta['elevation'] = zone_coords['elevation']
            
            # Now test the prediction
            prediction = self.predictor.predict_zone_weather(
                zone=zone,
                current_data=self.test_data,
                history=self.history
            )
            
            # Validate prediction structure and values
            self.assertIsNotNone(prediction)
            self.assertIn('temperature', prediction)
            self.assertIn('humidity', prediction)
            self.assertIn('pressure', prediction)
            self.assertIn('wind_speed', prediction)
            self.assertIn('confidence', prediction)
            
            # Value range checks with detailed error messages
            self.assertTrue(20 <= prediction['temperature'] <= 35, 
                          f"Temperature {prediction['temperature']} out of range for {zone}")
            self.assertTrue(30 <= prediction['humidity'] <= 90, 
                          f"Humidity {prediction['humidity']} out of range for {zone}")
            self.assertTrue(0 <= prediction['confidence'] <= 1, 
                          f"Confidence {prediction['confidence']} out of range for {zone}")

if __name__ == '__main__':
    unittest.main()