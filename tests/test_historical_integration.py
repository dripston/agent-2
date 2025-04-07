import unittest
from datetime import datetime, timedelta
from src.data.historical_analyzer import HistoricalWeatherAnalyzer
from src.data.multi_source_fetcher import MultiSourceFetcher
from src.data.weather_data import AtmosphericData
import requests
class TestHistoricalIntegration(unittest.TestCase):
    def setUp(self):
        self.analyzer = HistoricalWeatherAnalyzer()
        self.fetcher = MultiSourceFetcher()
        self.test_location = {
            'latitude': 12.9716,
            'longitude': 77.5946
        }
        
    def test_historical_data_fetch(self):
        # Test 7-day historical data fetch
        days_back = 7
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        data = self.analyzer.fetcher.fetch_historical_data(
            self.test_location['latitude'],
            self.test_location['longitude'],
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d")
        )
        
        self.assertIsNotNone(data)
        self.assertIn('temperature_2m', data.columns)
        self.assertIn('precipitation', data.columns)
        self.assertIn('wind_speed_10m', data.columns)
        
    def test_temperature_trend_analysis(self):
        analysis = self.analyzer.get_historical_analysis(
            self.test_location['latitude'],
            self.test_location['longitude']
        )
        
        trend = analysis['temperature_trend']
        self.assertIsNotNone(trend)
        self.assertTrue(len(trend) > 0)
        self.assertTrue(all(-50 <= temp <= 50 for temp in trend))
        
    def test_rain_probability_calculation(self):
        analysis = self.analyzer.get_historical_analysis(
            self.test_location['latitude'],
            self.test_location['longitude']
        )
        
        prob = analysis['rain_probability']
        self.assertIsNotNone(prob)
        self.assertTrue(0 <= prob <= 1)
        
    def test_elevation_based_temperature(self):
        analysis = self.analyzer.get_historical_analysis(
            self.test_location['latitude'],
            self.test_location['longitude']
        )
        
        adjusted_temp = analysis['elevation_adjusted_temp']
        self.assertIsNotNone(adjusted_temp)
        self.assertTrue(len(adjusted_temp) > 0)
        
        # Test elevation effect (Bangalore ~920m)
        # Expected temperature difference around -6°C
        expected_diff = -6
        # Calculate the mean difference instead of using just the last value
        actual_diff = (adjusted_temp - analysis['temperature_trend']).mean()
        self.assertAlmostEqual(actual_diff, expected_diff, delta=1)

if __name__ == '__main__':
    unittest.main()