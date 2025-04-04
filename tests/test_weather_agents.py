import unittest
from datetime import datetime
from src.data.weather_data import AtmosphericData
from src.models.equation_agents import *
from src.models.agent_orchestrator import AgentOrchestrator

class TestWeatherAgents(unittest.TestCase):
    def setUp(self):
        # Sample weather data for Bangalore
        self.test_data = AtmosphericData(
            timestamp=datetime.now(),
            location={'latitude': 12.9716, 'longitude': 77.5946},
            temperature=28.0,  # Celsius
            humidity=65.0,     # Percentage
            pressure=1013.0,   # hPa
            wind_speed=5.0,    # m/s
            wind_direction=180.0,  # degrees
            confidence=0.9
        )
        
    def test_ideal_gas_agent(self):
        agent = IdealGasAgent()
        result = agent.calculate(self.test_data)
        self.assertIn('air_density', result)
        self.assertTrue(1.0 < result['air_density'] < 1.5)  # Typical air density range
        
    def test_adiabatic_process_agent(self):
        agent = AdibaticProcessAgent()
        result = agent.calculate(self.test_data)
        self.assertIn('temperature', result)
        self.assertTrue(-50 < result['temperature'] < 50)  # Reasonable temperature range
        
    def test_dew_point_agent(self):
        agent = DewPointAgent()
        result = agent.calculate(self.test_data)
        self.assertIn('dew_point', result)
        self.assertTrue(result['dew_point'] < self.test_data.temperature)
        
    def test_solar_radiation_agent(self):
        agent = SolarRadiationAgent()
        result = agent.calculate(self.test_data)
        self.assertIn('temperature', result)
        self.assertTrue(-50 < result['temperature'] < 50)
        
    def test_cloud_formation_agent(self):
        agent = CloudFormationAgent()
        result = agent.calculate(self.test_data)
        self.assertIn('pressure', result)
        self.assertTrue(900 < result['pressure'] < 1100)  # reasonable pressure range
        
    def test_turbulence_agent(self):
        agent = TurbulenceAgent()
        result = agent.calculate(self.test_data)
        self.assertIn('wind_speed', result)
        self.assertTrue(result['wind_speed'] >= self.test_data.wind_speed)  # turbulence should increase wind speed

    def test_comprehensive_forecast(self):
        orchestrator = AgentOrchestrator()
        forecast = orchestrator.generate_forecast(self.test_data)
        
        # Test comprehensive results
        self.assertIsNotNone(forecast.prediction_data.temperature)
        self.assertIsNotNone(forecast.prediction_data.pressure)
        self.assertIsNotNone(forecast.prediction_data.wind_speed)
        
        # Test reasonable ranges
        self.assertTrue(-50 < forecast.prediction_data.temperature < 50)
        self.assertTrue(900 < forecast.prediction_data.pressure < 1100)
        self.assertTrue(0 <= forecast.prediction_data.wind_speed < 100)
        
    def test_orchestrator(self):
        orchestrator = AgentOrchestrator()
        prediction = orchestrator.generate_forecast(self.test_data)
        
        self.assertIsNotNone(prediction)
        self.assertEqual(prediction.location, self.test_data.location)
        self.assertTrue(0 <= prediction.confidence_score <= 1)
        
    def test_all_agents_integration(self):
        orchestrator = AgentOrchestrator()
        predictions = orchestrator.collect_predictions(self.test_data)
        
        # Check if we have predictions for each parameter
        self.assertTrue(len(predictions['temperature']) > 0)
        self.assertTrue(len(predictions['pressure']) > 0)

if __name__ == '__main__':
    unittest.main()