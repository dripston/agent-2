class HistoricalPatternAgent:
    async def predict(self, location):
        # Temporary mock data for testing
        return {
            'temperature': 27.5,
            'humidity': 65.0,
            'precipitation_prob': 0.3,
            'wind_speed': 2.5,
            'confidence': 0.85,
            'explanation_summary': "Expect mild temperatures with moderate humidity"
        }

class PhysicsModelAgent:
    async def predict(self, location):
        return {
            'temperature': 28.0,
            'humidity': 62.0,
            'precipitation_prob': 0.25,
            'wind_speed': 2.8,
            'confidence': 0.9,
            'explanation_summary': "Stable atmospheric conditions indicate clear skies"
        }

class NowcastingAgent:
    async def predict(self, location):
        return {
            'temperature': 27.8,
            'humidity': 63.0,
            'precipitation_prob': 0.28,
            'wind_speed': 2.6,
            'confidence': 0.95,
            'explanation_summary': "Current conditions suggest pleasant weather"
        }