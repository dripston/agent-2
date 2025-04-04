from abc import ABC, abstractmethod
from typing import Dict
from ..data.weather_data import AtmosphericData

class PhysicsAgent(ABC):
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.confidence_score = 0.0
        self.last_prediction = None
        
    @abstractmethod
    def calculate(self, data: AtmosphericData) -> Dict[str, float]:
        pass