from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from datetime import datetime
from .agents import HistoricalPatternAgent, PhysicsModelAgent, NowcastingAgent
from .consensus_engine import ConsensusEngine  # Add this import

@dataclass
class MicroLocation:
    latitude: float
    longitude: float
    altitude: float
    heat_island_index: float
    tree_cover_density: float
    water_body_proximity: Optional[str]
    ward_name: str

@dataclass
class BangaloreForecast:
    timestamp: datetime
    location: MicroLocation
    temperature: float
    humidity: float
    precipitation_prob: float
    wind_speed: float
    confidence: float
    explanation: str

class BangaloreWeatherSystem:
    def __init__(self):
        self.micro_locations = self._init_bangalore_grid()
        self.historical_agent = HistoricalPatternAgent()
        self.physics_agent = PhysicsModelAgent()
        self.nowcast_agent = NowcastingAgent()
        self.consensus_engine = ConsensusEngine()
        
    def _init_bangalore_grid(self) -> List[MicroLocation]:
        """Initialize Bangalore's micro-location grid with geographical features"""
        return [
            MicroLocation(
                latitude=12.9719,
                longitude=77.6412,
                altitude=920,
                heat_island_index=0.7,
                tree_cover_density=0.4,
                water_body_proximity=None,
                ward_name="Indiranagar"
            ),
            MicroLocation(
                latitude=12.9279,
                longitude=77.6271,
                altitude=915,
                heat_island_index=0.8,
                tree_cover_density=0.3,
                water_body_proximity=None,
                ward_name="Koramangala"
            )
        ]

    async def get_forecast(self, latitude: float, longitude: float) -> BangaloreForecast:
        """Generate forecast for specific location using all agents"""
        location = self._get_nearest_microlocation(latitude, longitude)
        
        # Get predictions from all agents
        historical_pred = await self.historical_agent.predict(location)
        physics_pred = await self.physics_agent.predict(location)
        nowcast = await self.nowcast_agent.predict(location)
        
        # Generate consensus
        consensus = self.consensus_engine.generate_consensus(
            historical_pred,
            physics_pred,
            nowcast
        )
        
        return self._format_forecast(consensus, location)

    def _get_nearest_microlocation(self, lat: float, lon: float) -> MicroLocation:
        """Find nearest pre-defined micro-location"""
        if not self.micro_locations:
            return self._init_bangalore_grid()[0]  # Return default location if none exists
            
        # Simple Euclidean distance calculation for now
        distances = [(loc, (lat - loc.latitude)**2 + (lon - loc.longitude)**2) 
                    for loc in self.micro_locations]
        return min(distances, key=lambda x: x[1])[0]

    def _format_forecast(self, consensus: Dict, location: MicroLocation) -> BangaloreForecast:
        """Format consensus data into user-friendly forecast"""
        return BangaloreForecast(
            timestamp=datetime.now(),
            location=location,
            temperature=consensus.get('temperature', 25.0),
            humidity=consensus.get('humidity', 60.0),
            precipitation_prob=consensus.get('precipitation_prob', 0.0),
            wind_speed=consensus.get('wind_speed', 0.0),
            confidence=consensus.get('confidence', 0.5),
            explanation=self._generate_explanation(consensus, location)
        )

    def _generate_explanation(self, data: Dict, location: MicroLocation) -> str:
        """Generate natural language explanation of the forecast"""
        template = (
            f"For {location.ward_name}: {data.get('explanation_summary', 'No explanation available')}. "
            f"Confidence: {data.get('confidence', 0.5)*100:.1f}%"
        )
        return template