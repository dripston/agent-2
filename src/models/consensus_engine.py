from typing import Dict, List, Optional
import numpy as np
from sklearn.metrics import mean_squared_error

class ConsensusEngine:
    def __init__(self):
        self.agent_weights = {
            'historical': 0.4,
            'physics': 0.4,
            'nowcast': 0.2
        }
        self.conflict_threshold = 0.2

    def generate_consensus(
        self,
        historical_pred: Dict,
        physics_pred: Dict,
        nowcast_pred: Dict
    ) -> Dict:
        """Generate consensus from multiple predictions"""
        
        # Check for conflicts
        conflicts = self._detect_conflicts([
            historical_pred,
            physics_pred,
            nowcast_pred
        ])
        
        if conflicts:
            # Adjust weights based on recent performance
            self._adjust_weights(conflicts)
        
        # Weighted average of predictions
        consensus = {}
        for key in ['temperature', 'humidity', 'precipitation_prob']:
            consensus[key] = (
                historical_pred[key] * self.agent_weights['historical'] +
                physics_pred[key] * self.agent_weights['physics'] +
                nowcast_pred[key] * self.agent_weights['nowcast']
            )
        
        # Calculate overall confidence
        consensus['confidence'] = self._calculate_confidence(
            historical_pred, physics_pred, nowcast_pred
        )
        
        return consensus

    def _detect_conflicts(self, predictions: List[Dict]) -> List[Dict]:
        """Detect significant disagreements between agents"""
        conflicts = []
        for key in ['temperature', 'precipitation_prob']:
            values = [pred[key] for pred in predictions]
            if np.std(values) > self.conflict_threshold:
                conflicts.append({
                    'parameter': key,
                    'values': values,
                    'std': np.std(values)
                })
        return conflicts

    def _adjust_weights(self, conflicts: List[Dict]):
        """Adjust agent weights based on recent performance"""
        # Implementation for dynamic weight adjustment
        pass

    def _calculate_confidence(self, *predictions: Dict) -> float:
        """Calculate overall confidence score"""
        # Average of individual confidences, weighted by agent weights
        confidence = sum(
            pred['confidence'] * self.agent_weights[agent]
            for agent, pred in zip(['historical', 'physics', 'nowcast'], predictions)
        )
        return min(confidence, 1.0)