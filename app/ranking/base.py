from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import numpy as np

class BaseRanker(ABC):
    """Base class for all ranking methods."""
    
    def __init__(self, name: str):
        self.name = name
        
    @abstractmethod
    def calculate_score(self, link_data: Dict) -> float:
        """Calculate relevance score for a single link."""
        pass
    
    @abstractmethod
    def batch_calculate_scores(self, links: List[Dict]) -> List[float]:
        """Calculate relevance scores for a batch of links."""
        pass
    
    def normalize_score(self, score: float) -> float:
        """Normalize score to be between 0 and 1."""
        return max(0.0, min(1.0, score))
    
    def normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize a list of scores to be between 0 and 1."""
        if not scores:
            return []
        
        # Convert to numpy array for vectorized operations
        scores_array = np.array(scores)
        
        # Handle case where all scores are the same
        if np.all(scores_array == scores_array[0]):
            return [0.5] * len(scores)  # Return middle value for all
            
        # Normalize using min-max scaling
        min_score = np.min(scores_array)
        max_score = np.max(scores_array)
        
        if min_score == max_score:
            return [0.5] * len(scores)
            
        normalized = (scores_array - min_score) / (max_score - min_score)
        return normalized.tolist()

class EnsembleRanker:
    """Combines multiple rankers and averages their scores."""
    
    def __init__(self, rankers: List[BaseRanker], weights: Optional[List[float]] = None):
        self.rankers = rankers
        if weights is None:
            # Equal weights if none provided
            self.weights = [1.0 / len(rankers)] * len(rankers)
        else:
            # Normalize weights to sum to 1
            total = sum(weights)
            self.weights = [w / total for w in weights]
            
    def calculate_score(self, link_data: Dict) -> float:
        """Calculate weighted average score from all rankers."""
        scores = []
        for ranker, weight in zip(self.rankers, self.weights):
            try:
                score = ranker.calculate_score(link_data)
                scores.append(score * weight)
            except Exception as e:
                print(f"Error in ranker {ranker.name}: {str(e)}")
                scores.append(0.0)
        
        return sum(scores)
    
    def batch_calculate_scores(self, links: List[Dict]) -> List[float]:
        """Calculate weighted average scores for a batch of links."""
        all_scores = []
        for ranker, weight in zip(self.rankers, self.weights):
            try:
                scores = ranker.batch_calculate_scores(links)
                all_scores.append([s * weight for s in scores])
            except Exception as e:
                print(f"Error in ranker {ranker.name}: {str(e)}")
                all_scores.append([0.0] * len(links))
        
        # Sum scores from all rankers
        final_scores = np.sum(all_scores, axis=0).tolist()
        return final_scores 
