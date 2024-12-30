from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import numpy as np

class BaseRanker(ABC):
    """Base class for all ranking methods."""
    
    def __init__(self, name: str):
        self.name = name
        
    @abstractmethod
    def score_url(self, url: str, context: str) -> float:
        """Score a single URL's relevance to the given context.
        
        Args:
            url: The URL to score
            context: The context to compare against (e.g., keywords or topic)
            
        Returns:
            float: Normalized relevance score between 0 and 1
        """
        pass
    
    @abstractmethod
    def rank_urls(self, urls: List[str], context: str) -> List[Tuple[str, float]]:
        """Rank a list of URLs based on their relevance to the context.
        
        Args:
            urls: List of URLs to rank
            context: The context to compare against
            
        Returns:
            List[Tuple[str, float]]: List of (url, score) pairs, sorted by score descending
        """
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
            
    def score_url(self, url: str, context: str) -> float:
        """Calculate weighted average score from all rankers."""
        scores = []
        for ranker, weight in zip(self.rankers, self.weights):
            try:
                score = ranker.score_url(url, context)
                scores.append(score * weight)
            except Exception as e:
                print(f"Error in ranker {ranker.name}: {str(e)}")
                scores.append(0.0)
        
        return sum(scores)
    
    def rank_urls(self, urls: List[str], context: str) -> List[Tuple[str, float]]:
        """Rank URLs using weighted average of all rankers."""
        scores = [(url, self.score_url(url, context)) for url in urls]
        return sorted(scores, key=lambda x: x[1], reverse=True) 
