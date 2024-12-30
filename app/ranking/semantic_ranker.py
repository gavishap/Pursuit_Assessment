from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from urllib.parse import urlparse
import re
from typing import List, Dict, Tuple
import logging
from .base import BaseRanker

logger = logging.getLogger(__name__)

class SemanticRanker(BaseRanker):
    """A ranker that uses BERT embeddings to score URLs based on context."""
    
    def __init__(self):
        """Initialize the ranker with a BERT model."""
        super().__init__(name="semantic")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def _parse_url(self, url: str) -> Dict[str, str]:
        """Parse URL into meaningful components."""
        parsed = urlparse(url)
        
        # Extract parts
        domain = parsed.netloc
        path = parsed.path
        
        # Clean and tokenize domain
        domain = re.sub(r'^www\.', '', domain)
        domain = re.sub(r'\.(com|org|gov|edu|net)$', '', domain)
        
        # Clean and tokenize path
        path = re.sub(r'[-_/]', ' ', path)
        path = re.sub(r'\.\w+$', '', path)  # Remove file extensions
        
        return {
            'domain': domain,
            'path': path,
            'full': f"{domain} {path}".strip()
        }
    
    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for a list of texts."""
        return self.model.encode(texts, convert_to_tensor=True)
    
    def _compute_similarity(self, url_embedding: np.ndarray, context_embedding: np.ndarray) -> float:
        """Compute cosine similarity between embeddings."""
        return float(url_embedding @ context_embedding.T)
    
    def score_url(self, url: str, context: str) -> float:
        """Score a URL's relevance to the given context."""
        # Parse URL into components
        url_parts = self._parse_url(url)
        
        # Get embeddings
        url_embedding = self._get_embeddings([url_parts['full']])[0]
        context_embedding = self._get_embeddings([context])[0]
        
        # Compute similarity
        similarity = self._compute_similarity(url_embedding, context_embedding)
        
        # Normalize to 0-1 range
        score = (similarity + 1) / 2
        
        logger.debug(f"URL: {url}")
        logger.debug(f"Context: {context}")
        logger.debug(f"Score: {score:.4f}")
        
        return score
    
    def rank_urls(self, urls: List[str], context: str) -> List[Tuple[str, float]]:
        """Rank a list of URLs based on their relevance to the context."""
        # Score all URLs
        scores = [(url, self.score_url(url, context)) for url in urls]
        
        # Sort by score descending
        ranked = sorted(scores, key=lambda x: x[1], reverse=True)
        
        return ranked 
