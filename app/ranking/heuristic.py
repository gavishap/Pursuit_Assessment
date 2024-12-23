from typing import Dict, List, Set
import re
from urllib.parse import urlparse
from .base import BaseRanker

class HeuristicRanker(BaseRanker):
    """Rule-based heuristic ranker using keyword matching and URL analysis."""
    
    def __init__(self):
        super().__init__("heuristic")
        # High-value keywords with weights
        self.keyword_weights = {
            'acfr': 1.0,
            'budget': 0.9,
            'financial': 0.8,
            'finance': 0.8,
            'report': 0.7,
            'treasury': 0.7,
            'tax': 0.6,
            'revenue': 0.6,
            'audit': 0.8,
            'fiscal': 0.7,
            'statement': 0.6,
            'annual': 0.5,
            'quarterly': 0.5,
            'policy': 0.4
        }
        
        # File extensions with weights
        self.file_weights = {
            '.pdf': 0.8,
            '.xlsx': 0.7,
            '.xls': 0.7,
            '.doc': 0.6,
            '.docx': 0.6,
            '.csv': 0.5
        }
        
        # URL path segments that indicate high value
        self.path_weights = {
            'finance': 0.8,
            'budget': 0.8,
            'treasury': 0.7,
            'report': 0.6,
            'document': 0.5,
            'download': 0.4
        }
        
        # Compile regex patterns
        self.year_pattern = re.compile(r'(19|20)\d{2}')  # Match years from 1900-2099
        
    def _calculate_keyword_score(self, text: str) -> float:
        """Calculate score based on keyword presence and weights."""
        if not text:
            return 0.0
            
        text = text.lower()
        score = 0.0
        matches = set()
        
        for keyword, weight in self.keyword_weights.items():
            if keyword in text:
                score += weight
                matches.add(keyword)
                
        # Bonus for multiple keyword matches
        if len(matches) > 1:
            score *= (1 + 0.1 * len(matches))
            
        return score
        
    def _calculate_url_score(self, url: str) -> float:
        """Calculate score based on URL analysis."""
        if not url:
            return 0.0
            
        score = 0.0
        parsed = urlparse(url.lower())
        
        # Check file extension
        for ext, weight in self.file_weights.items():
            if parsed.path.endswith(ext):
                score += weight
                break
                
        # Check path segments
        path_segments = parsed.path.split('/')
        for segment in path_segments:
            for key, weight in self.path_weights.items():
                if key in segment:
                    score += weight
                    
        # Bonus for year in URL (recent documents often more valuable)
        years = self.year_pattern.findall(url)
        if years:
            try:
                most_recent = max(int(year) for year in years)
                current_year = 2024  # You might want to make this dynamic
                if most_recent >= current_year - 2:  # Recent documents (within 2 years)
                    score *= 1.2
                elif most_recent >= current_year - 5:  # Somewhat recent (2-5 years)
                    score *= 1.1
            except ValueError:
                pass
                
        return score
        
    def calculate_score(self, link_data: Dict) -> float:
        """Calculate overall relevance score for a single link."""
        # Extract text content
        text_content = f"{link_data.get('title', '')} {' '.join(link_data.get('keywords', []))}"
        url = link_data.get('url', '')
        
        # Calculate component scores
        keyword_score = self._calculate_keyword_score(text_content)
        url_score = self._calculate_url_score(url)
        
        # Combine scores (giving more weight to keyword matches)
        combined_score = (0.7 * keyword_score + 0.3 * url_score)
        
        # Apply content type multiplier
        content_type = link_data.get('content_type', '').lower()
        if content_type == 'document':
            combined_score *= 1.2
        elif content_type == 'financial':
            combined_score *= 1.3
            
        return self.normalize_score(combined_score / 3.0)  # Normalize to 0-1 range
        
    def batch_calculate_scores(self, links: List[Dict]) -> List[float]:
        """Calculate relevance scores for a batch of links."""
        scores = [self.calculate_score(link) for link in links]
        return self.normalize_scores(scores) 
