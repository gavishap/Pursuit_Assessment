from typing import Dict, List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from .base import BaseRanker

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class NLPRanker(BaseRanker):
    """NLP-based ranker using TF-IDF and text analysis."""
    
    def __init__(self):
        super().__init__("nlp")
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Financial domain-specific terms
        self.domain_terms = {
            'financial': 1.0,
            'fiscal': 1.0,
            'budget': 1.0,
            'revenue': 0.9,
            'expenditure': 0.9,
            'audit': 0.9,
            'statement': 0.8,
            'report': 0.8,
            'balance': 0.8,
            'account': 0.7,
            'fund': 0.7,
            'tax': 0.7,
            'expense': 0.7,
            'income': 0.7,
            'asset': 0.7,
            'liability': 0.7,
            'treasury': 0.8,
            'investment': 0.6,
            'debt': 0.6,
            'capital': 0.6
        }
        
        # Initialize TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            tokenizer=self._tokenize,
            stop_words='english',
            ngram_range=(1, 2),  # Include bigrams
            max_features=1000
        )
        
        # Pre-compute TF-IDF for domain terms
        self.domain_text = ' '.join(self.domain_terms.keys())
        
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize and preprocess text."""
        if not text:
            return []
            
        # Tokenize
        tokens = word_tokenize(text.lower())
        
        # Remove stopwords and lemmatize
        tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token not in self.stop_words and token.isalnum()
        ]
        
        return tokens
        
    def _calculate_domain_relevance(self, text: str) -> float:
        """Calculate relevance based on domain-specific terms."""
        if not text:
            return 0.0
            
        tokens = self._tokenize(text)
        score = 0.0
        matches = set()
        
        for token in tokens:
            if token in self.domain_terms:
                score += self.domain_terms[token]
                matches.add(token)
                
        # Bonus for multiple domain term matches
        if len(matches) > 1:
            score *= (1 + 0.1 * len(matches))
            
        return score
        
    def _calculate_tfidf_similarity(self, text: str) -> float:
        """Calculate TF-IDF similarity with domain terms."""
        if not text:
            return 0.0
            
        # Create a small corpus with domain text and input text
        corpus = [self.domain_text, text]
        
        try:
            # Calculate TF-IDF
            tfidf_matrix = self.vectorizer.fit_transform(corpus)
            
            # Calculate cosine similarity between domain terms and input text
            similarity = (tfidf_matrix * tfidf_matrix.T).A[0][1]
            return float(similarity)
        except Exception as e:
            print(f"Error calculating TF-IDF similarity: {str(e)}")
            return 0.0
            
    def calculate_score(self, link_data: Dict) -> float:
        """Calculate overall relevance score for a single link."""
        # Combine all text content
        text_content = f"{link_data.get('title', '')} {' '.join(link_data.get('keywords', []))}"
        url = link_data.get('url', '')
        
        # Calculate component scores
        domain_score = self._calculate_domain_relevance(text_content)
        tfidf_score = self._calculate_tfidf_similarity(text_content)
        
        # URL analysis
        url_score = self._calculate_domain_relevance(url) * 0.5  # Lower weight for URL terms
        
        # Combine scores
        combined_score = (
            0.4 * domain_score +  # Domain term matching
            0.4 * tfidf_score +   # TF-IDF similarity
            0.2 * url_score       # URL analysis
        )
        
        # Apply content type multiplier
        content_type = link_data.get('content_type', '').lower()
        if content_type == 'document':
            combined_score *= 1.2
        elif content_type == 'financial':
            combined_score *= 1.3
            
        return self.normalize_score(combined_score)
        
    def batch_calculate_scores(self, links: List[Dict]) -> List[float]:
        """Calculate relevance scores for a batch of links."""
        scores = [self.calculate_score(link) for link in links]
        return self.normalize_scores(scores) 
