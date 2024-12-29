from typing import Dict, List, Optional, Tuple
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
import joblib
import os
from .base import BaseRanker
from bs4 import BeautifulSoup
import requests
import re
from urllib.parse import urlparse
import logging
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans
from pathlib import Path

logger = logging.getLogger(__name__)

class UnsupervisedScorer:
    """Score links using unsupervised learning with BERT embeddings."""
    
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.kmeans = KMeans(n_clusters=5, random_state=42)  # 5 relevance levels
        
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get BERT embeddings for texts."""
        embeddings = []
        for text in texts:
            # Tokenize and get BERT embedding
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
            # Use CLS token embedding
            embedding = outputs.last_hidden_state[:, 0, :].numpy()
            embeddings.append(embedding[0])
        return np.array(embeddings)
        
    def fit_predict(self, links: List[Dict]) -> List[float]:
        """Score links using clustering on embeddings."""
        # Combine URL, title, and keywords for each link
        texts = [
            f"{link.get('url', '')} {link.get('title', '')} {' '.join(link.get('keywords', []))}"
            for link in links
        ]
        
        # Get embeddings
        embeddings = self.get_embeddings(texts)
        
        # Cluster embeddings
        clusters = self.kmeans.fit_predict(embeddings)
        
        # Calculate cluster centers' distances to ideal financial document
        financial_text = "financial report budget treasury audit fiscal statement"
        financial_embedding = self.get_embeddings([financial_text])[0]
        
        # Calculate distance of each cluster center to financial embedding
        cluster_scores = {}
        for cluster in range(5):
            cluster_center = self.kmeans.cluster_centers_[cluster]
            distance = np.linalg.norm(cluster_center - financial_embedding)
            cluster_scores[cluster] = 1 / (1 + distance)  # Convert distance to similarity score
            
        # Normalize cluster scores to 0-1 range
        min_score = min(cluster_scores.values())
        max_score = max(cluster_scores.values())
        for cluster in cluster_scores:
            cluster_scores[cluster] = (cluster_scores[cluster] - min_score) / (max_score - min_score)
            
        # Assign scores based on cluster
        return [cluster_scores[cluster] for cluster in clusters]

class MSLRFeatureExtractor:
    """Extract MSLR-like features from URLs and content."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        
    def extract_url_features(self, url: str) -> Dict:
        """Extract features from URL structure."""
        parsed = urlparse(url)
        
        # URL structure features
        url_features = {
            'url_length': len(url),
            'url_depth': len([x for x in parsed.path.split('/') if x]),
            'domain_length': len(parsed.netloc),
            'is_https': 1 if parsed.scheme == 'https' else 0,
        }
        
        # File type features
        extensions = {'.pdf', '.doc', '.docx', '.xls', '.xlsx', '.csv'}
        url_features['is_document'] = 1 if any(ext in url.lower() for ext in extensions) else 0
        
        return url_features
        
    def extract_content_features(self, content: str, query_terms: List[str]) -> Dict:
        """Extract features from page content."""
        if not content:
            return self._get_default_content_features()
            
        # Clean content
        soup = BeautifulSoup(content, 'html.parser')
        text = soup.get_text()
        title = soup.title.string if soup.title else ""
        
        # Content length features
        features = {
            'text_length': len(text),
            'title_length': len(title) if title else 0,
        }
        
        # Query term features
        for term in query_terms:
            term = term.lower()
            features[f'term_{term}_title'] = title.lower().count(term) if title else 0
            features[f'term_{term}_body'] = text.lower().count(term)
            
        # Document structure features
        features.update({
            'num_headings': len(soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])),
            'num_links': len(soup.find_all('a')),
            'has_meta_desc': 1 if soup.find('meta', attrs={'name': 'description'}) else 0
        })
        
        return features
        
    def _get_default_content_features(self) -> Dict:
        """Return default features when content can't be extracted."""
        return {
            'text_length': 0,
            'title_length': 0,
            'num_headings': 0,
            'num_links': 0,
            'has_meta_desc': 0
        }

class MLRanker(BaseRanker):
    """LightGBM-based ranker supporting multiple training methods."""
    
    def __init__(self, model_path: Optional[str] = None):
        super().__init__("ml")
        self.model = None
        self.feature_extractor = MSLRFeatureExtractor()
        self.unsupervised_scorer = UnsupervisedScorer()
        self.default_terms = ["financial", "budget", "report", "document"]
        
        # Try to load MSLR model by default
        default_model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                        'models', 'lightgbm_ranker.txt')
        if os.path.exists(default_model_path):
            self.load_model(default_model_path)
        # If custom path provided, try that instead
        elif model_path:
            self.load_model(model_path)
            
    def _get_page_content(self, url: str) -> Optional[str]:
        """Fetch page content with error handling."""
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return response.text
        except Exception as e:
            logger.warning(f"Error fetching content for {url}: {str(e)}")
        return None
        
    def _extract_features(self, link_data: Dict, query_terms: Optional[List[str]] = None) -> np.ndarray:
        """Extract all features for a link."""
        if query_terms is None:
            query_terms = self.default_terms
            
        url = link_data.get('url', '')
        
        # Initialize feature vector with zeros for all 136 MSLR features
        features = np.zeros(136)
        
        # Get content first to extract query-dependent features
        content = None
        if link_data.get('content'):
            content = link_data['content']
        else:
            content = self._get_page_content(url)
            
        if content:
            soup = BeautifulSoup(content, 'html.parser')
            text = soup.get_text()
            title = soup.title.string if soup.title else ""
            
            # Stream length (feature 1)
            features[0] = len(text)
            
            # Query-dependent features
            for i, term in enumerate(query_terms[:5]):  # Use up to 5 query terms
                term = term.lower()
                base_idx = i * 20  # Each term gets 20 feature slots
                
                # Stream-based features
                if text:
                    term_count = text.lower().count(term)
                    features[base_idx + 1] = term_count  # Term frequency
                    features[base_idx + 2] = term_count / (len(text) + 1)  # Term density
                    if term in text.lower():
                        features[base_idx + 3] = text.lower().index(term) / len(text)  # First position
                
                # Title-based features
                if title:
                    term_count_title = title.lower().count(term)
                    features[base_idx + 11] = term_count_title  # Term frequency in title
                    features[base_idx + 12] = term_count_title / (len(title) + 1)  # Term density in title
                    if term in title.lower():
                        features[base_idx + 13] = title.lower().index(term) / len(title)  # First position in title
            
            # Document structure features
            features[120] = len(title) if title else 0  # Title length
            features[121] = len(soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']))  # Number of headings
            features[122] = len(soup.find_all('a'))  # Number of outbound links
            features[123] = 1 if soup.find('meta', attrs={'name': 'description'}) else 0  # Has meta description
            
        # URL features
        parsed = urlparse(url)
        features[124] = len(parsed.netloc)  # Domain length
        features[125] = len(url)  # URL length
        features[126] = len([x for x in parsed.path.split('/') if x])  # URL depth
        features[127] = 1 if parsed.scheme == 'https' else 0  # HTTPS
        
        # File type features
        extensions = {'.pdf', '.doc', '.docx', '.xls', '.xlsx', '.csv'}
        features[128] = 1 if any(ext in url.lower() for ext in extensions) else 0
        
        # Quality/behavioral features from metadata
        metadata = link_data.get('link_metadata', {})
        if metadata:
            features[129] = float(metadata.get('pagerank', 0))
            features[130] = float(metadata.get('inbound_links', 0))
            features[131] = float(metadata.get('outbound_links', 0))
            features[132] = float(metadata.get('domain_authority', 0))
            features[133] = float(metadata.get('click_count', 0))
            features[134] = float(metadata.get('bounce_rate', 0))
            features[135] = float(metadata.get('dwell_time', 0))
            
        return features
        
    def train_unsupervised(self, links: List[Dict]) -> None:
        """Train model using unsupervised learning."""
        # Get unsupervised scores
        scores = self.unsupervised_scorer.fit_predict(links)
        
        # Extract features and train LightGBM
        features = []
        for link in links:
            link_features = self._extract_features(link)
            features.append(link_features)
            
        X = np.array(features)
        y = np.array(scores)
        
        # Initialize and train model
        self.model = lgb.LGBMRegressor(
            objective='regression',
            n_estimators=100,
            learning_rate=0.1
        )
        self.model.fit(X, y)
        
    def train_supervised(self, training_data: List[Dict], labels: List[float]) -> None:
        """Train model on labeled data."""
        if not training_data or not labels:
            raise ValueError("Training data and labels cannot be empty")
            
        # Extract features
        features = []
        for link in training_data:
            link_features = self._extract_features(link)
            features.append(link_features)
            
        X = np.array(features)
        y = np.array(labels)
        
        # Initialize and train model
        self.model = lgb.LGBMRegressor(
            objective='regression',
            n_estimators=100,
            learning_rate=0.1
        )
        self.model.fit(X, y)
        
    def save_model(self, model_dir: str) -> None:
        """Save the trained model."""
        os.makedirs(model_dir, exist_ok=True)
        
        if self.model is None:
            raise ValueError("Model has not been trained yet")
            
        if isinstance(self.model, lgb.Booster):
            model_path = os.path.join(model_dir, 'lightgbm_ranker.txt')
            self.model.save_model(model_path)
        else:
            model_path = os.path.join(model_dir, 'model.joblib')
            joblib.dump(self.model, model_path)
        
    def load_model(self, model_path: str) -> None:
        """Load a trained model."""
        if model_path.endswith('.txt'):
            self.model = lgb.Booster(model_file=model_path)
        else:
            self.model = joblib.load(model_path)
        logger.info(f"Loaded model from {model_path}")
        
    def _normalize_score(self, score: float) -> float:
        """Normalize score to be between 0 and 1."""
        # Sigmoid normalization
        return 1 / (1 + np.exp(-score))
        
    def calculate_score(self, link_data: Dict) -> float:
        """Calculate relevance score for a single link."""
        if self.model is None:
            # Use unsupervised scoring if no model is loaded
            return self.unsupervised_scorer.fit_predict([link_data])[0]
            
        # Extract features
        features = self._extract_features(link_data)
        
        # Get prediction from MSLR model
        if isinstance(self.model, lgb.Booster):
            score = float(self.model.predict([features])[0])
        else:
            score = float(self.model.predict([features])[0])
            
        # Normalize score
        score = self._normalize_score(score)
            
        # If score is too low, fallback to unsupervised scoring
        if score < 0.01:
            logger.info("MSLR model returned very low score, using unsupervised scoring instead")
            score = self.unsupervised_scorer.fit_predict([link_data])[0]
            
        return score
        
    def batch_calculate_scores(self, links: List[Dict]) -> List[float]:
        """Calculate relevance scores for a batch of links."""
        if self.model is None:
            # Use unsupervised scoring if no model is loaded
            return self.unsupervised_scorer.fit_predict(links)
            
        # Extract features
        features = []
        for link in links:
            link_features = self._extract_features(link)
            features.append(link_features)
            
        # Get predictions from MSLR model
        if isinstance(self.model, lgb.Booster):
            scores = self.model.predict(features)
        else:
            scores = self.model.predict(features)
            
        # Normalize scores
        scores = [self._normalize_score(s) for s in scores]
            
        # For any scores that are too low, use unsupervised scoring
        if all(s < 0.01 for s in scores):
            logger.info("MSLR model returned very low scores, using unsupervised scoring instead")
            scores = self.unsupervised_scorer.fit_predict(links)
            
        return scores
        
