from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import joblib
import os
from .base import BaseRanker

class MLRanker(BaseRanker):
    """Machine learning-based ranker using Random Forest."""
    
    def __init__(self, model_path: Optional[str] = None):
        super().__init__("ml")
        self.model = None
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            stop_words='english'
        )
        self.scaler = StandardScaler()
        
        # Load pre-trained model if path provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            
    def _extract_features(self, link_data: Dict) -> np.ndarray:
        """Extract features from link data."""
        # Text features
        text_content = f"{link_data.get('title', '')} {' '.join(link_data.get('keywords', []))}"
        
        # URL features
        url = link_data.get('url', '').lower()
        
        # Numeric features
        numeric_features = [
            # Length features
            len(text_content),
            len(url),
            len(link_data.get('keywords', [])),
            
            # Binary features
            1 if 'financial' in url else 0,
            1 if 'budget' in url else 0,
            1 if 'report' in url else 0,
            1 if link_data.get('content_type') == 'document' else 0,
            1 if link_data.get('content_type') == 'financial' else 0,
            
            # Count features
            text_content.lower().count('finance'),
            text_content.lower().count('budget'),
            text_content.lower().count('report'),
            
            # Special features
            1 if any(ext in url for ext in ['.pdf', '.doc', '.docx', '.xls', '.xlsx']) else 0,
            1 if any(year in url for year in [str(y) for y in range(2020, 2025)]) else 0
        ]
        
        return np.array(numeric_features)
        
    def train(self, training_data: List[Dict], labels: List[float]) -> None:
        """Train the model on labeled data."""
        if not training_data or not labels:
            raise ValueError("Training data and labels cannot be empty")
            
        # Extract features
        numeric_features = []
        text_contents = []
        
        for link in training_data:
            numeric_features.append(self._extract_features(link))
            text_content = f"{link.get('title', '')} {' '.join(link.get('keywords', []))}"
            text_contents.append(text_content)
            
        # Convert to numpy arrays
        numeric_features = np.array(numeric_features)
        
        # Process text features
        text_features = self.vectorizer.fit_transform(text_contents)
        
        # Scale numeric features
        scaled_numeric = self.scaler.fit_transform(numeric_features)
        
        # Combine features
        X = np.hstack([scaled_numeric, text_features.toarray()])
        
        # Train model
        self.model.fit(X, labels)
        
    def save_model(self, model_dir: str) -> None:
        """Save the trained model and preprocessors."""
        os.makedirs(model_dir, exist_ok=True)
        
        if self.model is None:
            raise ValueError("Model has not been trained yet")
            
        joblib.dump(self.model, os.path.join(model_dir, 'model.joblib'))
        joblib.dump(self.vectorizer, os.path.join(model_dir, 'vectorizer.joblib'))
        joblib.dump(self.scaler, os.path.join(model_dir, 'scaler.joblib'))
        
    def load_model(self, model_dir: str) -> None:
        """Load a trained model and preprocessors."""
        self.model = joblib.load(os.path.join(model_dir, 'model.joblib'))
        self.vectorizer = joblib.load(os.path.join(model_dir, 'vectorizer.joblib'))
        self.scaler = joblib.load(os.path.join(model_dir, 'scaler.joblib'))
        
    def calculate_score(self, link_data: Dict) -> float:
        """Calculate relevance score for a single link."""
        if self.model is None:
            raise ValueError("Model has not been trained or loaded yet")
            
        # Extract features
        numeric_features = self._extract_features(link_data)
        text_content = f"{link_data.get('title', '')} {' '.join(link_data.get('keywords', []))}"
        
        # Process features
        text_features = self.vectorizer.transform([text_content])
        scaled_numeric = self.scaler.transform(numeric_features.reshape(1, -1))
        
        # Combine features
        X = np.hstack([scaled_numeric, text_features.toarray()])
        
        # Predict probability
        probabilities = self.model.predict_proba(X)
        score = float(probabilities[0][1])  # Probability of being relevant
        
        return self.normalize_score(score)
        
    def batch_calculate_scores(self, links: List[Dict]) -> List[float]:
        """Calculate relevance scores for a batch of links."""
        if self.model is None:
            raise ValueError("Model has not been trained or loaded yet")
            
        # Extract features
        numeric_features = []
        text_contents = []
        
        for link in links:
            numeric_features.append(self._extract_features(link))
            text_content = f"{link.get('title', '')} {' '.join(link.get('keywords', []))}"
            text_contents.append(text_content)
            
        # Process features
        numeric_features = np.array(numeric_features)
        text_features = self.vectorizer.transform(text_contents)
        scaled_numeric = self.scaler.transform(numeric_features)
        
        # Combine features
        X = np.hstack([scaled_numeric, text_features.toarray()])
        
        # Predict probabilities
        probabilities = self.model.predict_proba(X)
        scores = [float(p[1]) for p in probabilities]  # Probability of being relevant
        
        return self.normalize_scores(scores) 
