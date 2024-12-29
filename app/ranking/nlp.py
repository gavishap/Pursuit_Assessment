import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from gensim.models import KeyedVectors
from transformers import RobertaModel, RobertaTokenizer
import torch
from typing import List, Tuple
import logging
from urllib.parse import urlparse, parse_qs
import re

logger = logging.getLogger(__name__)

class AdvancedNLPRanker:
    """A comprehensive NLP ranker using multiple advanced techniques."""
    
    def __init__(self):
        """Initialize various NLP models and vectorizers."""
        # Load spaCy model for text processing and word vectors
        self.nlp = spacy.load('en_core_web_lg')
        
        # Initialize RoBERTa for contextual embeddings
        self.roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.roberta_model = RobertaModel.from_pretrained('roberta-base')
        
        # Initialize TF-IDF vectorizer with better parameters
        self.tfidf = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 3),  # Include trigrams
            max_features=20000,
            min_df=1,
            max_df=0.95
        )
        
        # Store processed URLs for reuse
        self.url_cache = {}
    
    def _preprocess_url(self, url: str) -> str:
        """Extract and clean text from URL with improved processing."""
        if url in self.url_cache:
            return self.url_cache[url]
        
        parsed = urlparse(url)
        
        # Extract meaningful parts
        parts = []
        
        # Process domain with less aggressive cleaning
        domain = parsed.netloc.lower()
        domain = re.sub(r'^www\.', '', domain)
        domain_parts = domain.split('.')
        parts.extend(domain_parts[:-1])  # Exclude TLD but keep subdomain structure
        
        # Process path more carefully
        path = parsed.path.lower()
        
        # Split path but preserve important separators
        path_parts = []
        current_part = ""
        for char in path:
            if char in '/-_':
                if current_part:
                    path_parts.append(current_part)
                current_part = ""
            else:
                current_part += char
        if current_part:
            path_parts.append(current_part)
        
        # Add path parts
        parts.extend(path_parts)
        
        # Process query parameters if they exist
        query_params = parse_qs(parsed.query)
        for key, values in query_params.items():
            # Clean parameter names
            key = re.sub(r'[^a-zA-Z0-9]', ' ', key)
            parts.extend(key.split())
            
            # Add parameter values if they look like words
            for value in values:
                if re.match(r'^[a-zA-Z0-9\s]+$', value):
                    parts.extend(value.split())
        
        # Clean and join with better text processing
        text = ' '.join(p for p in parts if p and len(p) > 1)
        
        # Use spaCy for better text processing
        doc = self.nlp(text)
        processed_parts = []
        for token in doc:
            if not token.is_stop and not token.is_punct and len(token.text) > 1:
                processed_parts.append(token.text)
        
        final_text = ' '.join(processed_parts)
        self.url_cache[url] = final_text
        return final_text
    
    def _get_spacy_similarity(self, url_text: str, context: str) -> float:
        """Calculate similarity using spaCy's word vectors with improved processing."""
        url_doc = self.nlp(url_text)
        context_doc = self.nlp(context)
        
        if not url_doc.vector_norm or not context_doc.vector_norm:
            return 0.0
        
        # Get similarity score
        base_similarity = url_doc.similarity(context_doc)
        
        # Boost score based on entity matching
        url_entities = set(e.text.lower() for e in url_doc.ents)
        context_entities = set(e.text.lower() for e in context_doc.ents)
        matching_entities = url_entities.intersection(context_entities)
        
        entity_bonus = len(matching_entities) * 0.1
        
        return min(1.0, base_similarity + entity_bonus)
    
    def _get_tfidf_similarity(self, url_text: str, context: str) -> float:
        """Calculate TF-IDF based cosine similarity with improved weighting."""
        try:
            # Fit and transform on both texts
            vectors = self.tfidf.fit_transform([url_text, context])
            
            # Get feature names for analysis
            feature_names = self.tfidf.get_feature_names_out()
            
            # Calculate base similarity
            url_vector = vectors[0].toarray().flatten()
            context_vector = vectors[1].toarray().flatten()
            
            # Find matching terms
            matching_terms = []
            for i, (u, c) in enumerate(zip(url_vector, context_vector)):
                if u > 0 and c > 0:
                    matching_terms.append((feature_names[i], min(u, c)))
            
            # Sort by importance
            matching_terms.sort(key=lambda x: x[1], reverse=True)
            
            # Calculate weighted similarity
            similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
            
            # Boost score based on number of matching important terms
            term_bonus = min(0.3, len(matching_terms) * 0.05)
            
            return min(1.0, similarity + term_bonus)
        except:
            return 0.0
    
    def _get_roberta_similarity(self, url_text: str, context: str) -> float:
        """Calculate similarity using RoBERTa embeddings with improved processing."""
        try:
            # Tokenize texts
            url_tokens = self.roberta_tokenizer(url_text, return_tensors='pt', padding=True, truncation=True)
            context_tokens = self.roberta_tokenizer(context, return_tensors='pt', padding=True, truncation=True)
            
            # Get embeddings from last 4 layers for better representation
            with torch.no_grad():
                url_outputs = self.roberta_model(**url_tokens, output_hidden_states=True)
                context_outputs = self.roberta_model(**context_tokens, output_hidden_states=True)
                
                # Combine last 4 layers
                url_layers = url_outputs.hidden_states[-4:]
                context_layers = context_outputs.hidden_states[-4:]
                
                url_embedding = torch.mean(torch.stack(url_layers), dim=0).mean(dim=1)
                context_embedding = torch.mean(torch.stack(context_layers), dim=0).mean(dim=1)
            
            # Calculate similarity
            similarity = torch.nn.functional.cosine_similarity(url_embedding, context_embedding).item()
            
            return max(0.0, min(1.0, similarity))
        except:
            return 0.0
    
    def score_url(self, url: str, context: str) -> float:
        """Score URL relevance using multiple NLP techniques."""
        # Preprocess URL
        url_text = self._preprocess_url(url)
        
        # Get scores from different methods
        scores = {
            'spacy': self._get_spacy_similarity(url_text, context),
            'tfidf': self._get_tfidf_similarity(url_text, context),
            'roberta': self._get_roberta_similarity(url_text, context)
        }
        
        # Log individual scores
        logger.debug(f"URL: {url}")
        logger.debug(f"Context: {context}")
        logger.debug("Individual scores:")
        for method, score in scores.items():
            logger.debug(f"  {method}: {score:.4f}")
        
        # Calculate weighted average with adjusted weights
        weights = {
            'spacy': 0.35,    # Increased weight for spaCy (good with entities)
            'tfidf': 0.30,    # Increased for exact matching
            'roberta': 0.35   # Balanced weight for deep understanding
        }
        
        weighted_score = sum(score * weights[method] for method, score in scores.items())
        return weighted_score
    
    def rank_urls(self, urls: List[str], context: str) -> List[Tuple[str, float]]:
        """Rank a list of URLs based on their relevance to the context."""
        # Score all URLs
        scores = [(url, self.score_url(url, context)) for url in urls]
        
        # Sort by score descending
        ranked = sorted(scores, key=lambda x: x[1], reverse=True)
        
        return ranked 
