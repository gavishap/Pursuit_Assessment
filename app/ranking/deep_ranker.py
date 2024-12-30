import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import logging
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from urllib.parse import urlparse, unquote, parse_qs, urlencode
import re
from .base import BaseRanker

logger = logging.getLogger(__name__)

class URLPreprocessor:
    """Preprocess URLs and context for better model understanding."""
    
    @staticmethod
    def preprocess_url(url: str) -> str:
        """Extract meaningful parts from URL and format them for the model."""
        try:
            # Parse URL
            parsed = urlparse(url)
            
            # Extract and clean domain
            domain_parts = parsed.netloc.split('.')
            domain = ' '.join([part for part in domain_parts if part not in ['com', 'org', 'net', 'edu', 'gov', 'www']])
            
            # Clean and split path
            path = unquote(parsed.path)
            # Handle dates first (before removing slashes)
            path = re.sub(r'\d{4}/\d{2}/\d{2}', ' DATE ', path)
            # Replace separators with spaces
            path = re.sub(r'[-_/]', ' ', path)
            # Replace remaining numbers
            path = re.sub(r'\d+', ' NUM ', path)
            
            # Extract meaningful words using regex
            words = [w for w in re.findall(r'[a-zA-Z]+', path.lower()) 
                    if len(w) > 2 and w not in ['www', 'html', 'htm', 'php', 'asp', 'jsp']]
            
            # Extract and clean query parameters
            query_words = []
            if parsed.query:
                params = parse_qs(parsed.query)
                # Extract words from parameter names and values
                for key, values in params.items():
                    query_words.extend(re.findall(r'[a-zA-Z]+', key.lower()))
                    for value in values:
                        query_words.extend(re.findall(r'[a-zA-Z]+', str(value).lower()))
                query_words = [w for w in query_words if len(w) > 2]
            
            # Combine parts with special tokens and extra context
            processed = f"[DOMAIN] {domain}"
            
            if words:
                processed += f" [PATH] {' '.join(words)}"
            
            if query_words:
                processed += f" [QUERY] {' '.join(query_words)}"
            
            # Add special tokens for URL types
            if 'news' in domain or 'article' in path:
                processed += " [TYPE] news"
            elif 'blog' in domain or 'post' in path:
                processed += " [TYPE] blog"
            elif 'wiki' in domain:
                processed += " [TYPE] wiki"
            elif any(x in domain or x in path for x in ['doc', 'documentation', 'guide', 'tutorial']):
                processed += " [TYPE] documentation"
            elif any(x in domain or x in path for x in ['shop', 'store', 'product']):
                processed += " [TYPE] commerce"
            
            return processed
        except:
            return url
    
    @staticmethod
    def preprocess_context(context: str) -> str:
        """Clean and normalize context text."""
        try:
            # Convert to lowercase first
            cleaned = context.lower()
            
            # Remove special characters but keep meaningful separators
            cleaned = re.sub(r'[^\w\s-]', ' ', cleaned)
            
            # Replace numbers with special tokens
            cleaned = re.sub(r'\d{4}', 'YEAR', cleaned)
            cleaned = re.sub(r'\d+', 'NUM', cleaned)
            
            # Normalize whitespace
            cleaned = re.sub(r'\s+', ' ', cleaned)
            
            # Add topic markers for common categories
            topics = ['news', 'sports', 'technology', 'science', 'business', 
                     'entertainment', 'health', 'politics', 'education']
            
            for topic in topics:
                if topic in cleaned:
                    cleaned = f"[TOPIC] {topic} " + cleaned
            
            return cleaned.strip()
        except:
            return context

class URLDataset(Dataset):
    def __init__(self, urls, topics, ranks, tokenizer, max_url_length=128, max_topic_length=32):
        self.urls = urls
        self.topics = topics
        self.ranks = ranks
        self.tokenizer = tokenizer
        self.max_url_length = max_url_length
        self.max_topic_length = max_topic_length
        self.preprocessor = URLPreprocessor()

    def __len__(self):
        return len(self.urls)

    def __getitem__(self, idx):
        url = str(self.urls[idx])
        topic = str(self.topics[idx])
        rank = float(self.ranks[idx])

        # Preprocess URL and topic
        processed_url = self.preprocessor.preprocess_url(url)
        processed_topic = self.preprocessor.preprocess_context(topic)

        # Tokenize with truncation and padding
        url_encoding = self.tokenizer(
            processed_url,
            padding='max_length',
            max_length=self.max_url_length,
            truncation=True,
            return_tensors='pt'
        )
        
        topic_encoding = self.tokenizer(
            processed_topic,
            padding='max_length',
            max_length=self.max_topic_length,
            truncation=True,
            return_tensors='pt'
        )

        return {
            'url_ids': url_encoding['input_ids'].squeeze(),
            'url_mask': url_encoding['attention_mask'].squeeze(),
            'topic_ids': topic_encoding['input_ids'].squeeze(),
            'topic_mask': topic_encoding['attention_mask'].squeeze(),
            'rank': torch.tensor(rank, dtype=torch.float32)
        }

class DeepURLRanker(nn.Module):
    """Neural network model for URL ranking."""

    def __init__(self, hidden_size=512):
        super().__init__()
        # Use MPNet base for better performance
        self.encoder = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')
        
        # Separate encoders for URL and topic
        self.url_encoder = nn.Sequential(
            nn.Linear(768, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.topic_encoder = nn.Sequential(
            nn.Linear(768, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Cross-attention mechanism
        self.cross_attention = nn.MultiheadAttention(hidden_size, num_heads=8, dropout=0.1)
        
        # Final classifier with residual connections
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, url_ids, url_mask, topic_ids, topic_mask):
        # Get embeddings
        url_output = self.encoder(url_ids, attention_mask=url_mask).last_hidden_state
        topic_output = self.encoder(topic_ids, attention_mask=topic_mask).last_hidden_state
        
        # Get CLS token embeddings
        url_emb = self.url_encoder(url_output[:, 0])
        topic_emb = self.topic_encoder(topic_output[:, 0])
        
        # Cross attention between URL and topic
        url_attn, _ = self.cross_attention(
            url_emb.unsqueeze(0),
            topic_emb.unsqueeze(0),
            topic_emb.unsqueeze(0)
        )
        url_attn = url_attn.squeeze(0)
        
        # Combine features with residual connection
        combined = torch.cat([
            url_emb,
            topic_emb,
            url_attn
        ], dim=1)
        
        # Final classification
        score = self.classifier(combined)
        return score.squeeze()

class TrainedDeepRanker(BaseRanker):
    """A ranker that uses a trained deep learning model to score URLs."""
    
    def __init__(self):
        """Initialize the deep learning ranker."""
        super().__init__(name="deep_learning")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = DeepURLRanker()
        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
        self.preprocessor = URLPreprocessor()
        
        # Load trained model if it exists
        model_path = Path(__file__).parent.parent.parent / 'models' / 'deep_ranker_v3.pt'
        if model_path.exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded trained model from {model_path}")
            logger.info(f"Model validation loss: {checkpoint['val_loss']:.4f}")
        else:
            # Train the model if no saved model exists
            csv_path = Path(__file__).parent.parent.parent / 'data' / 'ranked_url_classes.csv'
            if csv_path.exists():
                logger.info("No trained model found. Training new model...")
                self.train_model(csv_path, epochs=10)  # Increased epochs
            else:
                logger.warning("No trained model or training data found. Model will give random predictions.")
        
        self.model.eval()  # Set model to evaluation mode

    @classmethod
    def train_model(cls, csv_path: str, epochs: int = 10, batch_size: int = 32):
        """Train the model on the provided CSV data."""
        # Load and preprocess data
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} samples from {csv_path}")
        
        # Drop rows with missing values
        df = df.dropna()
        
        # Normalize ranks to [0, 1] with better scaling
        df['normalized_rank'] = 1 - ((df['Rank'] - df['Rank'].min()) / (df['Rank'].max() - df['Rank'].min()))
        
        # Initialize tokenizer and preprocessor
        tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
        preprocessor = URLPreprocessor()
        
        # Preprocess URLs and topics
        df['processed_url'] = df['Address'].apply(preprocessor.preprocess_url)
        df['processed_topic'] = df['Topic'].apply(preprocessor.preprocess_context)
        
        # Create dataset with processed data
        dataset = URLDataset(
            df['processed_url'].values,
            df['processed_topic'].values,
            df['normalized_rank'].values,
            tokenizer
        )
        
        # Split data with stratification
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        logger.info(f"Split data into {train_size} training and {val_size} validation samples")
        
        # Create data loaders with more workers
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            num_workers=4,
            pin_memory=True
        )
        
        # Initialize model and training
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = DeepURLRanker()
        model.to(device)
        
        # Use MSE loss with L1 regularization
        criterion = nn.MSELoss()
        
        # Use AdamW with cosine annealing
        optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        # Training loop with early stopping
        best_val_loss = float('inf')
        patience = 3
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            model.train()
            total_loss = 0
            num_batches = 0
            
            for batch in train_loader:
                optimizer.zero_grad()
                
                try:
                    url_ids = batch['url_ids'].to(device)
                    url_mask = batch['url_mask'].to(device)
                    topic_ids = batch['topic_ids'].to(device)
                    topic_mask = batch['topic_mask'].to(device)
                    ranks = batch['rank'].to(device)
                    
                    outputs = model(url_ids, url_mask, topic_ids, topic_mask)
                    loss = criterion(outputs, ranks)
                    
                    # Add L1 regularization
                    l1_lambda = 0.01
                    l1_norm = sum(p.abs().sum() for p in model.parameters())
                    loss = loss + l1_lambda * l1_norm
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                except Exception as e:
                    logger.error(f"Error in training batch: {str(e)}")
                    continue
            
            avg_train_loss = total_loss / num_batches if num_batches > 0 else float('inf')
            
            # Validation
            model.eval()
            val_loss = 0
            num_val_batches = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    try:
                        url_ids = batch['url_ids'].to(device)
                        url_mask = batch['url_mask'].to(device)
                        topic_ids = batch['topic_ids'].to(device)
                        topic_mask = batch['topic_mask'].to(device)
                        ranks = batch['rank'].to(device)
                        
                        outputs = model(url_ids, url_mask, topic_ids, topic_mask)
                        loss = criterion(outputs, ranks)
                        val_loss += loss.item()
                        num_val_batches += 1
                            
                    except Exception as e:
                        logger.error(f"Error in validation batch: {str(e)}")
                        continue
            
            avg_val_loss = val_loss / num_val_batches if num_val_batches > 0 else float('inf')
            
            logger.info(f"Epoch {epoch+1}/{epochs}")
            logger.info(f"Training Loss: {avg_train_loss:.4f}")
            logger.info(f"Validation Loss: {avg_val_loss:.4f}")
            
            # Learning rate scheduling
            scheduler.step()
            
            # Save best model and check early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                model_dir = Path(__file__).parent.parent.parent / 'models'
                model_dir.mkdir(exist_ok=True)
                model_path = model_dir / 'deep_ranker_v3.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'val_loss': avg_val_loss,
                }, model_path)
                logger.info(f"Saved best model checkpoint to {model_path}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break

    def score_url(self, url: str, context: str) -> float:
        """Score a URL's relevance to the given context."""
        self.model.eval()  # Ensure model is in eval mode
        with torch.no_grad():
            try:
                # Preprocess inputs
                processed_url = self.preprocessor.preprocess_url(url)
                processed_context = self.preprocessor.preprocess_context(context)
                
                # Tokenize inputs
                url_encoding = self.tokenizer(
                    processed_url,
                    padding='max_length',
                    max_length=128,
                    truncation=True,
                    return_tensors='pt'
                )
                
                topic_encoding = self.tokenizer(
                    processed_context,
                    padding='max_length',
                    max_length=64,
                    truncation=True,
                    return_tensors='pt'
                )
                
                # Move to device
                url_ids = url_encoding['input_ids'].to(self.device)
                url_mask = url_encoding['attention_mask'].to(self.device)
                topic_ids = topic_encoding['input_ids'].to(self.device)
                topic_mask = topic_encoding['attention_mask'].to(self.device)
                
                # Get prediction
                score = self.model(url_ids, url_mask, topic_ids, topic_mask)
                return float(score.cpu().numpy())
            except Exception as e:
                logger.error(f"Error scoring URL: {str(e)}")
                return 0.5  # Return neutral score on error
    
    def rank_urls(self, urls: List[str], context: str) -> List[Tuple[str, float]]:
        """Rank a list of URLs based on their relevance to the context."""
        # Score all URLs
        scores = [(url, self.score_url(url, context)) for url in urls]
        
        # Sort by score descending
        ranked = sorted(scores, key=lambda x: x[1], reverse=True)
        
        return ranked 
