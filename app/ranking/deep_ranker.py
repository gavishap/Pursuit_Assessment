import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import logging
from pathlib import Path
from typing import List, Tuple
import numpy as np

logger = logging.getLogger(__name__)

class URLDataset(Dataset):
    def __init__(self, urls, topics, ranks, tokenizer, max_url_length=128, max_topic_length=32):
        self.urls = urls
        self.topics = topics
        self.ranks = ranks
        self.tokenizer = tokenizer
        self.max_url_length = max_url_length
        self.max_topic_length = max_topic_length

    def __len__(self):
        return len(self.urls)

    def __getitem__(self, idx):
        url = str(self.urls[idx])
        topic = str(self.topics[idx])
        rank = float(self.ranks[idx])

        # Tokenize with truncation and padding
        url_encoding = self.tokenizer(
            url,
            padding='max_length',
            max_length=self.max_url_length,
            truncation=True,
            return_tensors='pt'
        )
        
        topic_encoding = self.tokenizer(
            topic,
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
    def __init__(self, hidden_size=384):
        super().__init__()
        # Use a smaller, faster model
        self.encoder = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        
        # Freeze encoder weights for faster training
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        self.url_proj = nn.Linear(384, hidden_size)
        self.topic_proj = nn.Linear(384, hidden_size)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()  # Ensure output is between 0 and 1
        )

    def forward(self, url_ids, url_mask, topic_ids, topic_mask):
        # Get embeddings
        url_output = self.encoder(url_ids, attention_mask=url_mask).last_hidden_state[:, 0]
        topic_output = self.encoder(topic_ids, attention_mask=topic_mask).last_hidden_state[:, 0]
        
        # Project embeddings
        url_emb = self.url_proj(url_output)
        topic_emb = self.topic_proj(topic_output)
        
        # Concatenate and classify
        combined = torch.cat([url_emb, topic_emb], dim=1)
        score = self.classifier(combined)
        
        return score.squeeze()

class TrainedDeepRanker:
    """A ranker that uses a trained deep learning model to score URLs."""
    
    def __init__(self):
        """Initialize the deep learning ranker."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = DeepURLRanker()
        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        
        # Load trained model if it exists
        model_path = Path(__file__).parent.parent.parent / 'models' / 'deep_ranker.pt'
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
                self.train_model(csv_path, epochs=5)
            else:
                logger.warning("No trained model or training data found. Model will give random predictions.")
        
        self.model.eval()  # Set model to evaluation mode

    @classmethod
    def train_model(cls, csv_path: str, epochs: int = 5, batch_size: int = 64):
        """Train the model on the provided CSV data."""
        # Load and preprocess data
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} samples from {csv_path}")
        
        # Drop rows with missing values
        df = df.dropna()
        
        # Normalize ranks to [0, 1]
        df['normalized_rank'] = (df['Rank'] - df['Rank'].min()) / (df['Rank'].max() - df['Rank'].min())
        
        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        
        # Create dataset
        dataset = URLDataset(df['Address'].values, df['Topic'].values, df['normalized_rank'].values, tokenizer)
        
        # Split data
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        logger.info(f"Split data into {train_size} training and {val_size} validation samples")
        
        # Create data loaders with more workers for faster loading
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)
        
        # Initialize model and training
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = DeepURLRanker()
        model.to(device)
        
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=1e-4)
        
        # Training loop
        best_val_loss = float('inf')
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
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                model_dir = Path(__file__).parent.parent.parent / 'models'
                model_dir.mkdir(exist_ok=True)
                model_path = model_dir / 'deep_ranker.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'val_loss': avg_val_loss,
                }, model_path)
                logger.info(f"Saved best model checkpoint to {model_path}")

    def score_url(self, url: str, context: str) -> float:
        """Score a URL's relevance to the given context."""
        self.model.eval()  # Ensure model is in eval mode
        with torch.no_grad():
            try:
                # Tokenize inputs
                url_encoding = self.tokenizer(
                    url,
                    padding='max_length',
                    max_length=128,
                    truncation=True,
                    return_tensors='pt'
                )
                
                topic_encoding = self.tokenizer(
                    context,
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
