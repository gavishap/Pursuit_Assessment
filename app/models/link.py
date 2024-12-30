from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, ForeignKey, Index
from datetime import datetime
from .base import Base

class Link(Base):
    __tablename__ = "links"

    id = Column(Integer, primary_key=True, index=True)
    base_url = Column(String, index=True)  # URL that was scraped to find this link
    url = Column(String, index=True)  # The actual link found
    title = Column(String, nullable=True)
    content_type = Column(String, nullable=True)
    
    # Individual ranking scores
    semantic_score = Column(Float, index=True)
    openai_score = Column(Float, index=True)
    nlp_score = Column(Float, index=True)
    deep_learning_score = Column(Float, index=True)
    ensemble_score = Column(Float, index=True)  # Average of all scores
    
    keywords = Column(JSON, nullable=True)  # Keywords found in the link
    link_metadata = Column(JSON, nullable=True)  # Additional metadata (content snippet, etc.)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Create composite index for efficient querying
    __table_args__ = (
        Index('idx_base_url_ensemble', 'base_url', 'ensemble_score'),
    ) 
