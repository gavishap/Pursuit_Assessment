from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, ForeignKey, Index, text
from sqlalchemy.dialects.postgresql import JSONB
from datetime import datetime
from .base import Base, engine

class Link(Base):
    """Model for storing and retrieving links with optimized indexing for large datasets."""
    
    __tablename__ = "links"

    id = Column(Integer, primary_key=True, index=True)
    base_url = Column(String(2048), index=True)  # Fixed length for better indexing
    url = Column(String(2048), index=True)  # Fixed length for better indexing
    title = Column(String(512), nullable=True)  # Reasonable limit for titles
    content_type = Column(String(128), nullable=True)  # Fixed length for MIME types
    
    # Individual ranking scores with index for range queries
    semantic_score = Column(Float, index=True)
    openai_score = Column(Float, index=True)
    nlp_score = Column(Float, index=True)
    deep_learning_score = Column(Float, index=True)
    ensemble_score = Column(Float, index=True)  # Average of all scores
    
    # Use JSONB for PostgreSQL, JSON for other databases
    keywords = Column(JSONB if 'postgresql' in engine.url.drivername else JSON, nullable=True)
    link_metadata = Column(JSONB if 'postgresql' in engine.url.drivername else JSON, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, index=True)
    
    # Optimized composite indexes for common query patterns
    __table_args__ = (
        # Index for filtering by base_url and sorting by ensemble score
        Index('idx_base_url_ensemble', 'base_url', 'ensemble_score'),
        # Index for time-based queries with score filtering
        Index('idx_created_ensemble', 'created_at', 'ensemble_score'),
        # Index for content type filtering with score
        Index('idx_content_ensemble', 'content_type', 'ensemble_score'),
        # Partial index for high-relevance links
        Index('idx_high_relevance', 'ensemble_score', postgresql_where=text("ensemble_score >= 0.8")),
    )
    
    def __repr__(self):
        """String representation for debugging."""
        return f"<Link(id={self.id}, url='{self.url}', ensemble_score={self.ensemble_score})>" 
