from sqlalchemy import Column, Integer, String, Float, DateTime, JSON
from datetime import datetime
from .base import Base

class Link(Base):
    __tablename__ = "links"

    id = Column(Integer, primary_key=True, index=True)
    url = Column(String, unique=True, index=True)
    title = Column(String, nullable=True)
    content_type = Column(String, nullable=True)
    relevance_score = Column(Float, index=True)
    keywords = Column(JSON, nullable=True)
    link_metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow) 
