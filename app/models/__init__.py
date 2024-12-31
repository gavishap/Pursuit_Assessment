"""
Database models package for SQLAlchemy ORM.

This package provides optimized database models and configurations for:
- Efficient batch operations
- Optimized indexing strategies
- Connection pooling and caching
- Memory-efficient query execution
"""

from .base import Base, engine, SessionLocal, get_db
from .link import Link

# Batch processing configuration
BATCH_CONFIG = {
    'chunk_size': 1000,  # Number of records to process at once
    'max_batch_size': 5000,  # Maximum number of records in a single transaction
    'commit_interval': 1000,  # Commit every N records
}

__all__ = [
    'Base', 
    'engine', 
    'SessionLocal', 
    'get_db', 
    'Link',
    'BATCH_CONFIG'
] 
