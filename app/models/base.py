from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./links.db")

# Configure engine with optimized settings for large data
engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=20,  # Maximum number of database connections in the pool
    max_overflow=10,  # Maximum number of connections that can be created beyond pool_size
    pool_timeout=30,  # Timeout for getting a connection from the pool
    pool_recycle=1800,  # Recycle connections after 30 minutes
    pool_pre_ping=True,  # Enable connection health checks
)

# Enable SQLite optimizations if using SQLite
if DATABASE_URL.startswith('sqlite'):
    @event.listens_for(engine, 'connect')
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        # Enable WAL mode for better concurrent access
        cursor.execute("PRAGMA journal_mode=WAL")
        # Increase cache size for better performance
        cursor.execute("PRAGMA cache_size=-2000000")  # Use 2GB of memory for cache
        # Enable memory-mapped I/O for better performance
        cursor.execute("PRAGMA mmap_size=2147483648")  # 2GB
        cursor.close()

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    expire_on_commit=False  # Prevent unnecessary database hits
)

Base = declarative_base()

def get_db():
    """Get a database session with optimized settings."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close() 
