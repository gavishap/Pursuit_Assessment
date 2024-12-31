"""
Core functionality package for web scraping and data processing.

This package provides optimized implementations for:
- Concurrent web scraping with rate limiting and retries
- Efficient data processing with batching and streaming
- Memory-efficient operations for large datasets
"""

# Default configuration for scalability
SCRAPER_CONFIG = {
    'max_concurrent_requests': 10,  # Maximum number of concurrent requests
    'rate_limit': 1.0,  # Minimum time between requests to same domain
    'batch_size': 100,  # Number of items to process in a batch
    'max_retries': 3,  # Maximum number of retry attempts
    'timeout': 30,  # Request timeout in seconds
    'memory_limit': '2GB',  # Maximum memory usage for processing
}

# Import core components
from .scraper import LinkScraper

__all__ = ['LinkScraper', 'SCRAPER_CONFIG'] 
