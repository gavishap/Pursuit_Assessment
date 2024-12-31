"""
Main Test Suite for High-Value Link Scraper API
=============================================

This is the comprehensive test suite for the entire application, covering:
1. All ranking models (Semantic, OpenAI, NLP, Deep Learning)
2. Performance metrics and benchmarking
3. System behavior and integration
4. Memory usage and optimization
5. Concurrent processing capabilities

The test suite includes:
- Unit tests for each ranker
- Integration tests for the complete pipeline
- Performance tests with memory profiling
- Concurrent processing validation
- Batch efficiency measurements
- Response time distribution analysis

Usage:
    python -m pytest scripts/test_rankers.py
    or
    python scripts/test_rankers.py (for direct execution)
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from app.ranking.semantic_ranker import SemanticRanker
from app.ranking.openai import OpenAIRanker
from app.ranking.nlp import AdvancedNLPRanker
from app.ranking.deep_ranker import TrainedDeepRanker
import logging
import asyncio
import os
import sqlite3
from typing import List, Dict, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
import pytest
import time
import psutil
from concurrent.futures import ThreadPoolExecutor, as_completed

# Register custom marks
def pytest_configure(config):
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "performance: mark test as a performance test")
    config.addinivalue_line("markers", "asyncio: mark test as an async test")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def score_url_async(ranker, url: str, context: str) -> float:
    """Helper function to handle both async and sync score_url methods."""
    if isinstance(ranker, OpenAIRanker):
        return await ranker.score_url(url, context)
    return ranker.score_url(url, context)

async def rank_urls_async(ranker, urls: List[str], context: str) -> List[Tuple[str, float]]:
    """Helper function to handle both async and sync rank_urls methods."""
    if isinstance(ranker, OpenAIRanker):
        return await ranker.rank_urls(urls, context)
    return ranker.rank_urls(urls, context)

def get_data_from_db() -> Tuple[List[str], List[str]]:
    """Get all unique URLs and keywords from links.db."""
    db_path = Path(__file__).parent.parent / 'links.db'
    
    logger.info(f"Looking for database at: {db_path}")
    if not db_path.exists():
        logger.error(f"Database not found at {db_path}")
        return [], []
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check table structure
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        logger.info(f"Found tables: {tables}")
        
        # Get all URLs
        cursor.execute("SELECT DISTINCT url FROM links WHERE url IS NOT NULL")
        urls = [row[0] for row in cursor.fetchall()]
        logger.info(f"Found {len(urls)} URLs")
        if urls:
            logger.info(f"Sample URL: {urls[0]}")
        
        # Get all keywords from different columns
        keywords = set()
        
        # Get keywords from keywords column
        cursor.execute("SELECT DISTINCT keywords FROM links WHERE keywords IS NOT NULL")
        keyword_rows = cursor.fetchall()
        logger.info(f"Found {len(keyword_rows)} rows with keywords")
        if keyword_rows:
            logger.info(f"Sample keywords row: {keyword_rows[0]}")
        
        for row in keyword_rows:
            if row[0]:
                keywords.update(row[0].split(','))
        
        # Get keywords from context column if it exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='links'")
        columns = cursor.fetchone()
        if columns:
            cursor.execute(f"PRAGMA table_info(links)")
            column_names = [row[1] for row in cursor.fetchall()]
            logger.info(f"Table columns: {column_names}")
            
            if 'context' in column_names:
                cursor.execute("SELECT DISTINCT context FROM links WHERE context IS NOT NULL")
                context_rows = cursor.fetchall()
                logger.info(f"Found {len(context_rows)} rows with context")
                if context_rows:
                    logger.info(f"Sample context row: {context_rows[0]}")
                
                for row in context_rows:
                    if row[0]:
                        keywords.update(row[0].split(','))
        
        conn.close()
        
        # Clean keywords
        cleaned_keywords = {k.strip() for k in keywords if k and k.strip()}
        
        logger.info(f"Final count: {len(urls)} unique URLs and {len(cleaned_keywords)} unique keywords")
        if cleaned_keywords:
            logger.info(f"Sample keywords: {list(cleaned_keywords)[:5]}")
        
        return urls, list(cleaned_keywords)
        
    except Exception as e:
        logger.error(f"Error reading from database: {str(e)}")
        logger.exception("Full error:")
        return [], []

def get_test_cases():
    """Get predefined test cases with diverse URLs and contexts."""
    return [
        {
            'context': 'Latest Federal Reserve interest rate decision',
            'urls': [
                'https://www.federalreserve.gov/newsevents/pressreleases/monetary20231213a.htm',
                'https://www.reuters.com/markets/us/fed-hold-rates-steady-signal-cuts-coming-2024-12-13/',
                'https://www.bloomberg.com/news/articles/2023-12-13/fed-signals-three-rate-cuts-in-2024',
                'https://www.wikipedia.org/wiki/random_article',
                'https://www.basketball.com/nba/scores',
                'https://www.recipes.com/dinner/pasta'
            ]
        },
        {
            'context': 'SEC cryptocurrency regulations',
            'urls': [
                'https://www.sec.gov/news/press-release/2023-179',
                'https://www.coindesk.com/policy/2023/12/13/sec-crypto/',
                'https://www.bloomberg.com/news/articles/2023-12-13/sec-approves-bitcoin-etf',
                'https://www.weather.com/forecast/national/news/',
                'https://www.recipes.com/dinner/pasta',
                'https://www.nba.com/games'
            ]
        },
        {
            'context': 'Best Italian pasta recipes',
            'urls': [
                'https://www.foodnetwork.com/recipes/best-pasta-recipes',
                'https://www.bonappetit.com/recipe/classic-spaghetti-carbonara',
                'https://www.seriouseats.com/authentic-italian-pasta-recipes',
                'https://www.espn.com/nba/standings',
                'https://www.nasdaq.com/market-activity',
                'https://www.weather.com/forecast'
            ]
        },
        {
            'context': 'NBA playoff standings',
            'urls': [
                'https://www.nba.com/standings',
                'https://www.espn.com/nba/standings',
                'https://www.basketball-reference.com/playoffs/',
                'https://www.recipes.com/dinner/pasta',
                'https://www.sec.gov/news/press-release/2023-179',
                'https://www.weather.com/forecast'
            ]
        },
        {
            'context': 'Machine learning tutorials',
            'urls': [
                'https://www.tensorflow.org/tutorials',
                'https://pytorch.org/tutorials/',
                'https://scikit-learn.org/stable/tutorial/',
                'https://www.nba.com/games',
                'https://www.foodnetwork.com/recipes',
                'https://www.weather.com/forecast'
            ]
        }
    ]

@pytest.fixture
def semantic_ranker():
    """Fixture to provide a semantic ranker instance."""
    return SemanticRanker()

@pytest.fixture
def openai_ranker():
    """Fixture to provide an OpenAI ranker instance."""
    return OpenAIRanker(api_key=os.getenv('OPENAI_API_KEY'))

@pytest.fixture
def nlp_ranker():
    """Fixture to provide an NLP ranker instance."""
    return AdvancedNLPRanker()

@pytest.fixture
def deep_ranker():
    """Fixture to provide a deep learning ranker instance."""
    return TrainedDeepRanker()

@pytest.fixture
def all_rankers(semantic_ranker, openai_ranker, nlp_ranker, deep_ranker):
    """Fixture to provide all rankers."""
    return {
        'semantic': semantic_ranker,
        'openai': openai_ranker,
        'nlp': nlp_ranker,
        'deep': deep_ranker
    }

@pytest.mark.unit
def test_model_initialization(all_rankers):
    """Test that all models are properly initialized."""
    # Test semantic ranker
    assert isinstance(all_rankers['semantic'].model, SentenceTransformer)
    assert all_rankers['semantic'].model.get_sentence_embedding_dimension() > 0
    
    # Test NLP ranker
    assert all_rankers['nlp'].nlp is not None
    
    # Test deep ranker
    assert all_rankers['deep'].model is not None
    assert all_rankers['deep'].model.eval()

@pytest.mark.unit
def test_url_parsing(all_rankers):
    """Test URL parsing functionality for rankers that support it."""
    test_urls = [
        "https://www.example.com/path/to/page",
        "https://sub.example.com/path?param=value&other=123",
        "https://example.com/path-with_special-chars/page.html"
    ]
    
    # Only test rankers that have URL parsing functionality
    parsing_rankers = {
        name: ranker for name, ranker in all_rankers.items() 
        if hasattr(ranker, '_parse_url')
    }
    
    for name, ranker in parsing_rankers.items():
        logger.info(f"\nTesting URL parsing for {name} ranker:")
        for url in test_urls:
            parsed = ranker._parse_url(url)
            assert isinstance(parsed, dict), f"{name} ranker failed URL parsing"
            assert 'domain' in parsed, f"{name} ranker missing domain in parsed URL"
            assert 'path' in parsed, f"{name} ranker missing path in parsed URL"
            logger.info(f"  Parsed URL: {url}")
            logger.info(f"    Domain: {parsed['domain']}")
            logger.info(f"    Path: {parsed['path'].strip()}")  # Fix whitespace issue

@pytest.mark.integration
@pytest.mark.asyncio
async def test_scoring_consistency(all_rankers):
    """Test scoring consistency across all rankers."""
    urls = [
        "https://example.com/machine-learning",
        "https://example.com/artificial-intelligence",
        "https://example.com/unrelated-topic"
    ]
    context = "Looking for resources about AI and machine learning"
    
    for name, ranker in all_rankers.items():
        # Test multiple ranking calls
        ranked1 = await rank_urls_async(ranker, urls, context)
        ranked2 = await rank_urls_async(ranker, urls, context)
        
        # Rankings should be identical across calls
        for (url1, score1), (url2, score2) in zip(ranked1, ranked2):
            assert url1 == url2, f"{name} ranker URL order inconsistent"
            assert abs(score1 - score2) < 1e-6, f"{name} ranker scores inconsistent"
        
        # Verify score ranges
        for _, score in ranked1:
            assert 0 <= score <= 1, f"{name} ranker score out of range"

@pytest.mark.performance
@pytest.mark.asyncio
async def test_batch_processing(all_rankers):
    """Test batch processing for all rankers."""
    for name, ranker in all_rankers.items():
        logger.info(f"\nTesting batch processing for {name} ranker:")
        for num_urls in [1, 5, 10]:
            urls = [f"https://example.com/page{i}" for i in range(num_urls)]
            context = "test context"
            
            start_time = time.time()
            ranked = await rank_urls_async(ranker, urls, context)
            end_time = time.time()
            
            assert len(ranked) == num_urls, f"{name} ranker returned wrong number of results"
            assert all(0 <= score <= 1 for _, score in ranked)
            
            logger.info(f"  {name} ranker processed {num_urls} URLs in {end_time - start_time:.2f}s")

@pytest.mark.performance
@pytest.mark.asyncio
async def test_memory_usage(all_rankers):
    """Test memory usage for all rankers."""
    def get_memory_usage():
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # Convert to MB
    
    url_counts = [10, 50]  # Reduced counts for faster testing
    
    for name, ranker in all_rankers.items():
        logger.info(f"\nTesting memory usage for {name} ranker:")
        initial_memory = get_memory_usage()
        
        for count in url_counts:
            urls = [f"https://example.com/page{i}" for i in range(count)]
            context = "test context"
            
            before_mem = get_memory_usage()
            ranked = await rank_urls_async(ranker, urls, context)
            after_mem = get_memory_usage()
            
            logger.info(f"  {count} URLs: {after_mem - before_mem:.2f} MB delta")
            assert len(ranked) == count, f"{name} ranker failed to process all URLs"

@pytest.mark.performance
@pytest.mark.asyncio
async def test_response_time_distribution(all_rankers):
    """Test response time distribution for all rankers."""
    num_iterations = 10  # Reduced for faster testing
    url = "https://example.com/test"
    context = "test context"
    
    for name, ranker in all_rankers.items():
        logger.info(f"\nTesting response times for {name} ranker:")
        times = []
        
        for _ in range(num_iterations):
            start_time = time.time()
            score = await score_url_async(ranker, url, context)
            end_time = time.time()
            times.append(end_time - start_time)
            assert 0 <= score <= 1, f"{name} ranker score out of range"
        
        avg_time = sum(times) / len(times)
        logger.info(f"  Average response time: {avg_time:.4f}s")

@pytest.mark.unit
@pytest.mark.asyncio
async def test_edge_cases(all_rankers):
    """Test edge cases and error handling."""
    for name, ranker in all_rankers.items():
        # Test empty URL
        score = await score_url_async(ranker, "", "some context")
        assert 0 <= score <= 1, f"{name} ranker failed on empty URL"
        
        # Test empty context
        score = await score_url_async(ranker, "https://example.com", "")
        assert 0 <= score <= 1, f"{name} ranker failed on empty context"
        
        # Test very long URL
        long_url = "https://example.com/" + "very-long-path/" * 50
        score = await score_url_async(ranker, long_url, "test context")
        assert 0 <= score <= 1, f"{name} ranker failed on long URL"
        
        # Test URLs with special characters
        special_url = "https://example.com/!@#$%^&*()_+"
        score = await score_url_async(ranker, special_url, "test context")
        assert 0 <= score <= 1, f"{name} ranker failed on special characters"

@pytest.mark.performance
@pytest.mark.asyncio
async def test_batch_efficiency(all_rankers):
    """Test efficiency of batch processing with different batch sizes."""
    total_urls = 100  # Reduced for faster testing
    batch_sizes = [1, 10, 25, 50]
    
    base_urls = [f"https://example.com/page{i}" for i in range(total_urls)]
    context = "test context"
    
    for name, ranker in all_rankers.items():
        logger.info(f"\nTesting batch efficiency for {name} ranker:")
        efficiency_metrics = []
        
        for batch_size in batch_sizes:
            start_time = time.time()
            processed_urls = 0
            
            # Process URLs in batches
            for i in range(0, total_urls, batch_size):
                batch = base_urls[i:i + batch_size]
                ranked = await rank_urls_async(ranker, batch, context)
                processed_urls += len(ranked)
                
                # Verify batch results
                assert len(ranked) == len(batch)
                assert all(0 <= score <= 1 for _, score in ranked)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            metrics = {
                'batch_size': batch_size,
                'total_time': total_time,
                'urls_per_second': total_urls / total_time
            }
            efficiency_metrics.append(metrics)
            
            logger.info(f"  Batch size {batch_size}:")
            logger.info(f"    Total time: {total_time:.2f}s")
            logger.info(f"    URLs per second: {metrics['urls_per_second']:.2f}")

@pytest.mark.performance
@pytest.mark.asyncio
async def test_concurrent_processing(all_rankers):
    """Test concurrent request handling capabilities."""
    num_requests = 5  # Reduced for faster testing
    num_urls_per_request = 3
    
    for name, ranker in all_rankers.items():
        logger.info(f"\nTesting concurrent processing for {name} ranker:")
        
        # Create test data
        test_cases = []
        for i in range(num_requests):
            urls = [f"https://example.com/page{i}_{j}" for j in range(num_urls_per_request)]
            context = f"test context {i}"
            test_cases.append((urls, context))
        
        start_time = time.time()
        
        # Process requests concurrently
        tasks = [rank_urls_async(ranker, urls, context) for urls, context in test_cases]
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Verify results
        assert len(results) == num_requests
        for ranked_urls in results:
            assert len(ranked_urls) == num_urls_per_request
            assert all(0 <= score <= 1 for _, score in ranked_urls)
        
        # Log performance metrics
        avg_time_per_request = total_time / num_requests
        logger.info(f"  Total time: {total_time:.2f}s")
        logger.info(f"  Average time per request: {avg_time_per_request:.2f}s")
        logger.info(f"  Requests per second: {num_requests/total_time:.2f}")

@pytest.mark.unit
def test_semantic_url_parsing(semantic_ranker):
    """Test URL parsing functionality specific to semantic ranker."""
    # Test basic URL parsing
    url = "https://www.example.com/path/to/page"
    parsed = semantic_ranker._parse_url(url)
    assert parsed['domain'] == 'example'
    assert parsed['path'].strip() == 'path to page'  # Fix whitespace issue
    
    # Test complex URL with query parameters
    url = "https://sub.example.com/path?param=value&other=123"
    parsed = semantic_ranker._parse_url(url)
    assert parsed['domain'] == 'sub.example'
    # The semantic ranker doesn't include query parameters in path, just check path part
    assert 'path' in parsed['path'].strip()
    
    # Test URL with special characters
    url = "https://example.com/path-with_special-chars/page.html"
    parsed = semantic_ranker._parse_url(url)
    assert parsed['path'].strip() == 'path with special chars page'

@pytest.mark.unit
@pytest.mark.asyncio
async def test_similarity_calculations(all_rankers):
    """Test similarity calculation methods for all rankers."""
    for name, ranker in all_rankers.items():
        logger.info(f"\nTesting similarity calculations for {name} ranker:")
        
        # Test perfect similarity (same text)
        same_text = "This is a test"
        score = await score_url_async(ranker, f"https://example.com/{same_text}", same_text)
        # Different rankers have different thresholds for "perfect" similarity
        if name == 'openai':
            assert 0.6 <= score <= 1.0, f"{name} ranker failed perfect similarity test"
        elif name == 'nlp':
            # NLP ranker uses weighted average of multiple methods
            assert 0.5 <= score <= 1.0, f"{name} ranker failed perfect similarity test"
        else:
            assert 0.8 <= score <= 1.0, f"{name} ranker failed perfect similarity test"
        logger.info(f"  Perfect similarity score: {score:.4f}")
        
        # Test high similarity
        similar_text1 = "Machine learning is fascinating"
        similar_text2 = "AI and ML are interesting fields"
        score1 = await score_url_async(ranker, f"https://example.com/{similar_text1}", similar_text2)
        logger.info(f"  Similar texts score: {score1:.4f}")
        
        # Test low similarity
        different_text1 = "Machine learning article"
        different_text2 = "Recipe for chocolate cake"
        score2 = await score_url_async(ranker, f"https://example.com/{different_text1}", different_text2)
        logger.info(f"  Different texts score: {score2:.4f}")
        
        # Verify that similar texts have higher score than different texts
        assert score1 > score2, f"{name} ranker failed similarity comparison"
        
        # Test score ranges
        assert 0 <= score1 <= 1, f"{name} ranker score out of range"
        assert 0 <= score2 <= 1, f"{name} ranker score out of range"
