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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    """Test URL parsing functionality for all rankers."""
    test_urls = [
        "https://www.example.com/path/to/page",
        "https://sub.example.com/path?param=value&other=123",
        "https://example.com/path-with_special-chars/page.html"
    ]
    
    for name, ranker in all_rankers.items():
        for url in test_urls:
            parsed = ranker._parse_url(url)
            assert isinstance(parsed, dict), f"{name} ranker failed URL parsing"
            assert 'domain' in parsed, f"{name} ranker missing domain in parsed URL"
            assert 'path' in parsed, f"{name} ranker missing path in parsed URL"

@pytest.mark.integration
def test_scoring_consistency(all_rankers):
    """Test scoring consistency across all rankers."""
    urls = [
        "https://example.com/machine-learning",
        "https://example.com/artificial-intelligence",
        "https://example.com/unrelated-topic"
    ]
    context = "Looking for resources about AI and machine learning"
    
    for name, ranker in all_rankers.items():
        # Test multiple ranking calls
        ranked1 = ranker.rank_urls(urls, context)
        ranked2 = ranker.rank_urls(urls, context)
        
        # Rankings should be identical across calls
        for (url1, score1), (url2, score2) in zip(ranked1, ranked2):
            assert url1 == url2, f"{name} ranker URL order inconsistent"
            assert abs(score1 - score2) < 1e-6, f"{name} ranker scores inconsistent"
        
        # Verify score ranges
        for _, score in ranked1:
            assert 0 <= score <= 1, f"{name} ranker score out of range"

@pytest.mark.performance
def test_batch_processing(all_rankers):
    """Test batch processing for all rankers."""
    for name, ranker in all_rankers.items():
        logger.info(f"\nTesting batch processing for {name} ranker:")
        for num_urls in [1, 5, 10]:
            urls = [f"https://example.com/page{i}" for i in range(num_urls)]
            context = "test context"
            
            start_time = time.time()
            ranked = ranker.rank_urls(urls, context)
            end_time = time.time()
            
            assert len(ranked) == num_urls, f"{name} ranker returned wrong number of results"
            assert all(0 <= score <= 1 for _, score in ranked), f"{name} ranker scores out of range"
            
            logger.info(f"  {name} ranker processed {num_urls} URLs in {end_time - start_time:.2f}s")

@pytest.mark.performance
def test_memory_usage(all_rankers):
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
            ranked = ranker.rank_urls(urls, context)
            after_mem = get_memory_usage()
            
            logger.info(f"  {count} URLs: {after_mem - before_mem:.2f} MB delta")
            assert len(ranked) == count, f"{name} ranker failed to process all URLs"

@pytest.mark.performance
def test_response_time_distribution(all_rankers):
    """Test response time distribution for all rankers."""
    num_iterations = 10  # Reduced for faster testing
    url = "https://example.com/test"
    context = "test context"
    
    for name, ranker in all_rankers.items():
        logger.info(f"\nTesting response times for {name} ranker:")
        times = []
        
        for _ in range(num_iterations):
            start_time = time.time()
            score = ranker.score_url(url, context)
            end_time = time.time()
            times.append(end_time - start_time)
            assert 0 <= score <= 1, f"{name} ranker score out of range"
        
        avg_time = sum(times) / len(times)
        logger.info(f"  Average response time: {avg_time:.4f}s")

@pytest.mark.integration
@pytest.mark.asyncio
async def test_all_rankers():
    """Integration test for all rankers with detailed analysis."""
    semantic_ranker = SemanticRanker()
    openai_ranker = OpenAIRanker(api_key=os.getenv('OPENAI_API_KEY'))
    nlp_ranker = AdvancedNLPRanker()
    deep_ranker = TrainedDeepRanker()
    
    # Get URLs and keywords from database
    urls, keywords = get_data_from_db()
    
    # Get predefined test cases
    test_cases = get_test_cases()
    
    # If database has data, add some test cases using database URLs
    if urls and keywords:
        logger.info("Adding test cases from database...")
        import random
        
        # Take first 5 keywords to avoid too many tests
        for keyword in keywords[:5]:
            # Take random sample of 6 URLs for each keyword
            test_urls = random.sample(urls, min(6, len(urls)))
            test_cases.append({
                'context': keyword,
                'urls': test_urls
            })
    
    # Run tests
    for test in test_cases:
        logger.info("\n" + "=" * 120)
        logger.info(f"\nTEST CASE: {test['context']}")
        logger.info("=" * 120)
        
        # Get rankings from all rankers
        semantic_ranked = semantic_ranker.rank_urls(test['urls'], test['context'])
        openai_ranked = await openai_ranker.rank_urls(test['urls'], test['context'])
        nlp_ranked = nlp_ranker.rank_urls(test['urls'], test['context'])
        deep_ranked = deep_ranker.rank_urls(test['urls'], test['context'])
        
        # Verify all rankers returned valid results
        for ranked_list in [semantic_ranked, openai_ranked, nlp_ranked, deep_ranked]:
            assert len(ranked_list) == len(test['urls'])
            assert all(0 <= score <= 1 for _, score in ranked_list)
        
        # Combine and sort results
        combined_results = []
        for url in test['urls']:
            semantic_score = next(score for u, score in semantic_ranked if u == url)
            openai_score = next(score for u, score in openai_ranked if u == url)
            nlp_score = next(score for u, score in nlp_ranked if u == url)
            deep_score = next(score for u, score in deep_ranked if u == url)
            
            # Calculate ensemble score (weighted average)
            ensemble_score = (
                0.25 * semantic_score +  # BERT-based semantic similarity
                0.25 * openai_score +    # GPT-based analysis
                0.25 * nlp_score +       # Comprehensive NLP analysis
                0.25 * deep_score        # Trained deep learning model
            )
            
            combined_results.append({
                'url': url,
                'semantic_score': semantic_score,
                'openai_score': openai_score,
                'nlp_score': nlp_score,
                'deep_score': deep_score,
                'ensemble_score': ensemble_score
            })
        
        # Sort by ensemble score
        combined_results.sort(key=lambda x: x['ensemble_score'], reverse=True)
        
        # Print results table
        logger.info("\n{:<40} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
            "URL", "Semantic", "OpenAI", "NLP", "Deep", "Ensemble"
        ))
        logger.info("-" * 100)
        
        for result in combined_results:
            url_display = result['url'][:37] + "..." if len(result['url']) > 40 else result['url']
            logger.info("{:<40} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f}".format(
                url_display,
                result['semantic_score'],
                result['openai_score'],
                result['nlp_score'],
                result['deep_score'],
                result['ensemble_score']
            ))
        
        # Calculate and verify statistics
        semantic_scores = [r['semantic_score'] for r in combined_results]
        openai_scores = [r['openai_score'] for r in combined_results]
        nlp_scores = [r['nlp_score'] for r in combined_results]
        deep_scores = [r['deep_score'] for r in combined_results]
        ensemble_scores = [r['ensemble_score'] for r in combined_results]
        
        # Calculate means
        semantic_mean = np.mean(semantic_scores)
        openai_mean = np.mean(openai_scores)
        nlp_mean = np.mean(nlp_scores)
        deep_mean = np.mean(deep_scores)
        ensemble_mean = np.mean(ensemble_scores)
        
        logger.info("\nSUMMARY STATISTICS:")
        logger.info("-" * 30)
        logger.info(f"Average Scores:")
        logger.info(f"  Semantic:  {semantic_mean:.4f}")
        logger.info(f"  OpenAI:    {openai_mean:.4f}")
        logger.info(f"  NLP:       {nlp_mean:.4f}")
        logger.info(f"  Deep:      {deep_mean:.4f}")
        logger.info(f"  Ensemble:  {ensemble_mean:.4f}")
        
        # Calculate and verify correlations
        correlations = {
            'Semantic-OpenAI': calculate_correlation(semantic_scores, openai_scores),
            'Semantic-NLP': calculate_correlation(semantic_scores, nlp_scores),
            'Semantic-Deep': calculate_correlation(semantic_scores, deep_scores),
            'OpenAI-NLP': calculate_correlation(openai_scores, nlp_scores),
            'OpenAI-Deep': calculate_correlation(openai_scores, deep_scores),
            'NLP-Deep': calculate_correlation(nlp_scores, deep_scores)
        }
        
        logger.info("\nScore Correlations:")
        for pair, corr in correlations.items():
            logger.info(f"  {pair}: {corr:.4f}")
            # Verify reasonable correlation (not too low or perfect)
            assert -1 <= corr <= 1, f"Invalid correlation for {pair}"
            assert abs(corr) < 0.99999, f"Suspiciously perfect correlation for {pair}" 

def calculate_correlation(x, y):
    """Calculate Pearson correlation coefficient between two lists."""
    if len(x) != len(y):
        return 0.0
    n = len(x)
    if n == 0:
        return 0.0
    
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    
    variance_x = sum((xi - mean_x) ** 2 for xi in x)
    variance_y = sum((yi - mean_y) ** 2 for yi in y)
    
    if variance_x == 0 or variance_y == 0:
        return 0.0
    
    covariance = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    correlation = covariance / (variance_x ** 0.5 * variance_y ** 0.5)
    
    return correlation

@pytest.mark.unit
def test_embedding_quality(semantic_ranker):
    """Test the quality of embeddings generated."""
    # Test embedding dimensionality
    texts = ["This is a test sentence"]
    embeddings = semantic_ranker._get_embeddings(texts)
    assert embeddings.shape[1] == semantic_ranker.model.get_sentence_embedding_dimension()
    
    # Test embedding normalization
    embedding = embeddings[0]
    norm = np.linalg.norm(embedding)
    assert abs(norm - 1.0) < 1e-6  # Should be approximately unit norm
    
    # Test semantic similarity preservation
    similar_texts = [
        "Machine learning is fascinating",
        "AI and ML are interesting fields"
    ]
    different_texts = [
        "Machine learning is fascinating",
        "I love playing basketball"
    ]
    
    similar_embeddings = semantic_ranker._get_embeddings(similar_texts)
    different_embeddings = semantic_ranker._get_embeddings(different_texts)
    
    similar_score = semantic_ranker._compute_similarity(similar_embeddings[0], similar_embeddings[1])
    different_score = semantic_ranker._compute_similarity(different_embeddings[0], different_embeddings[1])
    
    assert similar_score > different_score

@pytest.mark.unit
def test_edge_cases(all_rankers):
    """Test edge cases and error handling."""
    for name, ranker in all_rankers.items():
        # Test empty URL
        score = ranker.score_url("", "some context")
        assert 0 <= score <= 1, f"{name} ranker failed on empty URL"
        
        # Test empty context
        score = ranker.score_url("https://example.com", "")
        assert 0 <= score <= 1, f"{name} ranker failed on empty context"
        
        # Test very long URL
        long_url = "https://example.com/" + "very-long-path/" * 50
        score = ranker.score_url(long_url, "test context")
        assert 0 <= score <= 1, f"{name} ranker failed on long URL"
        
        # Test URLs with special characters
        special_url = "https://example.com/!@#$%^&*()_+"
        score = ranker.score_url(special_url, "test context")
        assert 0 <= score <= 1, f"{name} ranker failed on special characters"

@pytest.mark.performance
def test_batch_efficiency(all_rankers):
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
                ranked = ranker.rank_urls(batch, context)
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
def test_concurrent_processing(all_rankers):
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
        
        # Process requests concurrently using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(ranker.rank_urls, urls, context)
                for urls, context in test_cases
            ]
            
            # Collect results
            results = []
            for future in as_completed(futures):
                results.append(future.result())
        
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
    assert parsed['path'] == 'path to page'
    
    # Test complex URL with query parameters
    url = "https://sub.example.com/path?param=value&other=123"
    parsed = semantic_ranker._parse_url(url)
    assert parsed['domain'] == 'sub.example'
    assert 'param' in parsed['path']
    
    # Test URL with special characters
    url = "https://example.com/path-with_special-chars/page.html"
    parsed = semantic_ranker._parse_url(url)
    assert parsed['path'] == 'path with special chars page'

@pytest.mark.unit
def test_semantic_similarity_calculations(semantic_ranker):
    """Test similarity calculation methods specific to semantic ranker."""
    # Test perfect similarity
    same_text = "This is a test"
    embedding = semantic_ranker._get_embeddings([same_text])[0]
    similarity = semantic_ranker._compute_similarity(embedding, embedding)
    assert abs(similarity - 1.0) < 1e-6
    
    # Test dissimilar texts
    text1 = "Machine learning article"
    text2 = "Recipe for chocolate cake"
    emb1 = semantic_ranker._get_embeddings([text1])[0]
    emb2 = semantic_ranker._get_embeddings([text2])[0]
    similarity = semantic_ranker._compute_similarity(emb1, emb2)
    assert 0 <= similarity <= 1

@pytest.mark.performance
def test_semantic_response_time_distribution(semantic_ranker):
    """Test distribution of response times under various conditions for semantic ranker."""
    num_iterations = 100
    
    # Test single URL ranking
    single_url_times = []
    url = "https://example.com/test"
    context = "test context"
    
    for _ in range(num_iterations):
        start_time = time.time()
        score = semantic_ranker.score_url(url, context)
        end_time = time.time()
        single_url_times.append(end_time - start_time)
    
    # Calculate statistics
    avg_time = sum(single_url_times) / len(single_url_times)
    max_time = max(single_url_times)
    min_time = min(single_url_times)
    p95_time = sorted(single_url_times)[int(0.95 * len(single_url_times))]
    
    logger.info("\nSemantic ranker response time distribution:")
    logger.info(f"  Average: {avg_time:.4f}s")
    logger.info(f"  P95: {p95_time:.4f}s")
    logger.info(f"  Min: {min_time:.4f}s")
    logger.info(f"  Max: {max_time:.4f}s")
    
    # Test response time under load
    urls = [f"https://example.com/page{i}" for i in range(10)]
    batch_times = []
    
    for _ in range(num_iterations):
        start_time = time.time()
        ranked = semantic_ranker.rank_urls(urls, context)
        end_time = time.time()
        batch_times.append(end_time - start_time)
    
    # Calculate batch statistics
    avg_batch_time = sum(batch_times) / len(batch_times)
    max_batch_time = max(batch_times)
    min_batch_time = min(batch_times)
    p95_batch_time = sorted(batch_times)[int(0.95 * len(batch_times))]
    
    logger.info("\nSemantic ranker batch response time distribution:")
    logger.info(f"  Average: {avg_batch_time:.4f}s")
    logger.info(f"  P95: {p95_batch_time:.4f}s")
    logger.info(f"  Min: {min_batch_time:.4f}s")
    logger.info(f"  Max: {max_batch_time:.4f}s")

@pytest.mark.performance
def test_semantic_memory_usage(semantic_ranker):
    """Test memory usage during ranking operations for semantic ranker."""
    def get_memory_usage():
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # Convert to MB
    
    # Baseline memory usage
    initial_memory = get_memory_usage()
    logger.info(f"\nSemantic ranker initial memory usage: {initial_memory:.2f} MB")
    
    # Test with increasing load
    url_counts = [10, 50, 100, 200]
    memory_usage = []
    
    for count in url_counts:
        urls = [f"https://example.com/page{i}" for i in range(count)]
        context = "test context"
        
        # Measure memory before operation
        before_mem = get_memory_usage()
        
        # Perform ranking
        start_time = time.time()
        ranked = semantic_ranker.rank_urls(urls, context)
        end_time = time.time()
        
        # Measure memory after operation
        after_mem = get_memory_usage()
        memory_delta = after_mem - before_mem
        
        memory_usage.append({
            'url_count': count,
            'memory_delta': memory_delta,
            'processing_time': end_time - start_time
        })
        
        logger.info(f"Semantic ranker memory usage for {count} URLs:")
        logger.info(f"  Delta: {memory_delta:.2f} MB")
        logger.info(f"  Processing time: {end_time - start_time:.2f}s")
        
        # Verify results
        assert len(ranked) == count
        assert all(0 <= score <= 1 for _, score in ranked)

@pytest.mark.performance
def test_semantic_batch_efficiency(semantic_ranker):
    """Test efficiency of batch processing with different batch sizes for semantic ranker."""
    total_urls = 1000
    batch_sizes = [1, 10, 50, 100, 200]
    
    base_urls = [f"https://example.com/page{i}" for i in range(total_urls)]
    context = "test context"
    
    efficiency_metrics = []
    
    for batch_size in batch_sizes:
        start_time = time.time()
        processed_urls = 0
        
        # Process URLs in batches
        for i in range(0, total_urls, batch_size):
            batch = base_urls[i:i + batch_size]
            ranked = semantic_ranker.rank_urls(batch, context)
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
        
        logger.info(f"\nSemantic ranker batch size {batch_size} metrics:")
        logger.info(f"  Total time: {total_time:.2f}s")
        logger.info(f"  URLs per second: {metrics['urls_per_second']:.2f}")

@pytest.mark.performance
def test_semantic_concurrent_processing(semantic_ranker):
    """Test concurrent request handling capabilities for semantic ranker."""
    num_requests = 10
    num_urls_per_request = 5
    
    # Create test data
    test_cases = []
    for i in range(num_requests):
        urls = [f"https://example.com/page{i}_{j}" for j in range(num_urls_per_request)]
        context = f"test context {i}"
        test_cases.append((urls, context))
    
    start_time = time.time()
    
    # Process requests concurrently using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(semantic_ranker.rank_urls, urls, context)
            for urls, context in test_cases
        ]
        
        # Collect results
        results = []
        for future in as_completed(futures):
            results.append(future.result())
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Verify results
    assert len(results) == num_requests
    for ranked_urls in results:
        assert len(ranked_urls) == num_urls_per_request
        assert all(0 <= score <= 1 for _, score in ranked_urls)
    
    # Log performance metrics
    avg_time_per_request = total_time / num_requests
    logger.info(f"\nSemantic ranker concurrent processing metrics:")
    logger.info(f"  Total time: {total_time:.2f}s")
    logger.info(f"  Average time per request: {avg_time_per_request:.2f}s")
    logger.info(f"  Requests per second: {num_requests/total_time:.2f}") 

@pytest.mark.unit
def test_similarity_calculations(all_rankers):
    """Test similarity calculation methods for all rankers."""
    for name, ranker in all_rankers.items():
        logger.info(f"\nTesting similarity calculations for {name} ranker:")
        
        # Test perfect similarity (same text)
        same_text = "This is a test"
        score = ranker.score_url(f"https://example.com/{same_text}", same_text)
        assert 0.8 <= score <= 1.0, f"{name} ranker failed perfect similarity test"
        logger.info(f"  Perfect similarity score: {score:.4f}")
        
        # Test high similarity
        similar_text1 = "Machine learning is fascinating"
        similar_text2 = "AI and ML are interesting fields"
        score1 = ranker.score_url(f"https://example.com/{similar_text1}", similar_text2)
        logger.info(f"  Similar texts score: {score1:.4f}")
        
        # Test low similarity
        different_text1 = "Machine learning article"
        different_text2 = "Recipe for chocolate cake"
        score2 = ranker.score_url(f"https://example.com/{different_text1}", different_text2)
        logger.info(f"  Different texts score: {score2:.4f}")
        
        # Verify that similar texts have higher score than different texts
        assert score1 > score2, f"{name} ranker failed similarity comparison"
        
        # Test score ranges
        assert 0 <= score1 <= 1, f"{name} ranker score out of range"
        assert 0 <= score2 <= 1, f"{name} ranker score out of range"
