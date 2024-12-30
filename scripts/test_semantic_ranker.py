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

async def test_rankers():
    """Test all rankers with URLs from both database and predefined test cases."""
    semantic_ranker = SemanticRanker()
    openai_ranker = OpenAIRanker(api_key=os.getenv('OPENAI_API_KEY'))
    nlp_ranker = AdvancedNLPRanker()
    
    # Initialize deep ranker (it will handle training if needed)
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
        print("\n" + "=" * 120)
        print(f"\nTEST CASE: {test['context']}")
        print("=" * 120)
        
        # Get rankings from all rankers
        semantic_ranked = semantic_ranker.rank_urls(test['urls'], test['context'])
        openai_ranked = await openai_ranker.rank_urls(test['urls'], test['context'])
        nlp_ranked = nlp_ranker.rank_urls(test['urls'], test['context'])
        deep_ranked = deep_ranker.rank_urls(test['urls'], test['context'])
        
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
        
        # Print results in a table format
        print("\n{:<40} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
            "URL", "Semantic", "OpenAI", "NLP", "Deep", "Ensemble"
        ))
        print("-" * 100)
        
        for result in combined_results:
            url_display = result['url'][:37] + "..." if len(result['url']) > 40 else result['url']
            print("{:<40} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f}".format(
                url_display,
                result['semantic_score'],
                result['openai_score'],
                result['nlp_score'],
                result['deep_score'],
                result['ensemble_score']
            ))
        
        # Print summary statistics
        print("\nSUMMARY STATISTICS:")
        print("-" * 30)
        
        # Calculate means
        semantic_mean = np.mean([r['semantic_score'] for r in combined_results])
        openai_mean = np.mean([r['openai_score'] for r in combined_results])
        nlp_mean = np.mean([r['nlp_score'] for r in combined_results])
        deep_mean = np.mean([r['deep_score'] for r in combined_results])
        ensemble_mean = np.mean([r['ensemble_score'] for r in combined_results])
        
        print(f"Average Scores:")
        print(f"  Semantic:  {semantic_mean:.4f}")
        print(f"  OpenAI:    {openai_mean:.4f}")
        print(f"  NLP:       {nlp_mean:.4f}")
        print(f"  Deep:      {deep_mean:.4f}")
        print(f"  Ensemble:  {ensemble_mean:.4f}")
        
        # Calculate correlations
        print("\nScore Correlations:")
        semantic_scores = [r['semantic_score'] for r in combined_results]
        openai_scores = [r['openai_score'] for r in combined_results]
        nlp_scores = [r['nlp_score'] for r in combined_results]
        deep_scores = [r['deep_score'] for r in combined_results]
        
        print(f"  Semantic-OpenAI:  {calculate_correlation(semantic_scores, openai_scores):.4f}")
        print(f"  Semantic-NLP:     {calculate_correlation(semantic_scores, nlp_scores):.4f}")
        print(f"  Semantic-Deep:    {calculate_correlation(semantic_scores, deep_scores):.4f}")
        print(f"  OpenAI-NLP:       {calculate_correlation(openai_scores, nlp_scores):.4f}")
        print(f"  OpenAI-Deep:      {calculate_correlation(openai_scores, deep_scores):.4f}")
        print(f"  NLP-Deep:         {calculate_correlation(nlp_scores, deep_scores):.4f}")

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

async def main():
    await test_rankers()

if __name__ == "__main__":
    asyncio.run(main()) 
