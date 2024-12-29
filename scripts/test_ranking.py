import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from app.ranking.ml import MLRanker
from app.ranking.openai import OpenAIRanker
import logging
from pathlib import Path
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = Path(os.path.dirname(os.path.dirname(__file__))) / 'models' / 'url_ranker.txt'

class RankingTester:
    def __init__(self):
        # Initialize rankers once
        self.ml_ranker = MLRanker(model_path=str(MODEL_PATH))
        self.openai_ranker = OpenAIRanker()

    async def test_url(self, url: str, context: str = None):
        """Test a URL with both ML and OpenAI ranking methods."""
        # Prepare link data
        link_data = {
            'url': url,
            'keywords': context.split() if context else [],
            'title': '',  # Will be fetched by rankers
            'description': '',  # Will be fetched by rankers
        }
        
        # Calculate scores
        ml_score = self.ml_ranker.calculate_score(link_data)
        openai_score = self.openai_ranker.calculate_score(link_data)
        
        # Print results
        print("\nRanking Results:")
        print("-" * 50)
        print(f"URL: {url}")
        print(f"Context: {context if context else 'None'}")
        print(f"ML Model Score: {ml_score:.4f}")
        print(f"OpenAI Score: {openai_score:.4f}")
        print("-" * 50)

async def main():
    # Test cases with various URLs and contexts
    test_urls = [
        # Financial URLs
        ("https://www.treasury.gov/resource-center/data-chart-center/interest-rates/pages/textview.aspx?data=yield", 
         "treasury yield rates financial data"),
        ("https://www.sec.gov/edgar/searchedgar/companysearch", 
         "company financial reports SEC filings"),
        ("https://www.federalreserve.gov/releases/h15/",
         "federal reserve interest rates monetary policy"),
         
        # News URLs
        ("https://www.reuters.com/markets/",
         "financial markets news trading"),
        ("https://www.bloomberg.com/markets",
         "stock market financial news"),
         
        # Reference URLs
        ("https://www.investopedia.com/terms/y/yieldcurve.asp",
         "yield curve explanation financial terms"),
        ("https://www.wikipedia.org/wiki/Federal_funds_rate",
         "federal funds rate definition"),
         
        # General URLs
        ("https://www.wikipedia.org",
         "general information encyclopedia"),
        ("https://www.google.com",
         "search engine general"),
    ]
    
    print("\nTesting URLs with both ranking models...")
    print(f"ML Model: {MODEL_PATH}")
    print(f"OpenAI Model: GPT-4")
    
    # Create single instance of tester
    tester = RankingTester()
    
    for url, context in test_urls:
        await tester.test_url(url, context)

if __name__ == "__main__":
    asyncio.run(main()) 
