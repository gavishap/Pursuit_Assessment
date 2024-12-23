import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import json
from typing import List, Dict
from app.core.scraper import LinkScraper
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# List of financial websites to scrape for training data
TRAINING_URLS = [
    "https://www.bozeman.net/departments/finance",
    "https://transparency.mt.gov/",
    # Add more URLs here
]

async def gather_training_data():
    """Gather links from various sources for training data."""
    scraper = LinkScraper()
    all_links = []
    
    for url in TRAINING_URLS:
        try:
            logger.info(f"Scraping {url}")
            links = await scraper.scrape(url)
            all_links.extend(links)
            logger.info(f"Found {len(links)} links from {url}")
            
            # Add delay between requests
            await asyncio.sleep(2)
            
        except Exception as e:
            logger.error(f"Error scraping {url}: {str(e)}")
            continue
    
    return all_links

def save_links_for_labeling(links: List[Dict], output_file: str = "training_data_unlabeled.json"):
    """Save links to a JSON file for manual labeling."""
    # Add a label field to each link
    for link in links:
        link['manual_relevance_score'] = None  # To be filled manually
        
    with open(output_file, 'w') as f:
        json.dump(links, f, indent=2)
        
    logger.info(f"Saved {len(links)} links to {output_file}")

def load_labeled_data(input_file: str = "training_data_labeled.json") -> List[Dict]:
    """Load manually labeled training data."""
    if not os.path.exists(input_file):
        logger.error(f"Labeled data file {input_file} not found")
        return []
        
    with open(input_file, 'r') as f:
        links = json.load(f)
        
    # Filter out links without labels
    labeled_links = [
        link for link in links
        if link.get('manual_relevance_score') is not None
    ]
    
    logger.info(f"Loaded {len(labeled_links)} labeled links")
    return labeled_links

async def main():
    """Main function to gather and process training data."""
    # Gather links
    links = await gather_training_data()
    
    if not links:
        logger.error("No links found")
        return
        
    # Save for manual labeling
    save_links_for_labeling(links)
    
    logger.info("""
    Next steps:
    1. Open training_data_unlabeled.json
    2. For each link, add a manual_relevance_score between 0.0 and 1.0
    3. Save the labeled data as training_data_labeled.json
    4. Run train_model.py to train the ML model
    """)

if __name__ == "__main__":
    asyncio.run(main()) 
