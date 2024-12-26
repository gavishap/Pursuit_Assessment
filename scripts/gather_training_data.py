import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import json
from typing import List, Dict
from app.core.scraper import LinkScraper
import logging
import aiohttp
from datetime import datetime
import re
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Training data sources
TRAINING_SOURCES = {
    # Government Financial Portals
    "government": [
        "https://www.bozeman.net/departments/finance",
        "https://transparency.mt.gov/",
        "https://fiscaldata.treasury.gov/datasets/",
        "https://www.gfoa.org/resources",
    ],
    
    # State Auditor Offices
    "auditors": [
        "https://www.sao.wa.gov/reports-data/audit-reports/",
        "https://auditor.mo.gov/reports/",
        "https://www.auditor.state.mn.us/reports",
    ],
    
    # Municipal Finance Sites
    "municipal": [
        "https://emma.msrb.org/",
        "https://www.nasact.org/af_member_resources",
    ],
    
    # Financial Standards Organizations
    "standards": [
        "https://www.gasb.org/resources",
        "https://www.gfoa.org/best-practices",
    ]
}

# Automatic relevance scoring rules
RELEVANCE_RULES = {
    "high_value_terms": {
        "acfr": 1.0,
        "cafr": 1.0,
        "budget": 0.9,
        "financial report": 0.9,
        "audit report": 0.9,
        "fiscal year": 0.8,
        "financial statement": 0.8,
    },
    "file_types": {
        ".pdf": 0.7,
        ".xlsx": 0.6,
        ".xls": 0.6,
        ".csv": 0.5,
    },
    "url_patterns": {
        r"\d{4}(-\d{4})?": 0.3,  # Year or year range
        r"(q[1-4]|quarter)": 0.3,  # Quarterly reports
        r"(fy|fiscal)": 0.3,      # Fiscal year references
    }
}

def auto_score_relevance(link_data: Dict) -> float:
    """Automatically score link relevance based on rules."""
    score = 0.0
    url = link_data.get('url', '').lower()
    title = link_data.get('title', '').lower()
    text = f"{url} {title}"
    
    # Check high-value terms
    for term, value in RELEVANCE_RULES['high_value_terms'].items():
        if term in text:
            score += value
    
    # Check file types
    for ext, value in RELEVANCE_RULES['file_types'].items():
        if url.endswith(ext):
            score += value
            break
    
    # Check URL patterns
    for pattern, value in RELEVANCE_RULES['url_patterns'].items():
        if re.search(pattern, text):
            score += value
    
    # Normalize score
    return min(1.0, score)

async def gather_training_data():
    """Gather links from various sources for training data."""
    scraper = LinkScraper()
    all_links = []
    
    for category, urls in TRAINING_SOURCES.items():
        for url in urls:
            try:
                logger.info(f"Scraping {url}")
                links = await scraper.scrape(url)
                
                # Add metadata and auto-score
                for link in links:
                    link['category'] = category
                    link['auto_relevance_score'] = auto_score_relevance(link)
                    link['scraped_date'] = datetime.utcnow().isoformat()
                
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
    # Add fields for manual labeling
    for link in links:
        link['manual_relevance_score'] = None  # To be filled manually
        link['notes'] = None  # For annotator notes
        
    # Group links by category
    links_by_category = {}
    for link in links:
        category = link.get('category', 'uncategorized')
        if category not in links_by_category:
            links_by_category[category] = []
        links_by_category[category].append(link)
    
    # Save categorized links
    with open(output_file, 'w') as f:
        json.dump(links_by_category, f, indent=2)
        
    logger.info(f"Saved {len(links)} links to {output_file}")
    
    # Save summary
    summary = {
        'total_links': len(links),
        'categories': {cat: len(links) for cat, links in links_by_category.items()},
        'auto_scored': sum(1 for link in links if 'auto_relevance_score' in link),
        'date': datetime.utcnow().isoformat()
    }
    
    with open('training_data_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

def load_labeled_data(input_file: str = "training_data_labeled.json") -> List[Dict]:
    """Load manually labeled training data."""
    if not os.path.exists(input_file):
        logger.error(f"Labeled data file {input_file} not found")
        return []
        
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # If data is categorized, flatten it
    if isinstance(data, dict):
        links = []
        for category_links in data.values():
            links.extend(category_links)
    else:
        links = data
        
    # Filter out links without manual labels
    labeled_links = [
        link for link in links
        if link.get('manual_relevance_score') is not None
    ]
    
    # Add correlation analysis between auto and manual scores
    if labeled_links:
        auto_scores = [link.get('auto_relevance_score', 0) for link in labeled_links]
        manual_scores = [link['manual_relevance_score'] for link in labeled_links]
        correlation = np.corrcoef(auto_scores, manual_scores)[0, 1]
        logger.info(f"Correlation between auto and manual scores: {correlation:.3f}")
    
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
    2. For each link:
       - Review the auto_relevance_score
       - Add your manual_relevance_score (0.0 to 1.0)
       - Add any notes about the link
    3. Save as training_data_labeled.json
    4. Run train_model.py to train the ML model
    
    Scoring guidelines:
    1.0: Direct financial reports (ACFR, Budget)
    0.8-0.9: Important financial documents
    0.6-0.7: Related financial information
    0.4-0.5: Indirectly related content
    0.2-0.3: Marginally related content
    0.0-0.1: Unrelated content
    """)

if __name__ == "__main__":
    asyncio.run(main()) 
