import aiohttp
import asyncio
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
import logging
from urllib.parse import urljoin, urlparse
import os
import json
from dotenv import load_dotenv
import time
import random
from ..ranking.openai import OpenAIRanker

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LinkScraper:
    def __init__(self):
        self.keywords = self._load_keywords()
        self.openai_ranker = OpenAIRanker()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'sec-ch-ua': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Windows"',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'cross-site',
            'Sec-Fetch-User': '?1',
            'Pragma': 'no-cache',
            'Cache-Control': 'no-cache',
            'TE': 'trailers',
            'DNT': '1'
        }
        self.ranking_method = "openai"  # Default to OpenAI
        self.ml_ranker = None  # Will be set by API if ML ranking is chosen
        
    def _load_keywords(self) -> List[str]:
        """Load keywords from environment or use defaults."""
        default_keywords = ["ACFR", "Budget", "Finance", "Financial", "Report", "Treasury", "Tax", "Revenue", "Audit", "Fiscal", "Statement"]
        keywords_str = os.getenv("SCRAPER_KEYWORDS")
        if keywords_str:
            try:
                return json.loads(keywords_str)
            except json.JSONDecodeError:
                return default_keywords
        return default_keywords

    async def fetch_page(self, session: aiohttp.ClientSession, url: str) -> Optional[str]:
        """Fetch a page with error handling."""
        try:
            # Add a random delay between 2-4 seconds
            await asyncio.sleep(random.uniform(2, 4))
            
            timeout = aiohttp.ClientTimeout(total=30)
            
            # Update headers with correct host and origin for the specific URL
            parsed_url = urlparse(url)
            headers = self.headers.copy()
            headers.update({
                'Host': parsed_url.netloc,
                'Origin': f"{parsed_url.scheme}://{parsed_url.netloc}",
                'Referer': f"{parsed_url.scheme}://{parsed_url.netloc}/"
            })
            
            async with session.get(
                url, 
                headers=headers, 
                ssl=False, 
                timeout=timeout,
                allow_redirects=True
            ) as response:
                if response.status == 200:
                    return await response.text()
                elif response.status == 403:
                    logger.warning(f"Access forbidden for {url}, waiting 5 seconds and trying again...")
                    await asyncio.sleep(5)
                    # Try one more time
                    async with session.get(
                        url, 
                        headers=headers, 
                        ssl=False, 
                        timeout=timeout,
                        allow_redirects=True
                    ) as retry_response:
                        if retry_response.status == 200:
                            return await retry_response.text()
                logger.warning(f"Failed to fetch {url}: Status {response.status}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching {url}: {str(e)}")
            return None

    def extract_links(self, html: str, base_url: str) -> List[Dict]:
        """Extract and process links from HTML content."""
        soup = BeautifulSoup(html, 'html.parser')
        links = []
        
        for link in soup.find_all('a', href=True):
            url = urljoin(base_url, link['href'])
            if not self._is_valid_url(url):
                continue
                
            # Get text from the link and its children
            text = ' '.join(link.stripped_strings)
            if not text:
                text = link.get('title', '') or link.get('aria-label', '') or ''
                
            link_data = {
                'url': url,
                'title': text.strip(),
                'content_type': self._guess_content_type(url),
                'keywords': self._extract_keywords(link),
                'content': str(link)  # Store HTML content for ML features
            }
            links.append(link_data)
            
        return links

    async def rank_links_batch(self, links: List[Dict], batch_size: int = 10) -> List[Dict]:
        """Rank links using either OpenAI API or ML model."""
        if self.ranking_method == "ml" and self.ml_ranker is not None:
            # Use ML ranker
            scores = self.ml_ranker.batch_calculate_scores(links)
            for link, score in zip(links, scores):
                link['relevance_score'] = score
        else:
            # Use OpenAI ranker
            for link in links:
                # Add keywords to link_data for ranking
                link_data = {
                    'url': link['url'],
                    'title': link['title'],
                    'keywords': link['keywords'],
                    'description': '',  # Could be added if we extract meta descriptions
                }
                link['relevance_score'] = self.openai_ranker.calculate_score(link_data)
                # Add small delay between requests
                await asyncio.sleep(0.1)
            
        # Sort all links by relevance score
        links.sort(key=lambda x: x['relevance_score'], reverse=True)
        return links

    def _is_valid_url(self, url: str) -> bool:
        """Check if URL is valid and meets criteria."""
        try:
            parsed = urlparse(url)
            return all([parsed.scheme, parsed.netloc]) and parsed.scheme in ['http', 'https']
        except Exception:
            return False

    def _guess_content_type(self, url: str) -> str:
        """Guess content type based on URL pattern."""
        lower_url = url.lower()
        if any(ext in lower_url for ext in ['.pdf', '.doc', '.docx', '.xls', '.xlsx']):
            return 'document'
        if any(term in lower_url for term in ['contact', 'about', 'staff']):
            return 'info_page'
        if any(term in lower_url for term in ['finance', 'budget', 'report', 'acfr']):
            return 'financial'
        return 'webpage'

    def _extract_keywords(self, link_element) -> List[str]:
        """Extract keywords from link text and attributes."""
        keywords = []
        text = ' '.join([
            link_element.get_text(strip=True),
            link_element.get('title', ''),
            link_element.get('aria-label', ''),
            link_element.get('href', '')
        ]).lower()
        
        # Check for matches with predefined keywords
        for keyword in self.keywords:
            if keyword.lower() in text:
                keywords.append(keyword)
                
        return list(set(keywords))

    async def scrape(self, url: str) -> List[Dict]:
        """Main scraping method."""
        async with aiohttp.ClientSession() as session:
            html = await self.fetch_page(session, url)
            if not html:
                return []
                
            # First extract all links
            links = self.extract_links(html, url)
            
            # Then rank them using the selected method
            ranked_links = await self.rank_links_batch(links, batch_size=10)
            
            return ranked_links
