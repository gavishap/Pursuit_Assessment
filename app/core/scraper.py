import aiohttp
import asyncio
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
from urllib.parse import urljoin, urlparse, parse_qs
import logging
from urllib.parse import urljoin, urlparse
import os
import json
from dotenv import load_dotenv
import time
import random
from ..ranking.openai import OpenAIRanker
import re

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LinkScraper:
    """Scrape links from a webpage."""
    
    def __init__(self, semantic_ranker=None, openai_ranker=None, nlp_ranker=None, deep_ranker=None):
        """Initialize the scraper with rankers."""
        self.semantic_ranker = semantic_ranker
        self.openai_ranker = openai_ranker
        self.nlp_ranker = nlp_ranker
        self.deep_ranker = deep_ranker
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9'
        }

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
        """Extract links and basic metadata from HTML content."""
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
                'metadata': {'html': str(link)}  # Store HTML content as metadata
            }
            links.append(link_data)
            
        return links

    async def scrape(self, url: str) -> List[Dict]:
        """Main scraping method."""
        async with aiohttp.ClientSession() as session:
            html = await self.fetch_page(session, url)
            if not html:
                return []
                
            # Extract all links
            links = self.extract_links(html, url)
            
            # Return the links (ranking will be done by the API using request keywords)
            return links

    async def rank_links_batch(self, links: List[Dict], batch_size: int = 10) -> List[Dict]:
        """Rank links in batches to avoid overloading the API."""
        ranked_links = []
        
        # Process links in batches
        for i in range(0, len(links), batch_size):
            batch = links[i:i + batch_size]
            ranked_batch = await self._rank_links(batch)
            ranked_links.extend(ranked_batch)
            
        return ranked_links
    
    async def _rank_links(self, links: List[Dict]) -> List[Dict]:
        """Initialize links with zero scores. Actual ranking is done by the API."""
        try:
            for link in links:
                link['semantic_score'] = 0.0
                link['openai_score'] = 0.0
                link['nlp_score'] = 0.0
                link['deep_learning_score'] = 0.0
                link['ensemble_score'] = 0.0
            return links
        except Exception as e:
            logger.error(f"Error ranking links: {str(e)}")
            return []

    def _is_valid_url(self, url: str) -> bool:
        """Check if URL is valid and meets criteria."""
        try:
            parsed = urlparse(url)
            return all([parsed.scheme, parsed.netloc]) and parsed.scheme in ['http', 'https']
        except Exception:
            return False

    def _guess_content_type(self, url: str) -> str:
        """Guess content type based on URL pattern and structure.
        
        This method analyzes URL patterns to categorize content into different types:
        - document: Various document formats
        - media: Images, videos, audio files
        - info_page: About, contact, team pages
        - blog: Blog posts, articles, news
        - api: API endpoints, data feeds
        - archive: Archives, listings
        - landing: Landing or index pages
        - resource: Resource or download pages
        - interactive: Interactive content, tools
        - search: Search-related pages
        """
        lower_url = url.lower()
        
        # Document types
        if any(ext in lower_url for ext in [
            '.pdf', '.doc', '.docx', '.txt', '.rtf',
            '.xls', '.xlsx', '.csv', '.ppt', '.pptx',
            '.odt', '.ods', '.odp', '.epub', '.mobi'
        ]):
            return 'document'
            
        # Media files
        if any(ext in lower_url for ext in [
            '.jpg', '.jpeg', '.png', '.gif', '.svg',
            '.mp4', '.avi', '.mov', '.wmv', '.flv',
            '.mp3', '.wav', '.ogg', '.webm', '.webp'
        ]):
            return 'media'
            
        # Info pages
        if any(term in lower_url for term in [
            'about', 'contact', 'team', 'staff',
            'careers', 'jobs', 'faq', 'help',
            'support', 'privacy', 'terms', 'policy'
        ]):
            return 'info_page'
            
        # Blog/Article pages
        if any(term in lower_url for term in [
            'blog', 'article', 'post', 'news',
            'story', 'update', 'release', 'press',
            '/20', '/19'  # Common year patterns
        ]) or re.search(r'/\d{4}/\d{2}/', lower_url):  # Date pattern
            return 'blog'
            
        # API and data
        if any(term in lower_url for ext in [
            '/api/', '.json', '.xml', '.rss',
            '/feed/', '/data/', '/stream/', '.graphql'
        ]):
            return 'api'
            
        # Archive/listing pages
        if any(term in lower_url for term in [
            'archive', 'category', 'tag', 'topics',
            'list', 'directory', 'index', 'sitemap'
        ]):
            return 'archive'
            
        # Landing pages
        if lower_url.endswith('/') or lower_url.endswith('index.html') or lower_url.count('/') <= 3:
            return 'landing'
            
        # Resource pages
        if any(term in lower_url for term in [
            'download', 'resource', 'library',
            'docs', 'documentation', 'guide',
            'tutorial', 'example', 'template'
        ]):
            return 'resource'
            
        # Interactive content
        if any(term in lower_url for term in [
            'tool', 'calculator', 'generator',
            'quiz', 'test', 'survey', 'form',
            'dashboard', 'interactive', 'demo'
        ]):
            return 'interactive'
            
        # Search pages
        if any(term in lower_url for term in [
            'search', 'find', 'query', 'results',
            'lookup', 'browse'
        ]):
            return 'search'
            
        # Default for unmatched patterns
        return 'webpage'
