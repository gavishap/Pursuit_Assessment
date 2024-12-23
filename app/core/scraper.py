import aiohttp
import asyncio
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
import logging
from urllib.parse import urljoin, urlparse
import os
import json
import openai
from dotenv import load_dotenv
import time
import random

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LinkScraper:
    def __init__(self):
        self.keywords = self._load_keywords()
        self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
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
            'DNT': '1',
            'Host': 'www.bozeman.net',
            'Origin': 'https://www.bozeman.net',
            'Referer': 'https://www.bozeman.net/'
        }
        
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
            self.headers['Host'] = parsed_url.netloc
            self.headers['Origin'] = f"{parsed_url.scheme}://{parsed_url.netloc}"
            self.headers['Referer'] = f"{parsed_url.scheme}://{parsed_url.netloc}/"
            
            async with session.get(
                url, 
                headers=self.headers, 
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
                        headers=self.headers, 
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
            }
            links.append(link_data)
            
        return links

    async def rank_links_batch(self, links: List[Dict], batch_size: int = 10) -> List[Dict]:
        """Rank links in batches using OpenAI API."""
        ranked_links = []
        
        # Process links in batches
        for i in range(0, len(links), batch_size):
            batch = links[i:i + batch_size]
            
            # Create a single prompt for the entire batch
            batch_prompt = "Analyze the following links and rate their relevance to financial and budget content. For each link, provide a score between 0.0 and 1.0.\n\n"
            for idx, link in enumerate(batch, 1):
                batch_prompt += f"Link {idx}:\nURL: {link['url']}\nTitle: {link['title']}\nKeywords: {', '.join(link['keywords'])}\n\n"
            
            batch_prompt += "\nRespond with a JSON object where keys are link numbers and values are scores. Example: {'1': 0.8, '2': 0.4, ...}"
            
            try:
                # Make a single API call for the batch
                response = await asyncio.to_thread(
                    self.openai_client.chat.completions.create,
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a link relevance scorer. Respond only with a JSON object containing scores."},
                        {"role": "user", "content": batch_prompt}
                    ]
                )
                
                # Parse the response
                try:
                    scores = json.loads(response.choices[0].message.content.strip())
                    
                    # Apply scores to links
                    for idx, link in enumerate(batch, 1):
                        link['relevance_score'] = float(scores.get(str(idx), 0.0))
                        ranked_links.append(link)
                        
                except (json.JSONDecodeError, ValueError) as e:
                    logger.error(f"Error parsing OpenAI response: {str(e)}")
                    # Assign default scores if parsing fails
                    for link in batch:
                        link['relevance_score'] = 0.0
                        ranked_links.append(link)
                
                # Add delay between batches
                if i + batch_size < len(links):
                    await asyncio.sleep(2)
                    
            except Exception as e:
                logger.error(f"Error in batch processing: {str(e)}")
                # Add links with default score on error
                for link in batch:
                    link['relevance_score'] = 0.0
                    ranked_links.append(link)
        
        # Sort all links by relevance score
        ranked_links.sort(key=lambda x: x['relevance_score'], reverse=True)
        return ranked_links

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
            
            # Then rank them in batches
            ranked_links = await self.rank_links_batch(links, batch_size=10)
            
            return ranked_links
