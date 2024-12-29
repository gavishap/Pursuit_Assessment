import os
from typing import List, Tuple
import logging
from openai import AsyncOpenAI
import json

logger = logging.getLogger(__name__)

class OpenAIRanker:
    """A ranker that uses OpenAI to score URLs based on context."""
    
    def __init__(self, api_key: str = None):
        """Initialize the OpenAI client with API key."""
        if api_key is None:
            api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key must be provided or set in OPENAI_API_KEY environment variable")
        self.client = AsyncOpenAI(api_key=api_key)
        
    async def score_url(self, url: str, context: str) -> float:
        """Score a URL's relevance to the given context using OpenAI."""
        prompt = f"""You are a URL relevance scoring system. Analyze and score how relevant this URL is to the given context.

Context: {context}
URL: {url}

Score based on these weighted factors:

1. Content Match (40%):
   - How specific is the URL path to the topic?
   - Does it contain relevant keywords?
   - Is the content likely current/timely?

2. Source Authority (30%):
   - Is this an official/primary source?
   - Is the domain trustworthy for this topic?
   - Is this a recognized expert source?

3. URL Structure (20%):
   - Is this a direct link to relevant content?
   - Is the URL path logically organized?
   - Does the structure suggest quality content?

4. General Quality (10%):
   - Is this likely to be accessible?
   - Is this likely to be high-quality content?
   - Is this a stable, permanent URL?

Detailed Scoring Guide:
0.0-0.2: No relevance or extremely low quality
0.2-0.4: Minimal relevance, indirect connection
0.4-0.6: Moderate relevance, some connection
0.6-0.8: High relevance, strong connection
0.8-0.9: Very high relevance, excellent match
0.9-1.0: Perfect relevance, authoritative source

Respond with ONLY a single decimal number between 0.0 and 1.0 representing the final weighted score."""

        try:
            response = await self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a precise URL relevance scoring system. You analyze URLs carefully and provide nuanced scores based on multiple factors. You MUST respond with only a single decimal number between 0.0 and 1.0."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2  # Lower temperature for more consistent scoring
            )
            
            # Extract score from response and handle potential formatting issues
            response_text = response.choices[0].message.content.strip()
            
            # Try to extract a float from the response
            try:
                # First try direct conversion
                score = float(response_text)
            except ValueError:
                # If that fails, try to find any float in the text
                import re
                numbers = re.findall(r"[0-9]*\.?[0-9]+", response_text)
                if numbers:
                    # Use the last number found (often the final score)
                    score = float(numbers[-1])
                else:
                    # If no numbers found, use a default score
                    logger.error(f"Could not extract score from OpenAI response: {response_text}")
                    score = 0.5
            
            # Ensure score is between 0 and 1
            score = max(0.0, min(1.0, score))
            
            logger.debug(f"URL: {url}")
            logger.debug(f"Context: {context}")
            logger.debug(f"Score: {score:.4f}")
            
            return score
            
        except Exception as e:
            logger.error(f"Error scoring URL with OpenAI: {str(e)}")
            # Return a neutral score instead of 0 on error
            return 0.5
    
    async def rank_urls(self, urls: List[str], context: str) -> List[Tuple[str, float]]:
        """Rank a list of URLs based on their relevance to the context."""
        # Score all URLs
        scores = []
        for url in urls:
            score = await self.score_url(url, context)
            scores.append((url, score))
        
        # Sort by score descending
        ranked = sorted(scores, key=lambda x: x[1], reverse=True)
        
        return ranked 
