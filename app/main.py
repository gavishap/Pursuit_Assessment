"""
High-Value Link Scraper API: Main Application Flow
================================================

Here's the complete flow of operations from startup to request handling:

1. Startup Sequence
------------------
a) Logging Initialization
   - Configure logging with INFO level
   - Set up logger for application-wide use

b) Database Initialization
   - Create all database tables if they don't exist
   - Set up SQLite with optimized settings (WAL mode, connection pooling)

c) Ranker Initialization
   - Load and initialize all ranking models:
     * Semantic Ranker: Uses sentence transformers for text similarity
     * OpenAI Ranker: Leverages GPT for context understanding
     * NLP Ranker: Employs spaCy for linguistic analysis
     * Deep Learning Ranker: Custom neural network with model versioning

2. Request Handling Flow
-----------------------
a) Scraping Endpoint (/scrape)
   1. Receive scrape request with URL and parameters
   2. Initialize scraper with all rankers
   3. Extract links from target URL
   4. For each link:
      - Get scores from all rankers
      - Calculate ensemble score
      - Apply relevance threshold
      - Store in database if qualified
   5. Return ranked results

b) Link Retrieval (/links)
   1. Accept filtering parameters
   2. Build dynamic SQL query
   3. Apply filters:
      - Base URL matching
      - Minimum relevance threshold
      - Content type filtering
      - Keyword matching
   4. Apply sorting (by score type or date)
   5. Return paginated results

3. Data Processing Pipeline
--------------------------
a) Link Processing
   1. URL normalization and cleaning
   2. Content type detection
   3. Metadata extraction
   4. Keyword association

b) Ranking Process
   1. Context preparation
   2. Multi-model scoring
   3. Ensemble score calculation
   4. Threshold application

4. Database Operations
---------------------
a) Write Operations
   - New link creation
   - Existing link updates
   - Batch processing for efficiency

b) Read Operations
   - Filtered queries
   - Sorted retrievals
   - Pagination handling

5. Error Handling
----------------
- Graceful error recovery
- Detailed error logging
- Client-friendly error responses
- Transaction management

6. Testing Strategy
------------------
a) Unit Tests
   - Individual ranker testing:
     * Semantic similarity accuracy
     * OpenAI API integration
     * NLP processing correctness
     * Deep learning model predictions
   - URL preprocessing validation
   - Database operations verification
   - Content type detection accuracy

b) Integration Tests
   - Complete scraping pipeline
   - Multi-ranker ensemble scoring
   - Database transaction handling
   - API endpoint behavior
   - Error handling scenarios

c) Model Testing
   1. Deep Learning Ranker:
      - Model loading and versioning
      - Inference performance
      - Fallback mechanism
      - Training process validation
      - Prediction consistency

   2. Semantic Ranker:
      - Embedding quality
      - Similarity calculations
      - Edge case handling

   3. NLP Ranker:
      - Entity recognition
      - Topic extraction
      - Language processing
      - Token handling

d) Performance Testing
   - Concurrent request handling
   - Database query optimization
   - Memory usage monitoring
   - Response time benchmarking
   - Batch processing efficiency

e) Test Data Management
   - Synthetic test cases
   - Real-world URL samples
   - Edge case scenarios
   - Invalid input handling
   - Cross-domain testing

f) Continuous Testing
   - Automated test runs
   - Regression testing
   - Model performance monitoring
   - API endpoint validation
   - Error rate tracking

All tests are implemented in scripts/test_rankers.py, which serves as the main test suite
for the entire application. This comprehensive test suite covers all rankers (Semantic, OpenAI,
NLP, and Deep Learning), performance metrics, and system behavior. The tests include memory
profiling (using psutil), concurrent processing validation, and extensive performance
benchmarking for all components.

The application uses FastAPI's dependency injection system for database sessions
and implements comprehensive logging throughout the pipeline for monitoring and
debugging purposes.
"""

from fastapi import FastAPI, HTTPException, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc, String
from typing import List, Optional
from pathlib import Path
from datetime import datetime
from urllib.parse import urlparse, parse_qs
import os

from .api import models
from .core import scraper
from .models.base import SessionLocal, engine, Base
from .models.link import Link
from .ranking import SemanticRanker, OpenAIRanker, AdvancedNLPRanker, TrainedDeepRanker
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create database tables on startup if they don't exist
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="High-Value Link Scraper API",
    description="API for scraping and retrieving high-value links with intelligent ranking",
    version="1.0.0"
)

# Initialize rankers
semantic_ranker = SemanticRanker()
openai_ranker = OpenAIRanker(api_key=os.getenv('OPENAI_API_KEY'))
nlp_ranker = AdvancedNLPRanker()
deep_ranker = TrainedDeepRanker()

# Dependency to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/health")
def health_check():
    """Check if the API is running."""
    return {"status": "healthy", "version": "1.0.0"}

@app.post("/scrape", response_model=List[models.LinkResponse])
async def scrape_url(
    request: models.ScrapeRequest,
    db: Session = Depends(get_db)
):
    """
    Scrape links from a URL and store them in the database.
    
    - Scrapes all links from the given URL
    - Ranks them using multiple ranking methods
    - Stores results in database
    - Returns links above the minimum relevance threshold
    """
    try:
        # Initialize scraper with all rankers
        link_scraper = scraper.LinkScraper(
            semantic_ranker=semantic_ranker,
            openai_ranker=openai_ranker,
            nlp_ranker=nlp_ranker,
            deep_ranker=deep_ranker
        )
        
        # Scrape links
        links = await link_scraper.scrape(str(request.url))
        logger.info(f"Found {len(links)} links from {request.url}")
        
        # Process and store links
        stored_links = []
        
        for link_data in links:
            try:
                url = str(link_data['url'])
                # Use keywords from request as context
                context = ' '.join(request.keywords) if request.keywords else ''
                
                # Get scores from all rankers
                semantic_score = semantic_ranker.score_url(url, context)
                openai_score = await openai_ranker.score_url(url, context)
                nlp_score = nlp_ranker.score_url(url, context)
                deep_score = deep_ranker.score_url(url, context)
                
                # Calculate ensemble score
                ensemble_score = (semantic_score + openai_score + nlp_score + deep_score) / 4
                
                # Skip if below threshold
                if ensemble_score < request.min_relevance:
                    continue
                
                # Create or update link in database
                db_link = db.query(Link).filter(
                    Link.base_url == str(request.url),
                    Link.url == url
                ).first()
                
                if db_link:
                    # Update existing link
                    db_link.semantic_score = semantic_score
                    db_link.openai_score = openai_score
                    db_link.nlp_score = nlp_score
                    db_link.deep_learning_score = deep_score
                    db_link.ensemble_score = ensemble_score
                    db_link.title = link_data.get('title', '')
                    db_link.content_type = link_data.get('content_type', '')
                    db_link.keywords = request.keywords
                    db_link.link_metadata = link_data.get('metadata', {})
                    db_link.updated_at = datetime.utcnow()
                else:
                    # Create new link
                    db_link = Link(
                        base_url=str(request.url),
                        url=url,
                        title=link_data.get('title', ''),
                        content_type=link_data.get('content_type', ''),
                        semantic_score=semantic_score,
                        openai_score=openai_score,
                        nlp_score=nlp_score,
                        deep_learning_score=deep_score,
                        ensemble_score=ensemble_score,
                        keywords=request.keywords,
                        link_metadata=link_data.get('metadata', {})
                    )
                    db.add(db_link)
                
                stored_links.append(db_link)
                
                if len(stored_links) >= request.max_links:
                    break
                    
            except Exception as e:
                logger.error(f"Error processing link {link_data.get('url', 'unknown')}: {str(e)}")
                continue
        
        db.commit()
        
        # Sort by ensemble score and return
        stored_links.sort(key=lambda x: x.ensemble_score, reverse=True)
        return stored_links[:request.max_links]
        
    except Exception as e:
        logger.error(f"Error scraping URL: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/links", response_model=List[models.LinkResponse])
async def get_links(
    base_url: Optional[str] = None,
    min_relevance: float = Query(0.5, ge=0.0, le=1.0),
    content_type: Optional[str] = None,
    keywords: Optional[List[str]] = Query(None),
    sort_by: str = Query("ensemble", enum=["ensemble", "semantic", "openai", "nlp", "deep_learning", "date"]),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db)
):
    """
    Retrieve stored links with filtering and sorting options.
    
    - base_url: Filter by source URL
    - min_relevance: Minimum ensemble score threshold
    - content_type: Filter by content type
    - keywords: Filter by keywords
    - sort_by: Sort by score type or date
    """
    # First, get all links to see what's in the database
    all_links = db.query(Link).all()
    logger.info(f"Total links in database: {len(all_links)}")
    if all_links:
        sample_link = all_links[0]
        logger.info(f"Sample link data: base_url={sample_link.base_url}, keywords={sample_link.keywords}, ensemble_score={sample_link.ensemble_score}")
    
    # Now apply filters
    query = db.query(Link).filter(Link.ensemble_score >= min_relevance)
    
    if base_url:
        query = query.filter(Link.base_url == base_url)
        filtered_by_url = query.all()
        logger.info(f"Links after base_url filter: {len(filtered_by_url)}")
    
    if content_type:
        query = query.filter(Link.content_type == content_type)
        
    if keywords:
        # Split the comma-separated keywords and clean them
        if isinstance(keywords, str):
            keywords = [k.strip() for k in keywords.split(',')]
        elif isinstance(keywords, list) and len(keywords) == 1:
            # Handle case where a comma-separated string is passed as a list item
            keywords = [k.strip() for k in keywords[0].split(',')]
            
        logger.info(f"Filtering by keywords: {keywords}")
        
        # Filter links that have any of the specified keywords
        from sqlalchemy import or_
        keyword_filters = []
        for keyword in keywords:
            keyword_filters.append(Link.keywords.cast(String).like(f'%{keyword}%'))
        query = query.filter(or_(*keyword_filters))
        filtered_by_keywords = query.all()
        logger.info(f"Links after keyword filter: {len(filtered_by_keywords)}")
    
    # Apply sorting
    if sort_by == "ensemble":
        query = query.order_by(desc(Link.ensemble_score))
    elif sort_by == "semantic":
        query = query.order_by(desc(Link.semantic_score))
    elif sort_by == "openai":
        query = query.order_by(desc(Link.openai_score))
    elif sort_by == "nlp":
        query = query.order_by(desc(Link.nlp_score))
    elif sort_by == "deep_learning":
        query = query.order_by(desc(Link.deep_learning_score))
    else:  # sort by date
        query = query.order_by(desc(Link.created_at))
    
    links = query.offset(skip).limit(limit).all()
    logger.info(f"Final number of links returned: {len(links)}")
    return links

@app.delete("/links/{link_id}")
async def delete_link(
    link_id: int,
    db: Session = Depends(get_db)
):
    """Delete a link by ID."""
    link = db.query(Link).filter(Link.id == link_id).first()
    if not link:
        raise HTTPException(status_code=404, detail="Link not found")
    
    db.delete(link)
    db.commit()
    return {"status": "success", "message": f"Link {link_id} deleted"}
