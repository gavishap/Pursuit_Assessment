from fastapi import FastAPI, HTTPException, Depends, Query
from sqlalchemy.orm import Session
from typing import List, Optional, Dict
from . import models
from .models.base import get_db, engine
from .models.link import Link as DBLink
from .api.models import Link, LinkCreate, ScrapeRequest, HealthCheck
from .core.scraper import LinkScraper
from .ranking.ml import MLRanker
import logging
from datetime import datetime
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy import update, desc
from sqlalchemy.exc import IntegrityError
import os

# Create database tables
models.base.Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="High-Value Link Scraper API",
    description="API for scraping and retrieving high-value links with intelligent ranking",
    version="1.0.0"
)

logger = logging.getLogger(__name__)
scraper = LinkScraper()

# Initialize ML ranker if model exists
model_path = os.path.join(os.path.dirname(__file__), '..', 'models')
ml_ranker = MLRanker(model_path) if os.path.exists(model_path) else None

@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Check API health status."""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "ml_model_loaded": ml_ranker is not None
    }

@app.post("/scrape", response_model=List[Link])
async def scrape_url(
    request: ScrapeRequest,
    ranking_method: str = Query("openai", enum=["openai", "ml"]),
    keywords: Optional[List[str]] = Query(None),
    db: Session = Depends(get_db)
):
    """
    Scrape links from a URL and store them in the database.
    
    - ranking_method: Choose between 'openai' or 'ml' ranking
    - keywords: Optional list of keywords to prioritize
    """
    try:
        # Use ML ranker if requested and available
        if ranking_method == "ml" and ml_ranker is not None:
            scraper.ranking_method = "ml"
            scraper.ml_ranker = ml_ranker
        else:
            scraper.ranking_method = "openai"
            scraper.ml_ranker = None
            
        # Set custom keywords if provided
        if keywords:
            scraper.keywords = keywords
            
        links = await scraper.scrape(str(request.url))
        
        # Limit the number of links if specified
        links = links[:request.max_links]
        
        # Store links in database
        db_links = []
        seen_urls = set()
        
        for link_data in links:
            try:
                url = str(link_data['url'])
                
                if url in seen_urls:
                    continue
                seen_urls.add(url)
                
                # Try to find existing link
                existing_link = db.query(DBLink).filter(DBLink.url == url).first()
                
                if existing_link:
                    # Update existing link if the new score is higher
                    if link_data['relevance_score'] > existing_link.relevance_score:
                        existing_link.title = link_data['title']
                        existing_link.content_type = link_data['content_type']
                        existing_link.relevance_score = link_data['relevance_score']
                        existing_link.keywords = link_data['keywords']
                        existing_link.updated_at = datetime.utcnow()
                    db_links.append(existing_link)
                else:
                    # Create new link
                    new_link = DBLink(
                        url=url,
                        title=link_data['title'],
                        content_type=link_data['content_type'],
                        relevance_score=link_data['relevance_score'],
                        keywords=link_data['keywords'],
                        link_metadata={}
                    )
                    db.add(new_link)
                    db_links.append(new_link)
                
                db.commit()
                    
            except IntegrityError:
                logger.warning(f"Duplicate URL encountered: {url}, skipping...")
                db.rollback()
                continue
            except Exception as e:
                logger.error(f"Error processing link {link_data.get('url', 'unknown')}: {str(e)}")
                db.rollback()
                continue
        
        # Sort by relevance score before returning
        db_links.sort(key=lambda x: x.relevance_score, reverse=True)
        return db_links
        
    except Exception as e:
        logger.error(f"Error scraping URL: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/links", response_model=List[Link])
async def get_links(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    min_relevance: Optional[float] = Query(None, ge=0.0, le=1.0),
    content_type: Optional[str] = None,
    keywords: Optional[List[str]] = Query(None),
    sort_by: str = Query("relevance", enum=["relevance", "date"]),
    db: Session = Depends(get_db)
):
    """
    Retrieve stored links with filtering and sorting options.
    
    - skip: Number of records to skip
    - limit: Maximum number of records to return
    - min_relevance: Minimum relevance score filter
    - content_type: Filter by content type
    - keywords: Filter by keywords
    - sort_by: Sort by 'relevance' or 'date'
    """
    query = db.query(DBLink)
    
    # Apply filters
    if min_relevance is not None:
        query = query.filter(DBLink.relevance_score >= min_relevance)
    
    if content_type:
        query = query.filter(DBLink.content_type == content_type)
        
    if keywords:
        # Filter links that have any of the specified keywords
        query = query.filter(DBLink.keywords.overlap(keywords))
    
    # Apply sorting
    if sort_by == "date":
        query = query.order_by(desc(DBLink.created_at))
    else:  # sort by relevance
        query = query.order_by(desc(DBLink.relevance_score))
    
    links = query.offset(skip).limit(limit).all()
    return links

@app.delete("/links/{link_id}")
async def delete_link(
    link_id: int,
    db: Session = Depends(get_db)
):
    """Delete a link by ID."""
    link = db.query(DBLink).filter(DBLink.id == link_id).first()
    if not link:
        raise HTTPException(status_code=404, detail="Link not found")
    
    db.delete(link)
    db.commit()
    return {"status": "success", "message": f"Link {link_id} deleted"}
