from fastapi import FastAPI, HTTPException, Depends, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from . import models
from .models.base import get_db, engine
from .models.link import Link as DBLink
from .api.models import Link, LinkCreate, ScrapeRequest, HealthCheck
from .core.scraper import LinkScraper
import logging
from datetime import datetime
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy import update
from sqlalchemy.exc import IntegrityError

# Create database tables
models.base.Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="High-Value Link Scraper API",
    description="API for scraping and retrieving high-value links with intelligent ranking",
    version="1.0.0"
)

logger = logging.getLogger(__name__)
scraper = LinkScraper()

@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Check API health status."""
    return {"status": "healthy", "version": "1.0.0"}

@app.post("/scrape", response_model=List[Link])
async def scrape_url(
    request: ScrapeRequest,
    db: Session = Depends(get_db)
):
    """Scrape links from a URL and store them in the database."""
    try:
        links = await scraper.scrape(str(request.url))
        
        # Limit the number of links if specified
        links = links[:request.max_links]
        
        # Store links in database
        db_links = []
        seen_urls = set()  # Track URLs we've processed
        
        for link_data in links:
            try:
                url = str(link_data['url'])
                
                # Skip if we've already processed this URL in this batch
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
                
                # Commit after each successful operation
                db.commit()
                    
            except IntegrityError as e:
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
    db: Session = Depends(get_db)
):
    """Retrieve stored links with optional filtering."""
    query = db.query(DBLink)
    
    if min_relevance is not None:
        query = query.filter(DBLink.relevance_score >= min_relevance)
    
    if content_type:
        query = query.filter(DBLink.content_type == content_type)
    
    query = query.order_by(DBLink.relevance_score.desc())
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
