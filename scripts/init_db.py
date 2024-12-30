import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models.base import Base, engine, SessionLocal
import logging
from sqlalchemy import text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_db():
    """Initialize the database by creating all tables."""
    try:
        # Create new tables
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
        
        # Get a database session
        db = SessionLocal()
        
        try:
            # Check if we need to migrate old data
            inspector = db.get_inspector()
            if 'links' in inspector.get_table_names():
                # Check if old schema exists
                result = db.execute(text("SELECT COUNT(*) FROM links WHERE relevance_score IS NOT NULL"))
                has_old_data = result.scalar() > 0
                
                if has_old_data:
                    logger.info("Migrating old data to new schema...")
                    # Migrate data
                    db.execute(text("""
                        UPDATE links 
                        SET ensemble_score = relevance_score,
                            semantic_score = relevance_score,
                            openai_score = relevance_score,
                            nlp_score = relevance_score,
                            deep_learning_score = relevance_score,
                            metadata = link_metadata
                        WHERE relevance_score IS NOT NULL
                    """))
                    db.commit()
                    logger.info("Data migration complete")
            
        except Exception as e:
            logger.error(f"Error during data migration: {str(e)}")
            db.rollback()
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Error creating database tables: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    logger.info("Initializing database...")
    init_db() 
