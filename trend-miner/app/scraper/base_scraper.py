from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datetime import datetime
import time

from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class BaseScraper(ABC):
    """Abstract base class for all platform scrapers"""

    def __init__(self, platform_name: str):
        from app.storage.mongodb import MongoDBClient
        self.platform = platform_name
        self.db = MongoDBClient()
        self.session = None
        self.driver = None
        
    @abstractmethod
    def login(self, credentials: Dict[str, str]) -> bool:
        """Handle platform login if required"""
        pass
    
    @abstractmethod
    def scrape_feed(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Scrape posts from the feed"""
        pass
    
    @abstractmethod
    def extract_post_data(self, post_element: Any) -> Dict[str, Any]:
        """Extract raw data from a single post element"""
        pass
    
    def save_raw_data(self, raw_data: Dict[str, Any]) -> str:
        """Save raw scraped data to database"""
        raw_data.update({
            "platform": self.platform,
            "scraped_at": datetime.utcnow().isoformat(),
            "status": "raw"
        })
        
        post_id = self.db.insert_raw_post(raw_data)
        logger.debug(f"Saved raw post {post_id}")
        return post_id
    
    def scrape(self, max_posts: Optional[int] = None) -> List[Dict[str, Any]]:
        """Main scraping method with error handling and rate limiting"""
        try:
            logger.info(f"Starting scrape for {self.platform}")
            
            raw_posts = self.scrape_feed(limit=max_posts)
            
            # Save each post
            for post in raw_posts:
                self.save_raw_data(post)
                
            logger.info(f"Scraped {len(raw_posts)} posts from {self.platform}")
            return raw_posts
            
        except Exception as e:
            logger.error(f"Scraping failed: {e}", exc_info=True)
            return []
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources after scraping"""
        if self.driver:
            self.driver.quit()
            self.driver = None
            
    def rate_limit(self, seconds: float = 1.0):
        """Apply rate limiting between requests"""
        time.sleep(seconds)