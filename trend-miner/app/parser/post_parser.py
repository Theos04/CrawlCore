from typing import Dict, Any, List, Optional
from bs4 import BeautifulSoup
import re
from datetime import datetime

from app.parser.media_extractor import MediaExtractor
from app.parser.metadata_extractor import MetadataExtractor
from app.utils.logger import setup_logger
from app.utils.fingerprint import generate_fingerprint

logger = setup_logger(__name__)

class PostParser:
    """Parses raw HTML posts into structured format"""
    
    def __init__(self):
        self.media_extractor = MediaExtractor()
        self.metadata_extractor = MetadataExtractor()
        
    def parse(self, raw_post: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse raw post data into structured format"""
        try:
            platform = raw_post.get('platform', 'unknown')
            raw_html = raw_post.get('raw_html', '')
            
            if not raw_html:
                logger.warning("Empty raw HTML content")
                return None
            
            # Parse based on platform
            if platform == 'threads':
                parsed = self._parse_threads_post(raw_post)
            else:
                parsed = self._parse_generic_post(raw_post)
            
            if parsed:
                # Add parsing metadata
                parsed['parsed_at'] = datetime.utcnow().isoformat()
                parsed['raw_post_id'] = raw_post.get('_id')
                
                # Generate fingerprint for deduplication
                parsed['fingerprint'] = generate_fingerprint(
                    parsed.get('text', ''),
                    parsed.get('author', '')
                )
                
            return parsed
            
        except Exception as e:
            logger.error(f"Error parsing post: {e}", exc_info=True)
            return None
    
    def _parse_threads_post(self, raw_post: Dict[str, Any]) -> Dict[str, Any]:
        """Parse Threads-specific post format"""
        soup = BeautifulSoup(raw_post.get('raw_html', ''), 'html.parser')
        
        # Extract basic fields (already partially extracted by scraper)
        parsed = {
            'post_id': raw_post.get('post_id', ''),
            'platform': 'threads',
            'author': raw_post.get('author', ''),
            'text': raw_post.get('text', ''),
            'images': raw_post.get('images', []),
            'likes': raw_post.get('likes', 0),
            'replies': raw_post.get('replies', 0),
            'timestamp': raw_post.get('timestamp', ''),
        }
        
        # Extract additional metadata from HTML
        parsed['mentions'] = self._extract_mentions(parsed['text'])
        parsed['urls'] = self._extract_urls(parsed['text'])
        
        # Extract media URLs
        media = self.media_extractor.extract_from_html(soup)
        parsed['videos'] = media.get('videos', [])
        
        # If images not already extracted, get from HTML
        if not parsed['images']:
            parsed['images'] = media.get('images', [])
        
        # Extract metadata
        meta = self.metadata_extractor.extract(soup)
        parsed.update(meta)
        
        return parsed
    
    def _parse_generic_post(self, raw_post: Dict[str, Any]) -> Dict[str, Any]:
        """Parse generic post format (fallback)"""
        soup = BeautifulSoup(raw_post.get('raw_html', ''), 'html.parser')
        
        # Try to find common elements
        text = self._find_text_content(soup)
        
        return {
            'post_id': raw_post.get('post_id', generate_fingerprint(text)),
            'platform': raw_post.get('platform', 'unknown'),
            'author': self._find_author(soup),
            'text': text,
            'images': self.media_extractor.extract_images(soup),
            'mentions': self._extract_mentions(text),
            'urls': self._extract_urls(text),
            'timestamp': self._find_timestamp(soup),
            'likes': 0,
            'replies': 0
        }
    
    def _find_text_content(self, soup: BeautifulSoup) -> str:
        """Find main text content in HTML"""
        # Try common text containers
        selectors = [
            'div[dir="auto"]',
            'div.post-content',
            'div.tweet-text',
            'p',
            'article div'
        ]
        
        for selector in selectors:
            elements = soup.select(selector)
            if elements:
                # Combine texts from multiple elements
                texts = [elem.get_text(strip=True) for elem in elements if elem.get_text(strip=True)]
                if texts:
                    return ' '.join(texts)
        
        return ''
    
    def _find_author(self, soup: BeautifulSoup) -> str:
        """Find author name in HTML"""
        selectors = [
            'span.author',
            'a[href*="/profile/"]',
            'div.user-info'
        ]
        
        for selector in selectors:
            elem = soup.select_one(selector)
            if elem:
                return elem.get_text(strip=True)
        
        return ''
    
    def _find_timestamp(self, soup: BeautifulSoup) -> str:
        """Find timestamp in HTML"""
        selectors = [
            'time[datetime]',
            'span.timestamp',
            'div.date'
        ]
        
        for selector in selectors:
            elem = soup.select_one(selector)
            if elem:
                if elem.name == 'time':
                    return elem.get('datetime', '')
                return elem.get_text(strip=True)
        
        return ''
    
    def _extract_mentions(self, text: str) -> List[str]:
        """Extract @mentions from text"""
        if not text:
            return []
        return re.findall(r'@(\w+)', text)
    
    def _extract_urls(self, text: str) -> List[str]:
        """Extract URLs from text"""
        if not text:
            return []
        url_pattern = r'https?://[^\s]+'
        return re.findall(url_pattern, text)
    
    def parse_batch(self, raw_posts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Parse multiple raw posts"""
        parsed_posts = []
        
        for raw_post in raw_posts:
            parsed = self.parse(raw_post)
            if parsed:
                parsed_posts.append(parsed)
        
        logger.info(f"Parsed {len(parsed_posts)} posts from {len(raw_posts)} raw posts")
        return parsed_posts