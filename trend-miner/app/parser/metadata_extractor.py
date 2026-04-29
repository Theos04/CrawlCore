from typing import Dict, Any
from bs4 import BeautifulSoup
from datetime import datetime

from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class MetadataExtractor:
    """Extracts metadata from HTML content"""
    
    def extract(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract all metadata from HTML"""
        metadata = {}
        
        # Extract meta tags
        metadata.update(self._extract_meta_tags(soup))
        
        # Extract Open Graph data
        metadata.update(self._extract_open_graph(soup))
        
        # Extract Twitter Card data
        metadata.update(self._extract_twitter_card(soup))
        
        # Extract JSON-LD if present
        metadata.update(self._extract_json_ld(soup))
        
        return metadata
    
    def _extract_meta_tags(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract standard meta tags"""
        meta_data = {}
        
        for meta in soup.find_all('meta'):
            # Get name and content
            name = meta.get('name') or meta.get('property') or meta.get('http-equiv')
            content = meta.get('content')
            
            if name and content:
                # Clean up name
                name = name.lower().replace(':', '_').replace('-', '_')
                meta_data[f'meta_{name}'] = content
        
        return meta_data
    
    def _extract_open_graph(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract Open Graph protocol metadata"""
        og_data = {}
        
        for meta in soup.find_all('meta'):
            property = meta.get('property', '')
            if property.startswith('og:'):
                content = meta.get('content')
                if content:
                    key = property.replace(':', '_')
                    og_data[key] = content
        
        return og_data
    
    def _extract_twitter_card(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract Twitter Card metadata"""
        twitter_data = {}
        
        for meta in soup.find_all('meta'):
            name = meta.get('name', '')
            if name.startswith('twitter:'):
                content = meta.get('content')
                if content:
                    key = name.replace(':', '_')
                    twitter_data[key] = content
        
        return twitter_data
    
    def _extract_json_ld(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract JSON-LD structured data"""
        import json
        json_data = {}
        for script in soup.find_all('script', type='application/ld+json'):
            try:
                data = json.loads(script.string)
                
                # Store as json_ld_0, json_ld_1, etc.
                idx = len([k for k in json_data.keys() if k.startswith('json_ld_')])
                json_data[f'json_ld_{idx}'] = data
                
            except Exception as e:
                logger.debug(f"Error parsing JSON-LD: {e}")
        
        return json_data
    
    def extract_article_data(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract article-specific metadata"""
        article_data = {}
        
        # Article title
        title_tag = soup.find('title')
        if title_tag:
            article_data['title'] = title_tag.string
        
        # Article headings
        h1_tags = soup.find_all('h1')
        if h1_tags:
            article_data['headings'] = [h1.get_text(strip=True) for h1 in h1_tags]
        
        # Publication date
        for meta in soup.find_all('meta'):
            if meta.get('property') in ['article:published_time', 'article:modified_time']:
                date_str = meta.get('content')
                if date_str:
                    try:
                        # Try to parse ISO format
                        article_data['published_date'] = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                    except:
                        article_data['published_date_raw'] = date_str
        
        # Author info
        author_meta = soup.find('meta', {'name': 'author'})
        if author_meta:
            article_data['author'] = author_meta.get('content')
        
        return article_data
    
    def extract_social_counts(self, soup: BeautifulSoup) -> Dict[str, int]:
        """Extract social engagement counts if available"""
        counts = {}
        
        # Look for common patterns
        text = soup.get_text()
        
        # Find patterns like "1.2K likes" or "500 shares"
        import re
        
        # Likes
        like_patterns = [
            r'(\d+(?:\.\d+)?[KM]?)\s*(?:like|❤)s?',
            r'(\d+(?:\.\d+)?[KM]?)\s*(?:favorite|fav)s?'
        ]
        
        for pattern in like_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                counts['likes'] = self._parse_number(match.group(1))
                break
        
        # Comments/Replies
        comment_patterns = [
            r'(\d+(?:\.\d+)?[KM]?)\s*(?:comment|reply)s?',
            r'(\d+(?:\.\d+)?[KM]?)\s*(?:💬|discussion)'
        ]
        
        for pattern in comment_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                counts['comments'] = self._parse_number(match.group(1))
                break
        
        # Shares
        share_patterns = [
            r'(\d+(?:\.\d+)?[KM]?)\s*(?:share|retweet)s?',
            r'(\d+(?:\.\d+)?[KM]?)\s*(?:🔁|repost)'
        ]
        
        for pattern in share_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                counts['shares'] = self._parse_number(match.group(1))
                break
        
        return counts
    
    def _parse_number(self, text: str) -> int:
        """Parse number with K/M suffix"""
        try:
            text = text.strip()
            multiplier = 1
            
            if text.endswith('K'):
                multiplier = 1000
                text = text[:-1]
            elif text.endswith('M'):
                multiplier = 1000000
                text = text[:-1]
            
            return int(float(text) * multiplier)
        except:
            return 0