from typing import List, Dict, Any
from bs4 import BeautifulSoup
import re
from urllib.parse import urlparse

from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class MediaExtractor:
    """Extracts media content (images, videos) from HTML"""
    
    def __init__(self):
        self.supported_image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.svg'}
        self.supported_video_extensions = {'.mp4', '.webm', '.ogg', '.mov'}
        
    def extract_from_html(self, soup: BeautifulSoup) -> Dict[str, List[str]]:
        """Extract all media from HTML"""
        return {
            'images': self.extract_images(soup),
            'videos': self.extract_videos(soup)
        }
    
    def extract_images(self, soup: BeautifulSoup) -> List[str]:
        """Extract image URLs from HTML"""
        images = []
        
        # Find img tags
        for img in soup.find_all('img'):
            src = img.get('src', '')
            if src and self._is_valid_image(src):
                images.append(src)
            
            # Check data-src for lazy loading
            data_src = img.get('data-src', '')
            if data_src and self._is_valid_image(data_src):
                images.append(data_src)
        
        # Find background images in style attributes
        for tag in soup.find_all(style=True):
            style = tag['style']
            urls = re.findall(r'url\(["\']?(.*?)["\']?\)', style)
            for url in urls:
                if self._is_valid_image(url):
                    images.append(url)
        
        # Find Open Graph images
        og_image = soup.find('meta', property='og:image')
        if og_image and og_image.get('content'):
            images.append(og_image['content'])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_images = []
        for img in images:
            if img not in seen:
                seen.add(img)
                unique_images.append(img)
        
        return unique_images
    
    def extract_videos(self, soup: BeautifulSoup) -> List[str]:
        """Extract video URLs from HTML"""
        videos = []
        
        # Find video tags
        for video in soup.find_all('video'):
            # Check src attribute
            src = video.get('src', '')
            if src and self._is_valid_video(src):
                videos.append(src)
            
            # Check source tags inside video
            for source in video.find_all('source'):
                src = source.get('src', '')
                if src and self._is_valid_video(src):
                    videos.append(src)
        
        # Find iframe embeds (YouTube, Vimeo, etc.)
        for iframe in soup.find_all('iframe'):
            src = iframe.get('src', '')
            if src and self._is_video_embed(src):
                videos.append(src)
        
        # Find Open Graph videos
        og_video = soup.find('meta', property='og:video')
        if og_video and og_video.get('content'):
            videos.append(og_video['content'])
        
        return list(set(videos))  # Remove duplicates
    
    def _is_valid_image(self, url: str) -> bool:
        """Check if URL points to a valid image"""
        try:
            parsed = urlparse(url)
            ext = self._get_extension(parsed.path).lower()
            return ext in self.supported_image_extensions
        except:
            return False
    
    def _is_valid_video(self, url: str) -> bool:
        """Check if URL points to a valid video"""
        try:
            parsed = urlparse(url)
            ext = self._get_extension(parsed.path).lower()
            return ext in self.supported_video_extensions
        except:
            return False
    
    def _is_video_embed(self, url: str) -> bool:
        """Check if URL is a video embed (YouTube, Vimeo, etc.)"""
        video_domains = {
            'youtube.com', 'youtu.be', 'vimeo.com', 'dailymotion.com',
            'player.vimeo.com', 'www.youtube.com', 'www.vimeo.com'
        }
        
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            return any(vd in domain for vd in video_domains)
        except:
            return False
    
    def _get_extension(self, path: str) -> str:
        """Get file extension from path"""
        if '.' in path:
            return '.' + path.split('.')[-1].split('?')[0]
        return ''
    
    def is_media_url(self, url: str) -> bool:
        """Check if URL points to any media type"""
        return self._is_valid_image(url) or self._is_valid_video(url) or self._is_video_embed(url)