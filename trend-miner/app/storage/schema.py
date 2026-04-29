from typing import Dict, Any, List, Optional
from datetime import datetime

class SchemaValidator:
    """Validates data against schemas"""
    
    @staticmethod
    def validate_raw_post(post: Dict[str, Any]) -> bool:
        """Validate raw post schema"""
        required_fields = ['platform', 'raw_html', 'scraped_at']
        
        for field in required_fields:
            if field not in post:
                return False
        
        # Check types
        if not isinstance(post.get('platform'), str):
            return False
        if not isinstance(post.get('raw_html'), str):
            return False
        
        return True
    
    @staticmethod
    def validate_processed_post(post: Dict[str, Any]) -> bool:
        """Validate processed post schema"""
        required_fields = [
            'post_id', 'platform', 'author', 'text', 
            'timestamp', 'processed_at'
        ]
        
        for field in required_fields:
            if field not in post:
                return False
        
        # Optional but should be present if available
        optional_fields = ['likes', 'replies', 'images', 'hashtags', 
                          'mentions', 'emotion', 'sentiment']
        
        # Type checks
        if not isinstance(post.get('post_id'), str):
            return False
        if not isinstance(post.get('platform'), str):
            return False
        if not isinstance(post.get('likes', 0), (int, float)):
            return False
        
        return True
    
    @staticmethod
    def validate_embedding(embedding: Dict[str, Any]) -> bool:
        """Validate embedding schema"""
        if 'post_id' not in embedding:
            return False
        if 'vector' not in embedding:
            return False
        
        if not isinstance(embedding.get('vector'), list):
            return False
        
        # Check vector dimension (should be > 0)
        if len(embedding['vector']) == 0:
            return False
        
        return True
    
    @staticmethod
    def validate_trend(trend: Dict[str, Any]) -> bool:
        """Validate trend schema"""
        required_fields = ['topic', 'score', 'detected_at']
        
        for field in required_fields:
            if field not in trend:
                return False
        
        return True

class SchemaFactory:
    """Creates schema-compliant objects"""
    
    @staticmethod
    def create_raw_post(platform: str, raw_html: str) -> Dict[str, Any]:
        """Create a raw post object"""
        return {
            'platform': platform,
            'raw_html': raw_html,
            'scraped_at': datetime.utcnow().isoformat(),
            'status': 'raw'
        }
    
    @staticmethod
    def create_processed_post(post_id: str, platform: str, author: str, 
                            text: str, timestamp: str) -> Dict[str, Any]:
        """Create a processed post object with defaults"""
        return {
            'post_id': post_id,
            'platform': platform,
            'author': author,
            'text': text,
            'cleaned_text': '',
            'images': [],
            'hashtags': [],
            'mentions': [],
            'emotion': 'neutral',
            'emotion_score': 0.0,
            'sentiment': 'neutral',
            'likes': 0,
            'replies': 0,
            'engagement_score': 0.0,
            'word_count': 0,
            'timestamp': timestamp,
            'scraped_at': datetime.utcnow().isoformat(),
            'processed_at': datetime.utcnow().isoformat(),
            'processing_version': '1.0'
        }
    
    @staticmethod
    def create_embedding(post_id: str, vector: List[float]) -> Dict[str, Any]:
        """Create an embedding object"""
        return {
            'post_id': post_id,
            'vector': vector,
            'created_at': datetime.utcnow().isoformat()
        }
    
    @staticmethod
    def create_trend(topic: str, score: float, posts: List[str]) -> Dict[str, Any]:
        """Create a trend object"""
        return {
            'topic': topic,
            'score': score,
            'posts': posts,
            'detected_at': datetime.utcnow().isoformat(),
            'status': 'active'
        }