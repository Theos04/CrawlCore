import re
from typing import List, Dict, Any, Set
from collections import Counter
from datetime import datetime

from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class HashtagExtractor:
    """Extracts and analyzes hashtags from text"""
    
    def __init__(self):
        self.hashtag_pattern = re.compile(r'#(\w+)')
        
    def extract(self, text: str) -> List[str]:
        """Extract hashtags from text"""
        if not text:
            return []
        
        # Find all hashtags
        hashtags = self.hashtag_pattern.findall(text)
        
        # Normalize (lowercase) and remove duplicates while preserving order
        seen = set()
        normalized = []
        for tag in hashtags:
            tag_lower = tag.lower()
            if tag_lower not in seen:
                seen.add(tag_lower)
                normalized.append(tag_lower)
        
        return normalized
    
    def extract_with_positions(self, text: str) -> List[Dict[str, Any]]:
        """Extract hashtags with their positions in text"""
        if not text:
            return []
        
        hashtags = []
        for match in self.hashtag_pattern.finditer(text):
            hashtags.append({
                'tag': match.group(1),
                'tag_lower': match.group(1).lower(),
                'start': match.start(),
                'end': match.end(),
                'full_match': match.group(0)
            })
        
        return hashtags
    
    def analyze_hashtags(self, posts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze hashtag usage across multiple posts"""
        all_hashtags = []
        post_hashtags = []
        
        for post in posts:
            text = post.get('text', '')
            hashtags = self.extract(text)
            all_hashtags.extend(hashtags)
            post_hashtags.append({
                'post_id': post.get('post_id'),
                'hashtags': hashtags,
                'count': len(hashtags)
            })
        
        # Count frequencies
        hashtag_counts = Counter(all_hashtags)
        
        # Calculate statistics
        total_posts = len(posts)
        posts_with_hashtags = sum(1 for p in post_hashtags if p['count'] > 0)
        
        return {
            'total_hashtags_used': len(hashtag_counts),
            'total_occurrences': len(all_hashtags),
            'posts_with_hashtags': posts_with_hashtags,
            'hashtag_percentage': (posts_with_hashtags / total_posts * 100) if total_posts > 0 else 0,
            'top_hashtags': hashtag_counts.most_common(20),
            'avg_hashtags_per_post': len(all_hashtags) / total_posts if total_posts > 0 else 0,
            'post_details': post_hashtags
        }
    
    def extract_categories(self, hashtags: List[str]) -> Dict[str, List[str]]:
        """Categorize hashtags by common topics"""
        categories = {
            'technology': ['ai', 'tech', 'coding', 'programming', 'software', 'dev'],
            'business': ['business', 'marketing', 'startup', 'entrepreneur', 'money'],
            'health': ['health', 'fitness', 'wellness', 'mentalhealth', 'workout'],
            'lifestyle': ['life', 'lifestyle', 'travel', 'food', 'fashion'],
            'entertainment': ['music', 'movie', 'art', 'gaming', 'sports'],
            'news': ['news', 'breaking', 'politics', 'world', 'trending'],
            'other': []
        }
        
        categorized = {cat: [] for cat in categories}
        
        for tag in hashtags:
            categorized_flag = False
            tag_lower = tag.lower()
            
            for category, keywords in categories.items():
                if any(keyword in tag_lower for keyword in keywords):
                    categorized[category].append(tag)
                    categorized_flag = True
                    break
            
            if not categorized_flag:
                categorized['other'].append(tag)
        
        return categorized
    
    def extract_trending(self, posts: List[Dict[str, Any]], window_minutes: int = 60) -> List[Dict[str, Any]]:
        """Identify trending hashtags in recent posts"""
        now = datetime.utcnow()
        
        # Filter posts within time window
        recent_posts = []
        for post in posts:
            timestamp = post.get('timestamp')
            if timestamp:
                try:
                    post_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    time_diff = (now - post_time).total_seconds() / 60
                    if time_diff <= window_minutes:
                        recent_posts.append(post)
                except:
                    continue
        
        if not recent_posts:
            return []
        
        # Extract hashtags from recent posts
        recent_hashtags = []
        for post in recent_posts:
            recent_hashtags.extend(self.extract(post.get('text', '')))
        
        # Count frequencies
        tag_counts = Counter(recent_hashtags)
        
        # Calculate velocity (occurrences per minute)
        trending = []
        for tag, count in tag_counts.most_common(20):
            trending.append({
                'tag': tag,
                'occurrences': count,
                'velocity': count / window_minutes,
                'posts': [p for p in recent_posts if tag in self.extract(p.get('text', ''))]
            })
        
        return trending