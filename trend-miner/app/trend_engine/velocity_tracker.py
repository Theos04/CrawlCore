from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict, Counter

from app.storage.mongodb import MongoDBClient
from app.utils.logger import setup_logger
from app.config import VELOCITY_WINDOW_MINUTES

logger = setup_logger(__name__)

class VelocityTracker:
    """Tracks velocity of topics, hashtags, and engagement"""
    
    def __init__(self):
        self.db = MongoDBClient()
        
    def calculate_post_velocity(self, post: Dict[str, Any]) -> float:
        """Calculate engagement velocity for a single post"""
        likes = post.get('likes', 0)
        replies = post.get('replies', 0)
        timestamp = post.get('timestamp')
        
        if not timestamp:
            return 0.0
        
        try:
            # Parse timestamp
            if isinstance(timestamp, str):
                post_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            else:
                post_time = timestamp
            
            # Calculate minutes since post
            now = datetime.utcnow().replace(tzinfo=post_time.tzinfo)
            minutes_since = (now - post_time).total_seconds() / 60
            
            if minutes_since <= 0:
                return float(likes + replies * 2)
            
            # Calculate velocity (engagement per minute)
            velocity = (likes + replies * 2) / minutes_since
            return round(velocity, 3)
            
        except Exception as e:
            logger.error(f"Error calculating velocity: {e}")
            return 0.0
    
    def calculate_hashtag_velocity(self, hashtag: str, 
                                   window_minutes: int = VELOCITY_WINDOW_MINUTES) -> Dict[str, Any]:
        """Calculate velocity for a hashtag"""
        # Get posts with this hashtag in time window
        posts = self.db.get_posts_by_hashtag(hashtag, limit=100)
        
        if not posts:
            return {
                'hashtag': hashtag,
                'velocity': 0.0,
                'occurrences': 0,
                'avg_engagement': 0.0
            }
        
        # Filter by time window
        now = datetime.utcnow()
        window_posts = []
        
        for post in posts:
            timestamp = post.get('timestamp')
            if timestamp:
                try:
                    post_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    minutes_ago = (now - post_time).total_seconds() / 60
                    if minutes_ago <= window_minutes:
                        window_posts.append(post)
                except:
                    continue
        
        if not window_posts:
            return {
                'hashtag': hashtag,
                'velocity': 0.0,
                'occurrences': 0,
                'avg_engagement': 0.0
            }
        
        # Calculate metrics
        occurrences = len(window_posts)
        total_engagement = sum(p.get('engagement_score', 0) for p in window_posts)
        avg_engagement = total_engagement / occurrences if occurrences > 0 else 0
        
        # Velocity = occurrences per minute * average engagement
        velocity = (occurrences / window_minutes) * (1 + avg_engagement)
        
        return {
            'hashtag': hashtag,
            'velocity': round(velocity, 3),
            'occurrences': occurrences,
            'avg_engagement': round(avg_engagement, 3),
            'posts': [p.get('post_id') for p in window_posts]
        }
    
    def calculate_topic_velocity(self, cluster_id: str,
                                window_minutes: int = VELOCITY_WINDOW_MINUTES) -> Dict[str, Any]:
        """Calculate velocity for a topic cluster"""
        # Get posts in this cluster
        posts = list(self.db.db[self.db.PROCESSED_COLLECTION].find(
            {"cluster_id": cluster_id}
        ))
        
        if not posts:
            return {
                'cluster_id': cluster_id,
                'velocity': 0.0,
                'occurrences': 0,
                'avg_engagement': 0.0
            }
        
        # Filter by time window
        now = datetime.utcnow()
        window_posts = []
        
        for post in posts:
            timestamp = post.get('timestamp')
            if timestamp:
                try:
                    post_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    minutes_ago = (now - post_time).total_seconds() / 60
                    if minutes_ago <= window_minutes:
                        window_posts.append(post)
                except:
                    continue
        
        if not window_posts:
            return {
                'cluster_id': cluster_id,
                'velocity': 0.0,
                'occurrences': 0,
                'avg_engagement': 0.0
            }
        
        # Calculate metrics
        occurrences = len(window_posts)
        total_engagement = sum(p.get('engagement_score', 0) for p in window_posts)
        avg_engagement = total_engagement / occurrences if occurrences > 0 else 0
        
        # Velocity with momentum factor
        # Also consider rate of increase compared to previous window
        previous_posts = [p for p in posts if p not in window_posts]
        prev_occurrences = len(previous_posts)
        
        if prev_occurrences > 0:
            growth_rate = occurrences / prev_occurrences
        else:
            growth_rate = 1.0
        
        velocity = (occurrences / window_minutes) * (1 + avg_engagement) * growth_rate
        
        return {
            'cluster_id': cluster_id,
            'velocity': round(velocity, 3),
            'occurrences': occurrences,
            'avg_engagement': round(avg_engagement, 3),
            'growth_rate': round(growth_rate, 3),
            'posts': [p.get('post_id') for p in window_posts]
        }
    
    def get_trending_hashtags(self, limit: int = 20,
                             window_minutes: int = VELOCITY_WINDOW_MINUTES) -> List[Dict[str, Any]]:
        """Get trending hashtags by velocity"""
        # Get all hashtag stats
        hashtag_stats = self.db.get_hashtag_stats(limit=100)
        
        trending = []
        for stat in hashtag_stats:
            hashtag = stat['_id']
            velocity_data = self.calculate_hashtag_velocity(hashtag, window_minutes)
            
            if velocity_data['velocity'] > 0:
                velocity_data.update({
                    'total_count': stat['count'],
                    'avg_engagement_all': stat['avg_engagement']
                })
                trending.append(velocity_data)
        
        # Sort by velocity
        trending.sort(key=lambda x: x['velocity'], reverse=True)
        
        return trending[:limit]
    
    def get_trending_topics(self, limit: int = 10,
                           window_minutes: int = VELOCITY_WINDOW_MINUTES) -> List[Dict[str, Any]]:
        """Get trending topics by velocity"""
        # Get all clusters
        clusters = self.db.db[self.db.PROCESSED_COLLECTION].aggregate([
            {"$group": {
                "_id": "$cluster_id",
                "count": {"$sum": 1},
                "avg_engagement": {"$avg": "$engagement_score"}
            }},
            {"$match": {"_id": {"$ne": "cluster_-1"}}}
        ])
        
        trending = []
        for cluster in clusters:
            cluster_id = cluster['_id']
            if cluster_id:
                velocity_data = self.calculate_topic_velocity(cluster_id, window_minutes)
                
                if velocity_data['velocity'] > 0:
                    velocity_data.update({
                        'total_count': cluster['count'],
                        'avg_engagement_all': cluster['avg_engagement']
                    })
                    trending.append(velocity_data)
        
        # Sort by velocity
        trending.sort(key=lambda x: x['velocity'], reverse=True)
        
        return trending[:limit]
    
    def get_velocity_over_time(self, hashtag: str, 
                              hours: int = 24, 
                              interval_minutes: int = 60) -> List[Dict[str, Any]]:
        """Get velocity over time for a hashtag"""
        # Get posts
        posts = self.db.get_posts_by_hashtag(hashtag, limit=1000)
        
        if not posts:
            return []
        
        # Create time buckets
        now = datetime.utcnow()
        intervals = []
        
        for i in range(hours * 60 // interval_minutes):
            end_time = now - timedelta(minutes=i * interval_minutes)
            start_time = end_time - timedelta(minutes=interval_minutes)
            
            # Count posts in this interval
            interval_posts = []
            for post in posts:
                timestamp = post.get('timestamp')
                if timestamp:
                    try:
                        post_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        if start_time <= post_time < end_time:
                            interval_posts.append(post)
                    except:
                        continue
            
            # Calculate velocity for this interval
            if interval_posts:
                occurrences = len(interval_posts)
                avg_engagement = sum(p.get('engagement_score', 0) for p in interval_posts) / occurrences
                velocity = (occurrences / interval_minutes) * (1 + avg_engagement)
            else:
                velocity = 0.0
            
            intervals.append({
                'timestamp': end_time.isoformat(),
                'velocity': round(velocity, 3),
                'occurrences': len(interval_posts)
            })
        
        # Reverse to get chronological order
        intervals.reverse()
        
        return intervals