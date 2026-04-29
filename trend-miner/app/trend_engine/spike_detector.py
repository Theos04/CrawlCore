from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict, Counter


def _get_logger():
    from app.utils.logger import setup_logger
    return setup_logger(__name__)


class SpikeDetector:
    """Detects sudden spikes in topics and engagement"""

    def __init__(self):
        from app.storage.mongodb import MongoDBClient
        from app.trend_engine.velocity_tracker import VelocityTracker
        self.db = MongoDBClient()
        self.velocity_tracker = VelocityTracker()

    def detect_hashtag_spikes(self, baseline_hours: int = 24,
                              spike_threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """Detect spikes in hashtag usage"""
        if spike_threshold is None:
            from app.config import SPIKE_THRESHOLD
            spike_threshold = SPIKE_THRESHOLD
        # Get baseline hashtag frequencies
        baseline_posts = self.db.get_recent_posts(limit=1000, hours=baseline_hours)
        
        baseline_counts = Counter()
        for post in baseline_posts:
            hashtags = post.get('hashtags', [])
            baseline_counts.update(hashtags)
        
        # Get current window frequencies
        current_posts = self.db.get_recent_posts(limit=500, hours=1)
        
        current_counts = Counter()
        for post in current_posts:
            hashtags = post.get('hashtags', [])
            current_counts.update(hashtags)
        
        # Detect spikes
        spikes = []
        
        for hashtag, current_count in current_counts.items():
            baseline_count = baseline_counts.get(hashtag, 0)
            
            # Avoid division by zero
            if baseline_count == 0:
                if current_count > 0:
                    ratio = float('inf')
                else:
                    ratio = 0
            else:
                ratio = current_count / baseline_count
            
            # Calculate normalized ratio
            if ratio > spike_threshold:
                # Get velocity data
                velocity_data = self.velocity_tracker.calculate_hashtag_velocity(
                    hashtag, window_minutes=60
                )
                
                spike = {
                    'type': 'hashtag',
                    'topic': hashtag,
                    'current_count': current_count,
                    'baseline_count': baseline_count,
                    'ratio': ratio if ratio != float('inf') else 999.0,
                    'velocity': velocity_data['velocity'],
                    'occurrences': velocity_data['occurrences'],
                    'avg_engagement': velocity_data['avg_engagement'],
                    'detected_at': datetime.utcnow().isoformat(),
                    'posts': velocity_data.get('posts', [])
                }
                
                spikes.append(spike)
        
        # Sort by spike intensity
        spikes.sort(key=lambda x: x['ratio'], reverse=True)
        
        _get_logger().info(f"Detected {len(spikes)} hashtag spikes")
        return spikes

    def detect_topic_spikes(self, baseline_hours: int = 24,
                           spike_threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """Detect spikes in topic clusters"""
        if spike_threshold is None:
            from app.config import SPIKE_THRESHOLD
            spike_threshold = SPIKE_THRESHOLD
        # Get all clusters
        clusters = self.db.db[self.db.PROCESSED_COLLECTION].aggregate([
            {"$group": {
                "_id": "$cluster_id",
                "count": {"$sum": 1},
                "avg_engagement": {"$avg": "$engagement_score"}
            }},
            {"$match": {"_id": {"$ne": "cluster_-1"}}}
        ])
        
        clusters_list = list(clusters)
        
        if not clusters_list:
            return []
        
        # Calculate baseline and current for each cluster
        now = datetime.utcnow()
        baseline_time = now - timedelta(hours=baseline_hours)
        current_time = now - timedelta(hours=1)
        
        spikes = []
        
        for cluster in clusters_list:
            cluster_id = cluster['_id']
            
            # Get posts in this cluster
            posts = list(self.db.db[self.db.PROCESSED_COLLECTION].find(
                {"cluster_id": cluster_id}
            ))
            
            # Split into baseline and current
            baseline_posts = []
            current_posts = []
            
            for post in posts:
                timestamp = post.get('timestamp')
                if timestamp:
                    try:
                        post_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        if post_time < baseline_time:
                            continue
                        elif post_time >= current_time:
                            current_posts.append(post)
                        else:
                            baseline_posts.append(post)
                    except:
                        continue
            
            baseline_count = len(baseline_posts)
            current_count = len(current_posts)
            
            # Calculate ratio
            if baseline_count == 0:
                if current_count > 0:
                    ratio = float('inf')
                else:
                    ratio = 0
            else:
                ratio = current_count / baseline_count
            
            if ratio > spike_threshold:
                # Get velocity data
                velocity_data = self.velocity_tracker.calculate_topic_velocity(
                    cluster_id, window_minutes=60
                )
                
                # Get representative keywords
                keywords = []
                if current_posts:
                    # Extract common words
                    all_words = ' '.join([p.get('text', '') for p in current_posts]).lower().split()
                    word_counts = Counter(all_words)
                    keywords = [w for w, c in word_counts.most_common(10) if len(w) > 3]
                
                spike = {
                    'type': 'topic',
                    'topic': f"Topic {cluster_id}",
                    'cluster_id': cluster_id,
                    'current_count': current_count,
                    'baseline_count': baseline_count,
                    'ratio': ratio if ratio != float('inf') else 999.0,
                    'velocity': velocity_data['velocity'],
                    'keywords': keywords[:5],
                    'avg_engagement': velocity_data['avg_engagement'],
                    'detected_at': datetime.utcnow().isoformat(),
                    'posts': velocity_data.get('posts', [])
                }
                
                spikes.append(spike)
        
        # Sort by spike intensity
        spikes.sort(key=lambda x: x['ratio'], reverse=True)
        
        _get_logger().info(f"Detected {len(spikes)} topic spikes")
        return spikes
    
    def detect_engagement_spikes(self, threshold_percentile: float = 95) -> List[Dict[str, Any]]:
        """Detect posts with unusually high engagement"""
        # Get recent posts
        posts = self.db.get_recent_posts(limit=1000, hours=24)
        
        if not posts:
            return []
        
        # Calculate engagement distribution
        engagement_scores = [p.get('engagement_score', 0) for p in posts]
        
        if not engagement_scores:
            return []
        
        # Calculate threshold (e.g., 95th percentile)
        threshold = np.percentile(engagement_scores, threshold_percentile)
        
        # Find spikes
        spikes = []
        for post in posts:
            score = post.get('engagement_score', 0)
            if score >= threshold:
                # Calculate how much above threshold
                magnitude = (score - threshold) / threshold if threshold > 0 else score
                
                spikes.append({
                    'type': 'engagement',
                    'post_id': post.get('post_id'),
                    'text': post.get('text', '')[:200],
                    'engagement_score': score,
                    'threshold': threshold,
                    'magnitude': round(magnitude, 2),
                    'likes': post.get('likes', 0),
                    'replies': post.get('replies', 0),
                    'emotion': post.get('primary_emotion', 'neutral'),
                    'timestamp': post.get('timestamp'),
                    'detected_at': datetime.utcnow().isoformat()
                })
        
        # Sort by magnitude
        spikes.sort(key=lambda x: x['magnitude'], reverse=True)
        
        _get_logger().info(f"Detected {len(spikes)} engagement spikes")
        return spikes
    
    def detect_all_spikes(self) -> Dict[str, List[Dict[str, Any]]]:
        """Run all spike detection methods"""
        hashtag_spikes = self.detect_hashtag_spikes()
        topic_spikes = self.detect_topic_spikes()
        engagement_spikes = self.detect_engagement_spikes()
        
        # Store detected spikes
        all_spikes = {
            'hashtag_spikes': hashtag_spikes,
            'topic_spikes': topic_spikes,
            'engagement_spikes': engagement_spikes,
            'detected_at': datetime.utcnow().isoformat()
        }
        
        # Save to database
        self.db.insert_trend({
            'type': 'spike_detection',
            'data': all_spikes
        })
        
        return all_spikes
    
    def detect_trends(self) -> List[Dict[str, Any]]:
        """Main trend detection method - combines all detectors"""
        spikes = self.detect_all_spikes()
        
        # Combine and score all trends
        all_trends = []
        
        # Process hashtag spikes
        for spike in spikes.get('hashtag_spikes', []):
            all_trends.append({
                'trend_type': 'hashtag',
                'name': spike['topic'],
                'score': spike['ratio'] * spike['velocity'],
                'velocity': spike['velocity'],
                'volume': spike['occurrences'],
                'engagement': spike['avg_engagement'],
                'posts': spike['posts'],
                'detected_at': spike['detected_at']
            })
        
        # Process topic spikes
        for spike in spikes.get('topic_spikes', []):
            all_trends.append({
                'trend_type': 'topic',
                'name': spike['topic'],
                'cluster_id': spike['cluster_id'],
                'keywords': spike['keywords'],
                'score': spike['ratio'] * spike['velocity'],
                'velocity': spike['velocity'],
                'volume': spike['current_count'],
                'engagement': spike['avg_engagement'],
                'posts': spike['posts'],
                'detected_at': spike['detected_at']
            })
        
        # Sort by score
        all_trends.sort(key=lambda x: x['score'], reverse=True)
        
        # Save top trends
        for trend in all_trends[:20]:
            self.db.insert_trend(trend)
        
        _get_logger().info(f"Detected {len(all_trends)} total trends")
        return all_trends