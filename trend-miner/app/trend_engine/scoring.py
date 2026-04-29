from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import numpy as np

from app.storage.mongodb import MongoDBClient
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class TrendScorer:
    """Scores and ranks detected trends"""
    
    def __init__(self):
        self.db = MongoDBClient()
        
    def score_trend(self, trend: Dict[str, Any]) -> float:
        """Calculate comprehensive trend score"""
        score = 0.0
        weights = {
            'velocity': 0.3,
            'volume': 0.2,
            'engagement': 0.25,
            'growth': 0.15,
            'freshness': 0.1
        }
        
        # Velocity score (normalized)
        velocity = trend.get('velocity', 0)
        if velocity > 0:
            # Log scale to handle wide range
            velocity_score = np.log1p(velocity) / 10  # Normalize
            score += velocity_score * weights['velocity']
        
        # Volume score
        volume = trend.get('volume', 0) or trend.get('occurrences', 0)
        if volume > 0:
            volume_score = min(volume / 100, 1.0)  # Cap at 100 posts
            score += volume_score * weights['volume']
        
        # Engagement score
        engagement = trend.get('engagement', 0) or trend.get('avg_engagement', 0)
        if engagement > 0:
            engagement_score = min(engagement, 1.0)
            score += engagement_score * weights['engagement']
        
        # Growth score (ratio)
        ratio = trend.get('ratio', 1.0)
        if ratio > 1:
            growth_score = min((ratio - 1) / 10, 1.0)  # Cap at 10x growth
            score += growth_score * weights['growth']
        
        # Freshness score (newer trends score higher)
        detected_at = trend.get('detected_at')
        if detected_at:
            try:
                detected_time = datetime.fromisoformat(detected_at.replace('Z', '+00:00'))
                now = datetime.utcnow().replace(tzinfo=detected_time.tzinfo)
                hours_ago = (now - detected_time).total_seconds() / 3600
                freshness_score = max(0, 1 - (hours_ago / 24))  # Decay over 24 hours
                score += freshness_score * weights['freshness']
            except:
                pass
        
        return round(min(score, 1.0), 3)  # Ensure score is between 0-1
    
    def rank_trends(self, trends: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank trends by combined score"""
        ranked = []
        
        for trend in trends:
            trend_copy = trend.copy()
            trend_copy['combined_score'] = self.score_trend(trend)
            ranked.append(trend_copy)
        
        # Sort by combined score
        ranked.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return ranked
    
    def calculate_trend_potential(self, trend: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate future potential of a trend"""
        # Get historical data
        trend_type = trend.get('trend_type')
        name = trend.get('name')
        
        if trend_type == 'hashtag':
            # Get velocity over time
            from app.trend_engine.velocity_tracker import VelocityTracker
            tracker = VelocityTracker()
            velocity_history = tracker.get_velocity_over_time(name, hours=24, interval_minutes=60)
            
            if len(velocity_history) < 2:
                return {'potential': 0.5, 'confidence': 0.3}
            
            # Calculate trend
            velocities = [v['velocity'] for v in velocity_history]
            
            # Simple linear regression to predict trend
            x = np.arange(len(velocities))
            y = np.array(velocities)
            
            # Calculate slope
            if len(x) > 1 and np.std(x) > 0:
                slope = np.polyfit(x, y, 1)[0]
            else:
                slope = 0
            
            # Calculate momentum (acceleration)
            if len(velocities) > 2:
                momentum = velocities[-1] - velocities[-2]
            else:
                momentum = 0
            
            # Calculate volatility
            volatility = np.std(velocities) / (np.mean(velocities) if np.mean(velocities) > 0 else 1)
            
            # Potential score based on slope and momentum
            potential = 0.5 + (slope * 10) + (momentum * 5)
            potential = max(0, min(1, potential))
            
            # Confidence based on consistency (lower volatility = higher confidence)
            confidence = max(0, 1 - volatility)
            
            return {
                'potential': round(potential, 3),
                'confidence': round(confidence, 3),
                'slope': round(float(slope), 5),
                'momentum': round(float(momentum), 5),
                'volatility': round(float(volatility), 3),
                'velocity_trend': velocities[-5:] if len(velocities) >= 5 else velocities
            }
        
        return {'potential': 0.5, 'confidence': 0.3}
    
    def get_trend_summary(self, limit: int = 20) -> Dict[str, Any]:
        """Get summary of top trends"""
        # Get recent trends from database
        trends = self.db.get_active_trends(hours=24)
        
        if not trends:
            return {'trends': [], 'summary': {}}
        
        # Score and rank trends
        ranked_trends = self.rank_trends(trends)
        
        # Calculate potential for top trends
        for trend in ranked_trends[:10]:
            potential = self.calculate_trend_potential(trend)
            trend['potential'] = potential
        
        # Group by type
        by_type = {}
        for trend in ranked_trends:
            t_type = trend.get('trend_type', 'unknown')
            if t_type not in by_type:
                by_type[t_type] = []
            by_type[t_type].append(trend)
        
        # Calculate summary statistics
        summary = {
            'total_trends': len(ranked_trends),
            'avg_score': np.mean([t.get('combined_score', 0) for t in ranked_trends]),
            'max_score': max([t.get('combined_score', 0) for t in ranked_trends]),
            'by_type': {k: len(v) for k, v in by_type.items()},
            'top_trend_type': max(by_type.items(), key=lambda x: len(x[1]))[0] if by_type else 'none'
        }
        
        return {
            'trends': ranked_trends[:limit],
            'summary': summary,
            'generated_at': datetime.utcnow().isoformat()
        }
    
    def compare_trends(self, trend1: str, trend2: str) -> Dict[str, Any]:
        """Compare two trends"""
        # Get trend data
        trends = self.db.get_active_trends(hours=48)
        
        t1_data = None
        t2_data = None
        
        for trend in trends:
            if trend.get('name') == trend1 or trend.get('hashtag') == trend1:
                t1_data = trend
            if trend.get('name') == trend2 or trend.get('hashtag') == trend2:
                t2_data = trend
        
        if not t1_data or not t2_data:
            return {'error': 'One or both trends not found'}
        
        # Get velocity history
        from app.trend_engine.velocity_tracker import VelocityTracker
        tracker = VelocityTracker()
        
        t1_history = tracker.get_velocity_over_time(trend1, hours=24, interval_minutes=60)
        t2_history = tracker.get_velocity_over_time(trend2, hours=24, interval_minutes=60)
        
        # Calculate comparison metrics
        comparison = {
            'trend1': {
                'name': trend1,
                'current_velocity': t1_data.get('velocity', 0),
                'peak_velocity': max([v['velocity'] for v in t1_history]) if t1_history else 0,
                'total_volume': t1_data.get('volume', 0),
                'avg_engagement': t1_data.get('engagement', 0)
            },
            'trend2': {
                'name': trend2,
                'current_velocity': t2_data.get('velocity', 0),
                'peak_velocity': max([v['velocity'] for v in t2_history]) if t2_history else 0,
                'total_volume': t2_data.get('volume', 0),
                'avg_engagement': t2_data.get('engagement', 0)
            }
        }
        
        # Determine leader
        scores = self.rank_trends([t1_data, t2_data])
        if len(scores) >= 2:
            comparison['leader'] = scores[0].get('name')
            comparison['score_difference'] = scores[0]['combined_score'] - scores[1]['combined_score']
        
        return comparison

class TrendPredictor:
    """Predicts future trend trajectories"""
    
    def __init__(self):
        self.db = MongoDBClient()
        
    def predict_hashtag_trajectory(self, hashtag: str, hours_ahead: int = 24) -> Dict[str, Any]:
        """Predict hashtag trajectory"""
        from app.trend_engine.velocity_tracker import VelocityTracker
        tracker = VelocityTracker()
        
        # Get historical data
        history = tracker.get_velocity_over_time(hashtag, hours=48, interval_minutes=60)
        
        if len(history) < 6:  # Need at least 6 data points
            return {
                'hashtag': hashtag,
                'predictions': [],
                'confidence': 'low',
                'message': 'Insufficient historical data'
            }
        
        # Extract velocities
        timestamps = [h['timestamp'] for h in history]
        velocities = [h['velocity'] for h in history]
        
        # Simple prediction models
        predictions = []
        
        # Model 1: Linear trend
        x = np.arange(len(velocities))
        y = np.array(velocities)
        
        if len(x) > 1:
            z = np.polyfit(x, y, 1)
            linear_model = np.poly1d(z)
            
            # Predict future points
            for i in range(1, hours_ahead + 1):
                future_idx = len(velocities) + i - 1
                pred_velocity = linear_model(future_idx)
                predictions.append({
                    'hours_ahead': i,
                    'linear_prediction': max(0, float(pred_velocity)),
                    'timestamp': (datetime.utcnow() + timedelta(hours=i)).isoformat()
                })
        
        # Model 2: Moving average
        window = 3
        if len(velocities) >= window:
            ma = np.convolve(velocities, np.ones(window)/window, mode='valid')
            last_ma = ma[-1]
            
            for i, pred in enumerate(predictions):
                # Simple persistence of moving average
                pred['ma_prediction'] = max(0, float(last_ma))
        
        # Calculate confidence
        # Higher confidence if recent trend is stable
        recent_velocities = velocities[-6:]
        if len(recent_velocities) > 1:
            volatility = np.std(recent_velocities) / (np.mean(recent_velocities) if np.mean(recent_velocities) > 0 else 1)
            
            if volatility < 0.3:
                confidence = 'high'
            elif volatility < 0.7:
                confidence = 'medium'
            else:
                confidence = 'low'
        else:
            confidence = 'low'
        
        return {
            'hashtag': hashtag,
            'predictions': predictions,
            'confidence': confidence,
            'historical_velocities': velocities[-12:] if len(velocities) >= 12 else velocities,
            'trend_direction': 'increasing' if velocities[-1] > velocities[0] else 'decreasing' if velocities[-1] < velocities[0] else 'stable'
        }
    
    def predict_topic_lifetime(self, cluster_id: str) -> Dict[str, Any]:
        """Predict how long a topic trend will last"""
        # Get posts in this cluster
        posts = list(self.db.db[self.db.PROCESSED_COLLECTION].find(
            {"cluster_id": cluster_id},
            sort=[("timestamp", 1)]
        ))
        
        if len(posts) < 5:
            return {
                'cluster_id': cluster_id,
                'estimated_lifetime_hours': None,
                'stage': 'emerging',
                'confidence': 'low'
            }
        
        # Extract timestamps
        timestamps = []
        for post in posts:
            ts = post.get('timestamp')
            if ts:
                try:
                    timestamps.append(datetime.fromisoformat(ts.replace('Z', '+00:00')))
                except:
                    continue
        
        if len(timestamps) < 2:
            return {
                'cluster_id': cluster_id,
                'estimated_lifetime_hours': None,
                'stage': 'emerging',
                'confidence': 'low'
            }
        
        # Calculate post frequency over time
        timestamps.sort()
        time_span = (timestamps[-1] - timestamps[0]).total_seconds() / 3600
        
        if time_span == 0:
            return {
                'cluster_id': cluster_id,
                'estimated_lifetime_hours': 1,
                'stage': 'peak',
                'confidence': 'medium'
            }
        
        post_rate = len(timestamps) / time_span
        
        # Get recent activity (last 3 hours)
        now = datetime.utcnow()
        recent_cutoff = now - timedelta(hours=3)
        recent_posts = [ts for ts in timestamps if ts >= recent_cutoff]
        recent_rate = len(recent_posts) / 3 if recent_posts else 0
        
        # Determine stage
        if recent_rate > post_rate * 1.5:
            stage = 'growing'
        elif recent_rate < post_rate * 0.5:
            stage = 'declining'
        else:
            stage = 'mature'
        
        # Estimate remaining lifetime
        if recent_rate > 0:
            if stage == 'growing':
                # Assume growth will continue for a while
                remaining = time_span * 2
            elif stage == 'declining':
                # Linear decay estimate
                decay_rate = (post_rate - recent_rate) / 3  # per hour
                if decay_rate > 0:
                    remaining = recent_rate / decay_rate
                else:
                    remaining = time_span
            else:
                # Stable - assume similar remaining time
                remaining = time_span
        else:
            remaining = 0
        
        return {
            'cluster_id': cluster_id,
            'estimated_lifetime_hours': round(remaining, 1),
            'stage': stage,
            'total_posts': len(posts),
            'post_rate_per_hour': round(post_rate, 2),
            'recent_rate_per_hour': round(recent_rate, 2),
            'time_span_hours': round(time_span, 1),
            'confidence': 'medium' if remaining > 0 else 'low'
        }