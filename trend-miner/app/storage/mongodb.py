from typing import List, Dict, Any, Optional
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import ConnectionFailure, DuplicateKeyError
from datetime import datetime

from app.config import (
    MONGODB_URI, MONGODB_DB,
    RAW_COLLECTION, PROCESSED_COLLECTION,
    EMBEDDINGS_COLLECTION, TRENDS_COLLECTION
)
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class MongoDBClient:
    """MongoDB client for data storage"""
    
    def __init__(self):
        self.client = None
        self.db = None
        self.connect()
        
    def connect(self):
        """Establish connection to MongoDB"""
        try:
            self.client = MongoClient(MONGODB_URI)
            self.db = self.client[MONGODB_DB]
            
            # Test connection
            self.client.admin.command('ping')
            
            # Create indexes
            self._create_indexes()
            
            logger.info(f"Connected to MongoDB: {MONGODB_DB}")
            
        except ConnectionFailure as e:
            logger.error(f"MongoDB connection failed: {e}")
            raise
    
    def _create_indexes(self):
        """Create necessary indexes"""
        # Raw posts indexes
        self.db[RAW_COLLECTION].create_index([("scraped_at", DESCENDING)])
        self.db[RAW_COLLECTION].create_index([("platform", ASCENDING)])
        self.db[RAW_COLLECTION].create_index([("status", ASCENDING)])
        
        # Processed posts indexes
        self.db[PROCESSED_COLLECTION].create_index([("post_id", ASCENDING)], unique=True)
        self.db[PROCESSED_COLLECTION].create_index([("timestamp", DESCENDING)])
        self.db[PROCESSED_COLLECTION].create_index([("platform", ASCENDING)])
        self.db[PROCESSED_COLLECTION].create_index([("author", ASCENDING)])
        self.db[PROCESSED_COLLECTION].create_index([("emotion", ASCENDING)])
        self.db[PROCESSED_COLLECTION].create_index([("cluster_id", ASCENDING)])
        self.db[PROCESSED_COLLECTION].create_index([("engagement_score", DESCENDING)])
        self.db[PROCESSED_COLLECTION].create_index([("hashtags", ASCENDING)])
        
        # Embeddings indexes
        self.db[EMBEDDINGS_COLLECTION].create_index([("post_id", ASCENDING)], unique=True)
        self.db[EMBEDDINGS_COLLECTION].create_index([("created_at", DESCENDING)])
        
        # Trends indexes
        self.db[TRENDS_COLLECTION].create_index([("detected_at", DESCENDING)])
        self.db[TRENDS_COLLECTION].create_index([("topic", ASCENDING)])
        self.db[TRENDS_COLLECTION].create_index([("score", DESCENDING)])
        
        logger.info("Database indexes created")
    
    # Raw posts operations
    
    def insert_raw_post(self, post: Dict[str, Any]) -> str:
        """Insert a raw post"""
        try:
            result = self.db[RAW_COLLECTION].insert_one(post)
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Error inserting raw post: {e}")
            return None
    
    def get_raw_post(self, post_id: str) -> Optional[Dict[str, Any]]:
        """Get raw post by ID"""
        from bson.objectid import ObjectId
        try:
            return self.db[RAW_COLLECTION].find_one({"_id": ObjectId(post_id)})
        except Exception as e:
            logger.error(f"Error getting raw post: {e}")
            return None
    
    def get_unprocessed_raw_posts(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get raw posts that haven't been processed yet"""
        return list(self.db[RAW_COLLECTION].find(
            {"status": "raw"},
            sort=[("scraped_at", ASCENDING)],
            limit=limit
        ))
    
    def mark_raw_processed(self, raw_id: str):
        """Mark a raw post as processed"""
        from bson.objectid import ObjectId
        try:
            self.db[RAW_COLLECTION].update_one(
                {"_id": ObjectId(raw_id)},
                {"$set": {"status": "processed", "processed_at": datetime.utcnow().isoformat()}}
            )
        except Exception as e:
            logger.error(f"Error marking raw post as processed: {e}")
    
    def count_raw_posts(self) -> int:
        """Count total raw posts"""
        return self.db[RAW_COLLECTION].count_documents({})
    
    # Processed posts operations
    
    def insert_processed_post(self, post: Dict[str, Any]) -> str:
        """Insert a processed post"""
        try:
            # Ensure post_id exists
            if 'post_id' not in post:
                from app.utils.fingerprint import generate_fingerprint
                post['post_id'] = generate_fingerprint(post.get('text', ''), post.get('author', ''))
            
            # Use upsert to avoid duplicates
            result = self.db[PROCESSED_COLLECTION].update_one(
                {"post_id": post['post_id']},
                {"$set": post},
                upsert=True
            )
            return post['post_id']
        except Exception as e:
            logger.error(f"Error inserting processed post: {e}")
            return None
    
    def get_processed_post(self, post_id: str) -> Optional[Dict[str, Any]]:
        """Get processed post by ID"""
        return self.db[PROCESSED_COLLECTION].find_one({"post_id": post_id})
    
    def get_recent_posts(self, limit: int = 100, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent posts within time window"""
        from datetime import datetime, timedelta
        
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        
        return list(self.db[PROCESSED_COLLECTION].find(
            {"timestamp": {"$gte": cutoff.isoformat()}},
            sort=[("timestamp", DESCENDING)],
            limit=limit
        ))
    
    def get_posts_by_emotion(self, emotion: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get posts by emotion"""
        return list(self.db[PROCESSED_COLLECTION].find(
            {"emotion": emotion},
            sort=[("engagement_score", DESCENDING)],
            limit=limit
        ))
    
    def get_posts_by_hashtag(self, hashtag: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get posts by hashtag"""
        return list(self.db[PROCESSED_COLLECTION].find(
            {"hashtags": hashtag.lower()},
            sort=[("timestamp", DESCENDING)],
            limit=limit
        ))
    
    def update_post_cluster(self, post_id: str, cluster_id: str, method: str):
        """Update post cluster information"""
        self.db[PROCESSED_COLLECTION].update_one(
            {"post_id": post_id},
            {"$set": {
                "cluster_id": cluster_id,
                "cluster_method": method,
                "clustered_at": datetime.utcnow().isoformat()
            }}
        )
    
    def count_processed_posts(self) -> int:
        """Count total processed posts"""
        return self.db[PROCESSED_COLLECTION].count_documents({})
    
    # Embeddings operations
    
    def store_embedding(self, post_id: str, embedding: List[float]):
        """Store embedding for a post"""
        try:
            self.db[EMBEDDINGS_COLLECTION].update_one(
                {"post_id": post_id},
                {"$set": {
                    "vector": embedding,
                    "created_at": datetime.utcnow().isoformat()
                }},
                upsert=True
            )
        except Exception as e:
            logger.error(f"Error storing embedding: {e}")
    
    def get_embedding(self, post_id: str) -> Optional[List[float]]:
        """Get embedding for a post"""
        result = self.db[EMBEDDINGS_COLLECTION].find_one({"post_id": post_id})
        return result.get('vector') if result else None
    
    def get_all_embeddings(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """Get all embeddings (for clustering)"""
        return list(self.db[EMBEDDINGS_COLLECTION].find(
            {}, 
            sort=[("created_at", DESCENDING)],
            limit=limit
        ))
    
    # Trends operations
    
    def insert_trend(self, trend: Dict[str, Any]):
        """Insert a detected trend"""
        try:
            trend['detected_at'] = datetime.utcnow().isoformat()
            self.db[TRENDS_COLLECTION].insert_one(trend)
        except Exception as e:
            logger.error(f"Error inserting trend: {e}")
    
    def get_active_trends(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get trends detected in last N hours"""
        from datetime import datetime, timedelta
        
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        
        return list(self.db[TRENDS_COLLECTION].find(
            {"detected_at": {"$gte": cutoff.isoformat()}},
            sort=[("score", DESCENDING)]
        ))
    
    # Aggregation operations
    
    def get_emotion_stats(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get emotion statistics for recent posts"""
        from datetime import datetime, timedelta
        
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        
        pipeline = [
            {"$match": {"timestamp": {"$gte": cutoff.isoformat()}}},
            {"$group": {
                "_id": "$emotion",
                "count": {"$sum": 1},
                "avg_engagement": {"$avg": "$engagement_score"},
                "total_engagement": {"$sum": "$engagement_score"}
            }},
            {"$sort": {"count": -1}}
        ]
        
        return list(self.db[PROCESSED_COLLECTION].aggregate(pipeline))
    
    def get_hashtag_stats(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get hashtag statistics"""
        pipeline = [
            {"$unwind": "$hashtags"},
            {"$group": {
                "_id": "$hashtags",
                "count": {"$sum": 1},
                "avg_engagement": {"$avg": "$engagement_score"}
            }},
            {"$sort": {"count": -1}},
            {"$limit": limit}
        ]
        
        return list(self.db[PROCESSED_COLLECTION].aggregate(pipeline))
    
    def get_author_stats(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get author statistics"""
        pipeline = [
            {"$group": {
                "_id": "$author",
                "post_count": {"$sum": 1},
                "avg_engagement": {"$avg": "$engagement_score"},
                "total_engagement": {"$sum": "$engagement_score"},
                "avg_likes": {"$avg": "$likes"},
                "avg_replies": {"$avg": "$replies"}
            }},
            {"$match": {"post_count": {"$gt": 1}}},
            {"$sort": {"total_engagement": -1}},
            {"$limit": limit}
        ]
        
        return list(self.db[PROCESSED_COLLECTION].aggregate(pipeline))
    
    def close(self):
        """Close database connection"""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")