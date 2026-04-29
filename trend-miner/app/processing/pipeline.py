from typing import List, Dict, Any, Optional
from datetime import datetime

from app.parser.post_parser import PostParser
from app.processing.text_cleaner import TextCleaner
from app.processing.hashtag_extractor import HashtagExtractor
from app.processing.emotion_classifier import EmotionClassifier
from app.processing.embedding_generator import EmbeddingGenerator
from app.processing.topic_cluster import TopicCluster
from app.storage.mongodb import MongoDBClient
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class ProcessingPipeline:
    """Main processing pipeline for posts"""
    
    def __init__(self):
        self.parser = PostParser()
        self.text_cleaner = TextCleaner()
        self.hashtag_extractor = HashtagExtractor()
        self.emotion_classifier = EmotionClassifier()
        self.embedding_generator = EmbeddingGenerator()
        self.topic_cluster = TopicCluster()
        self.db = MongoDBClient()
        
    def process(self, raw_post: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single raw post through the entire pipeline"""
        try:
            # Step 1: Parse raw HTML
            parsed = self.parser.parse(raw_post)
            if not parsed:
                return None
            
            # Step 2: Clean text
            cleaned_versions = self.text_cleaner.extract_clean_version(parsed)
            parsed.update(cleaned_versions)
            
            # Step 3: Extract components
            components = self.text_cleaner.extract_components(parsed.get('text', ''))
            parsed.update(components)
            
            # Step 4: Extract hashtags specifically
            hashtags = self.hashtag_extractor.extract(parsed.get('text', ''))
            parsed['hashtags'] = hashtags
            
            # Step 5: Classify emotion
            emotion_features = self.emotion_classifier.extract_emotion_features(
                parsed.get('text', '')
            )
            parsed.update(emotion_features)
            
            # Step 6: Calculate readability
            readability = self.text_cleaner.calculate_readability(
                parsed.get('text', '')
            )
            parsed.update(readability)
            
            # Step 7: Calculate engagement score
            parsed['engagement_score'] = self._calculate_engagement_score(parsed)
            
            # Step 8: Add processing metadata
            parsed['processed_at'] = datetime.utcnow().isoformat()
            parsed['processing_version'] = '1.0'
            
            # Step 9: Store processed post
            self.db.insert_processed_post(parsed)
            
            logger.debug(f"Processed post: {parsed.get('post_id')}")
            return parsed
            
        except Exception as e:
            logger.error(f"Error processing post: {e}", exc_info=True)
            return None
    
    def process_batch(self, raw_posts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process multiple raw posts"""
        processed_posts = []
        
        for raw_post in raw_posts:
            processed = self.process(raw_post)
            if processed:
                processed_posts.append(processed)
        
        # Step 10: Topic clustering (requires all posts)
        if len(processed_posts) >= 5:  # Minimum for clustering
            processed_posts = self.topic_cluster.cluster_by_embeddings(processed_posts)
            
            # Update cluster info in database
            for post in processed_posts:
                self.db.update_post_cluster(
                    post.get('post_id'),
                    post.get('cluster_id'),
                    post.get('cluster_method')
                )
        
        logger.info(f"Processed {len(processed_posts)} posts from {len(raw_posts)} raw posts")
        return processed_posts
    
    def _calculate_engagement_score(self, post: Dict[str, Any]) -> float:
        """Calculate normalized engagement score"""
        likes = post.get('likes', 0)
        replies = post.get('replies', 0)
        
        # Simple engagement score
        if likes == 0 and replies == 0:
            return 0.0
        
        # Normalize (log scale to handle large numbers)
        import math
        score = math.log1p(likes + replies * 2)  # Replies weighted more
        
        # Normalize to 0-1 range (assuming max ~100K engagement)
        max_expected = math.log1p(100000)
        normalized = min(score / max_expected, 1.0)
        
        return round(normalized, 3)
    
    def reprocess_existing(self, limit: int = 100):
        """Reprocess existing raw posts"""
        raw_posts = self.db.get_unprocessed_raw_posts(limit)
        
        if not raw_posts:
            logger.info("No unprocessed posts found")
            return
        
        logger.info(f"Reprocessing {len(raw_posts)} posts")
        processed = self.process_batch(raw_posts)
        
        # Mark raw posts as processed
        for raw_post in raw_posts:
            self.db.mark_raw_processed(raw_post.get('_id'))
        
        return processed
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get statistics about the processing pipeline"""
        raw_count = self.db.count_raw_posts()
        processed_count = self.db.count_processed_posts()
        
        return {
            'raw_posts': raw_count,
            'processed_posts': processed_count,
            'processing_rate': processed_count / raw_count if raw_count > 0 else 0,
            'pending_processing': raw_count - processed_count
        }