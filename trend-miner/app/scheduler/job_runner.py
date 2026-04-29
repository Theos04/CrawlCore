import schedule
import time
import threading
from datetime import datetime
from typing import Dict, Any, List, Callable
import signal
import sys

from app.utils.logger import setup_logger
from app.config import SCRAPE_INTERVAL_MINUTES
from app.scraper.threads_scraper import ThreadsScraper
from app.processing.pipeline import ProcessingPipeline
from app.processing.embedding_generator import EmbeddingGenerator
from app.trend_engine.spike_detector import SpikeDetector
from app.trend_engine.scoring import TrendScorer, TrendPredictor
from app.storage.mongodb import MongoDBClient
from app.utils.helpers import export_to_jsonl

logger = setup_logger(__name__)

class JobRunner:
    """Manages scheduled jobs"""
    
    def __init__(self):
        self.running = False
        self.thread = None
        self.jobs = []
        self.db = MongoDBClient()
        
        # Initialize components
        self.scraper = ThreadsScraper()
        self.pipeline = ProcessingPipeline()
        self.embedding_gen = EmbeddingGenerator()
        self.spike_detector = SpikeDetector()
        self.trend_scorer = TrendScorer()
        self.trend_predictor = TrendPredictor()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
    def setup_jobs(self):
        """Setup scheduled jobs"""
        # Clear existing jobs
        schedule.clear()
        
        # Scraping job
        schedule.every(SCRAPE_INTERVAL_MINUTES).minutes.do(self.job_scrape)
        
        # Processing job (run shortly after scraping)
        schedule.every(SCRAPE_INTERVAL_MINUTES).minutes.do(self.job_process).delay(2)
        
        # Embedding generation (every 30 minutes)
        schedule.every(30).minutes.do(self.job_generate_embeddings)
        
        # Trend detection (every hour)
        schedule.every(1).hour.do(self.job_detect_trends)
        
        # Dataset export (every 6 hours)
        schedule.every(6).hours.do(self.job_export_dataset)
        
        # Cleanup old data (daily)
        schedule.every().day.at("03:00").do(self.job_cleanup)
        
        # Log scheduled jobs
        self.jobs = schedule.get_jobs()
        logger.info(f"Setup {len(self.jobs)} scheduled jobs")
        for job in self.jobs:
            logger.debug(f"Job: {job}")
    
    def job_scrape(self):
        """Scraping job"""
        logger.info("Starting scheduled scrape")
        try:
            posts = self.scraper.scrape(max_posts=50)
            logger.info(f"Scraped {len(posts)} posts")
            return len(posts)
        except Exception as e:
            logger.error(f"Scraping job failed: {e}", exc_info=True)
            return 0
    
    def job_process(self):
        """Processing job"""
        logger.info("Starting scheduled processing")
        try:
            # Get unprocessed raw posts
            raw_posts = self.db.get_unprocessed_raw_posts(limit=100)
            
            if not raw_posts:
                logger.info("No unprocessed posts found")
                return 0
            
            # Process posts
            processed = self.pipeline.process_batch(raw_posts)
            
            # Mark as processed
            for raw_post in raw_posts:
                self.db.mark_raw_processed(raw_post.get('_id'))
            
            logger.info(f"Processed {len(processed)} posts")
            return len(processed)
        except Exception as e:
            logger.error(f"Processing job failed: {e}", exc_info=True)
            return 0
    
    def job_generate_embeddings(self):
        """Embedding generation job"""
        logger.info("Starting scheduled embedding generation")
        try:
            # Get recent processed posts without embeddings
            recent_posts = self.db.get_recent_posts(limit=200)
            
            # Filter posts that don't have embeddings yet
            posts_to_process = []
            for post in recent_posts:
                post_id = post.get('post_id')
                embedding = self.db.get_embedding(post_id)
                if not embedding:
                    posts_to_process.append(post)
            
            if not posts_to_process:
                logger.info("No new posts to generate embeddings for")
                return 0
            
            # Generate embeddings
            self.embedding_gen.generate_and_store(posts_to_process)
            
            logger.info(f"Generated embeddings for {len(posts_to_process)} posts")
            return len(posts_to_process)
        except Exception as e:
            logger.error(f"Embedding generation job failed: {e}", exc_info=True)
            return 0
    
    def job_detect_trends(self):
        """Trend detection job"""
        logger.info("Starting scheduled trend detection")
        try:
            # Detect spikes
            spikes = self.spike_detector.detect_all_spikes()
            
            # Get trends
            trends = self.spike_detector.detect_trends()
            
            # Score and rank
            ranked_trends = self.trend_scorer.rank_trends(trends)
            
            # Predict for top trends
            for trend in ranked_trends[:5]:
                if trend.get('trend_type') == 'hashtag':
                    prediction = self.trend_predictor.predict_hashtag_trajectory(
                        trend.get('name')
                    )
                    trend['prediction'] = prediction
            
            logger.info(f"Detected {len(trends)} trends")
            return len(trends)
        except Exception as e:
            logger.error(f"Trend detection job failed: {e}", exc_info=True)
            return 0
    
    def job_export_dataset(self):
        """Dataset export job"""
        logger.info("Starting scheduled dataset export")
        try:
            # Get recent posts
            posts = self.db.get_recent_posts(limit=1000, hours=48)
            
            if not posts:
                logger.info("No posts to export")
                return 0
            
            # Prepare dataset
            dataset = []
            for post in posts:
                dataset.append({
                    'text': post.get('text', ''),
                    'cleaned_text': post.get('cleaned_text', ''),
                    'emotion': post.get('primary_emotion', 'neutral'),
                    'sentiment': post.get('sentiment', 'neutral'),
                    'engagement_score': post.get('engagement_score', 0),
                    'hashtags': post.get('hashtags', []),
                    'word_count': post.get('word_count', 0),
                    'timestamp': post.get('timestamp', '')
                })
            
            # Export
            filename = f"trend_dataset_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.jsonl"
            export_to_jsonl(dataset, filename)
            
            logger.info(f"Exported {len(dataset)} posts to {filename}")
            return len(dataset)
        except Exception as e:
            logger.error(f"Dataset export job failed: {e}", exc_info=True)
            return 0
    
    def job_cleanup(self):
        """Cleanup old data"""
        logger.info("Starting scheduled cleanup")
        try:
            # Delete raw posts older than 7 days that have been processed
            from datetime import datetime, timedelta
            cutoff = datetime.utcnow() - timedelta(days=7)
            
            result = self.db.db[self.db.RAW_COLLECTION].delete_many({
                "status": "processed",
                "scraped_at": {"$lt": cutoff.isoformat()}
            })
            
            logger.info(f"Cleaned up {result.deleted_count} old raw posts")
            return result.deleted_count
        except Exception as e:
            logger.error(f"Cleanup job failed: {e}", exc_info=True)
            return 0
    
    def run_job_once(self, job_name: str):
        """Run a specific job once"""
        jobs = {
            'scrape': self.job_scrape,
            'process': self.job_process,
            'embeddings': self.job_generate_embeddings,
            'trends': self.job_detect_trends,
            'export': self.job_export_dataset,
            'cleanup': self.job_cleanup
        }
        
        if job_name in jobs:
            logger.info(f"Running job: {job_name}")
            return jobs[job_name]()
        else:
            logger.error(f"Unknown job: {job_name}")
            return None
    
    def run_continuously(self):
        """Run scheduler continuously"""
        self.setup_jobs()
        self.running = True
        
        logger.info("Scheduler started")
        
        while self.running:
            try:
                schedule.run_pending()
                time.sleep(1)
            except Exception as e:
                logger.error(f"Scheduler error: {e}", exc_info=True)
                time.sleep(60)  # Wait a minute before retrying
        
        logger.info("Scheduler stopped")
    
    def start(self):
        """Start the scheduler in a background thread"""
        if self.thread and self.thread.is_alive():
            logger.warning("Scheduler already running")
            return
        
        self.thread = threading.Thread(target=self.run_continuously)
        self.thread.daemon = True
        self.thread.start()
        
        logger.info("Scheduler started in background thread")
    
    def stop(self):
        """Stop the scheduler"""
        logger.info("Stopping scheduler...")
        self.running = False
        
        if self.thread:
            self.thread.join(timeout=10)
        
        # Close database connection
        self.db.close()
        
        logger.info("Scheduler stopped")
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}")
        self.stop()
        sys.exit(0)
    
    def get_job_status(self) -> List[Dict[str, Any]]:
        """Get status of all jobs"""
        status = []
        for job in self.jobs:
            status.append({
                'job': str(job),
                'next_run': job.next_run.isoformat() if job.next_run else None,
                'last_run': job.last_run.isoformat() if job.last_run else None,
                'period': job.period
            })
        return status