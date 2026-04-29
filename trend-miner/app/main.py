import argparse
import sys
from datetime import datetime
from typing import Dict, Any

from app.utils.logger import setup_logger
from app.scheduler.job_runner import JobRunner
from app.config import CURRENT_TIMESTAMP

logger = setup_logger(__name__)

class TrendMiner:
    """Main application orchestrator"""
    
    def __init__(self):
        self.job_runner = JobRunner()
        self.start_time = CURRENT_TIMESTAMP
        
    def run_once(self, platform: str = "threads", max_posts: int = None):
        """Run a single scraping and processing cycle"""
        logger.info(f"Running one-time mining cycle for {platform}")
        
        # Run scraping
        from app.scraper.threads_scraper import ThreadsScraper
        scraper = ThreadsScraper()
        raw_posts = scraper.scrape(max_posts=max_posts)
        
        if not raw_posts:
            logger.warning("No posts scraped")
            return
        
        # Run processing pipeline
        from app.processing.pipeline import ProcessingPipeline
        pipeline = ProcessingPipeline()
        processed_posts = pipeline.process_batch(raw_posts)
        
        # Generate embeddings
        from app.processing.embedding_generator import EmbeddingGenerator
        emb_gen = EmbeddingGenerator()
        emb_gen.generate_and_store(processed_posts)
        
        # Detect trends
        from app.trend_engine.spike_detector import SpikeDetector
        detector = SpikeDetector()
        trends = detector.detect_trends()
        
        logger.info(f"Completed cycle: {len(processed_posts)} posts processed, {len(trends)} trends detected")
        
    def start_scheduler(self):
        """Start the scheduled job runner"""
        logger.info("Starting trend miner scheduler")
        self.job_runner.start()
        
    def stop(self):
        """Stop all services gracefully"""
        logger.info("Stopping trend miner")
        self.job_runner.stop()

def main():
    parser = argparse.ArgumentParser(description="Trend Mining & AI Content Radar System")
    parser.add_argument("--mode", choices=["once", "scheduler"], default="once",
                       help="Run mode: once (single cycle) or scheduler (continuous)")
    parser.add_argument("--platform", default="threads",
                       help="Platform to scrape (threads, twitter, etc.)")
    parser.add_argument("--max-posts", type=int, help="Maximum posts to scrape")
    
    args = parser.parse_args()
    
    miner = TrendMiner()
    
    try:
        if args.mode == "once":
            miner.run_once(platform=args.platform, max_posts=args.max_posts)
        else:
            miner.start_scheduler()
            
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
        miner.stop()
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()