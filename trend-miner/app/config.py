import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# MongoDB Configuration
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
MONGODB_DB = os.getenv("MONGODB_DB", "trend_miner")

# Collections
RAW_COLLECTION = "raw_posts"
PROCESSED_COLLECTION = "processed_posts"
EMBEDDINGS_COLLECTION = "embeddings"
TRENDS_COLLECTION = "trends"

# Scraper Settings
SCRAPE_INTERVAL_MINUTES = int(os.getenv("SCRAPE_INTERVAL_MINUTES", 15))
MAX_SCROLLS = int(os.getenv("MAX_SCROLLS", 10))
SCROLL_DELAY = int(os.getenv("SCROLL_DELAY", 2))
HEADLESS_BROWSER = os.getenv("HEADLESS_BROWSER", "False").lower() == "true"

# Chromium: attach to existing browser (start it with --remote-debugging-port=9250)
# or leave unset to let Selenium start a new Chrome/Chromium.
CHROMIUM_DEBUG_PORT = os.getenv("CHROMIUM_DEBUG_PORT", "").strip()
CHROMIUM_DEBUG_PORT = int(CHROMIUM_DEBUG_PORT) if CHROMIUM_DEBUG_PORT else None
# Optional: use your Chromium executable and profile when not using debug port
CHROMIUM_EXECUTABLE = os.getenv("CHROMIUM_EXECUTABLE", "").strip() or None
CHROMIUM_USER_DATA_DIR = os.getenv("CHROMIUM_USER_DATA_DIR", "").strip() or None

# Threads Credentials (if needed)
THREADS_USERNAME = os.getenv("THREADS_USERNAME", "")
THREADS_PASSWORD = os.getenv("THREADS_PASSWORD", "")

# AI/ML Settings
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))
EMOTION_THRESHOLD = float(os.getenv("EMOTION_THRESHOLD", 0.5))

# Trend Detection
VELOCITY_WINDOW_MINUTES = int(os.getenv("VELOCITY_WINDOW_MINUTES", 60))
SPIKE_THRESHOLD = float(os.getenv("SPIKE_THRESHOLD", 2.5))
MIN_CLUSTER_SIZE = int(os.getenv("MIN_CLUSTER_SIZE", 5))

# Paths
DATA_DIR = os.getenv("DATA_DIR", "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
CLEANED_DATA_DIR = os.path.join(DATA_DIR, "cleaned")
EXPORTS_DIR = os.path.join(DATA_DIR, "exports")
MODELS_DIR = os.getenv("MODELS_DIR", "models")

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", "trend_miner.log")

# Ensure directories exist
for dir_path in [RAW_DATA_DIR, CLEANED_DATA_DIR, EXPORTS_DIR, MODELS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Current timestamp
CURRENT_TIMESTAMP = datetime.utcnow().isoformat()