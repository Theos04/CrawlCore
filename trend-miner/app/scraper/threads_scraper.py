from typing import List, Dict, Any, Optional
import time
import hashlib
import re
import logging
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium.webdriver.chrome.service import Service

# -------------------------
# Configuration
# -------------------------
class Config:
    CHROME_BINARY_PATH = r"D:\Win_1130649_chrome-win\chrome-win\chrome.exe"
    CHROME_DRIVER_PATH = r"D:\Win_1130649_chromedriver_win32\chromedriver_win32\chromedriver.exe"
    HEADLESS_BROWSER = False
    MAX_SCROLLS = 20
    SCROLL_DELAY = 2
    # For now, run Chrome with an isolated Selenium profile for maximum stability.
    # If you want to attach to an existing browser, set CHROMIUM_DEBUG_PORT and
    # handle the browser startup yourself.
    CHROMIUM_DEBUG_PORT = 9250  # e.g., 9250 if using remote debugging
    CHROMIUM_USER_DATA_DIR = None
    THREADS_USERNAME = ""  # Optional – set if you want in-script login
    THREADS_PASSWORD = ""


# -------------------------
# Logger
# -------------------------
logger = logging.getLogger("ThreadsScraper")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(ch)


# -------------------------
# Threads Scraper
# -------------------------
class ThreadsScraper:
    """Scraper for Threads.net using Selenium"""
    
    def __init__(self):
        self.base_url = "https://www.threads.com"
        self.driver: Optional[webdriver.Chrome] = None
        self.scroll_handler = None
        self.setup_driver()

    def scrape(self, max_posts: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        High-level scrape wrapper for compatibility with the rest of the app.
        Attempts login (if configured) and then scrapes the main feed.
        """
        try:
            # Safe to call even when using an attached Chromium session:
            # - debug-port mode: login() is a no-op
            # - no credentials: login() is a no-op
            self.login()
        except Exception as e:
            logger.error(f"Login step failed before scraping: {e}", exc_info=True)

        return self.scrape_feed(limit=max_posts)
        
    def setup_driver(self):
        """Configure Selenium WebDriver and optionally attach to existing Chromium"""
        options = webdriver.ChromeOptions()
        
        # Remote debugging (attach to existing browser)
        if Config.CHROMIUM_DEBUG_PORT:
            options.add_experimental_option(
                "debuggerAddress", f"127.0.0.1:{Config.CHROMIUM_DEBUG_PORT}"
            )
            service = Service(Config.CHROME_DRIVER_PATH)
            self.driver = webdriver.Chrome(service=service, options=options)
            logger.info(f"Attached to existing Chromium at port {Config.CHROMIUM_DEBUG_PORT}")
            return
        
        # Binary and user data
        if Config.CHROME_BINARY_PATH:
            options.binary_location = Config.CHROME_BINARY_PATH
        if Config.CHROMIUM_USER_DATA_DIR:
            # IMPORTANT: no other Chrome/Chromium instance can be using this profile
            options.add_argument(f"--user-data-dir={Config.CHROMIUM_USER_DATA_DIR}")
        
        if Config.HEADLESS_BROWSER:
            options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--window-size=1920,1080")
        options.add_argument("--disable-blink-features=AutomationControlled")
        # Required on newer Chrome builds to allow Selenium-controlled connections
        options.add_argument("--remote-allow-origins=*")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option("useAutomationExtension", False)
        
        # Start driver
        service = Service(Config.CHROME_DRIVER_PATH)
        self.driver = webdriver.Chrome(service=service, options=options)
        # Anti-detection JS
        self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        logger.info("ChromeDriver started successfully")
    
    # -------------------------
    # Login
    # -------------------------
    def login(self) -> bool:
        """Login to Threads if credentials are provided"""
        if Config.CHROMIUM_DEBUG_PORT:
            logger.info("Using existing Chromium session; already logged in")
            return True
        if not Config.THREADS_USERNAME or not Config.THREADS_PASSWORD:
            logger.info("No credentials provided; skipping login")
            return False
        
        try:
            self.driver.get(f"{self.base_url}/login")
            time.sleep(50)
            
            username_field = WebDriverWait(self.driver, 70).until(
                EC.presence_of_element_located((By.NAME, "username"))
            )
            username_field.send_keys(Config.THREADS_USERNAME)
            
            password_field = self.driver.find_element(By.NAME, "password")
            password_field.send_keys(Config.THREADS_PASSWORD)
            
            login_button = self.driver.find_element(By.XPATH, "//button[@type='submit']")
            login_button.click()
            time.sleep(3)
            
            logger.info("Login successful")
            return True
        except Exception as e:
            logger.error(f"Login failed: {e}")
            return False
    
    # -------------------------
    # Feed Scraping
    # -------------------------
    def scrape_feed(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Scrape posts from the main feed"""
        posts = []
        try:
            self.driver.get(f"{self.base_url}/")
            time.sleep(3)
            
            scroll_count = 0
            max_scrolls = limit // 5 if limit else Config.MAX_SCROLLS
            
            while scroll_count < max_scrolls:
                post_elements = self.driver.find_elements(By.XPATH, "//article")
                
                for element in post_elements[len(posts):]:
                    post_data = self.extract_post_data(element)
                    if post_data and post_data.get("text"):
                        posts.append(post_data)
                        if limit and len(posts) >= limit:
                            logger.info(f"Reached limit of {limit} posts")
                            return posts
                
                # Scroll down
                self.driver.execute_script("window.scrollBy(0, window.innerHeight);")
                time.sleep(Config.SCROLL_DELAY)
                scroll_count += 1
        
        except Exception as e:
            logger.error(f"Error scraping feed: {e}", exc_info=True)
        
        logger.info(f"Scraped {len(posts)} posts")
        return posts
    
    # -------------------------
    # Post Data Extraction
    # -------------------------
    def extract_post_data(self, post_element) -> Dict[str, Any]:
        try:
            # Author
            author = ""
            try:
                author_elem = post_element.find_element(By.XPATH, ".//span[contains(@class,'author')]")
                author = author_elem.text
            except NoSuchElementException:
                pass
            
            # Text
            text = ""
            try:
                text_elem = post_element.find_element(By.XPATH, ".//div[@dir='auto']")
                text = text_elem.text
            except NoSuchElementException:
                pass
            
            # Images
            images = []
            try:
                img_elements = post_element.find_elements(By.XPATH, ".//img")
                for img in img_elements:
                    src = img.get_attribute("src")
                    if src and "profile" not in src.lower():
                        images.append(src)
            except:
                pass
            
            # Engagement metrics (simplified)
            likes, replies = 0, 0
            try:
                metrics = post_element.text.split("\n")
                for line in metrics:
                    if "like" in line.lower() or "❤" in line:
                        likes = self.extract_number(line)
                    elif "reply" in line.lower() or "💬" in line:
                        replies = self.extract_number(line)
            except:
                pass
            
            # Timestamp
            timestamp = ""
            try:
                time_elem = post_element.find_element(By.XPATH, ".//time")
                timestamp = time_elem.get_attribute("datetime")
            except:
                pass
            
            return {
                "post_id": self.generate_post_id(text, author),
                "author": author,
                "text": text,
                "images": images,
                "likes": likes,
                "replies": replies,
                "timestamp": timestamp,
                "raw_html": post_element.get_attribute("outerHTML"),
            }
        except Exception as e:
            logger.error(f"extract_post_data error: {e}")
            return {}
    
    # -------------------------
    # Helpers
    # -------------------------
    @staticmethod
    def extract_number(text: str) -> int:
        try:
            match = re.search(r"([\d.]+)([KM])?", text)
            if match:
                num = float(match.group(1))
                suffix = match.group(2)
                if suffix == "K":
                    num *= 1000
                elif suffix == "M":
                    num *= 1_000_000
                return int(num)
        except:
            pass
        return 0
    
    @staticmethod
    def generate_post_id(text: str, author: str) -> str:
        unique_string = f"{author}{text}{time.time()}"
        return hashlib.md5(unique_string.encode()).hexdigest()
    
    def close(self):
        if self.driver:
            self.driver.quit()
            logger.info("ChromeDriver closed successfully")


# -------------------------
# Usage Example
# -------------------------
if __name__ == "__main__":
    scraper = ThreadsScraper()
    scraper.login()
    posts = scraper.scrape_feed(limit=10)
    for i, p in enumerate(posts, 1):
        logger.info(f"Post {i}: {p['author']} - {p['text'][:50]}...")
    scraper.close()
