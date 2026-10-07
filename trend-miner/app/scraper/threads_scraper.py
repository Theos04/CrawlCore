from typing import List, Dict, Any, Optional
import os
import time
import hashlib
import re
import logging
import tempfile
import shutil

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException,
    NoSuchElementException,
    SessionNotCreatedException,
)
from selenium.webdriver.chrome.service import Service


# -------------------------
# Configuration
# -------------------------
class Config:
    # --- Linux defaults ---
    CHROME_BINARY_PATH = "/usr/bin/google-chrome-stable"

    # Leave as None to use Selenium Manager (recommended).
    CHROME_DRIVER_PATH = None

    HEADLESS_BROWSER = False

    # --- Scroll / wait tuning ---
    MAX_SCROLLS = 30            # was 20 — more headroom for slow feeds
    SCROLL_DELAY = 3            # was 2 — give each scroll more render time
    POST_LOAD_WAIT = 5          # NEW — initial page render wait (was hard-coded 3)
    NO_NEW_POSTS_LIMIT = 5      # NEW — stop after N scrolls with no new posts

    # --- Selenium waits ---
    PAGE_LOAD_TIMEOUT = 60      # seconds
    IMPLICIT_WAIT = 10          # seconds
    LOGIN_ELEMENT_TIMEOUT = 60  # seconds (was 30)

    # --- Attach mode ---
    # Set to 9222 to attach to the Chrome you started with
    #   /usr/bin/google-chrome-stable --remote-debugging-port=9222 ...
    # Leave as None to let the scraper launch its own Chrome.
    CHROMIUM_DEBUG_PORT = 9222

    # Marker dir — only used for reference / non-attach runs
    CHROMIUM_USER_DATA_DIR = os.path.expanduser(
        "~/CrawlCore/trend-miner/chrome-profile"
    )

    THREADS_USERNAME = ""
    THREADS_PASSWORD = ""


# -------------------------
# Logger
# -------------------------
logger = logging.getLogger("ThreadsScraper")
if not logger.handlers:
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
        self._temp_profile_dir = None
        self.setup_driver()

    # ------------------------------------------------------------------
    # Driver setup
    # ------------------------------------------------------------------
    def setup_driver(self):
        """
        Configure Selenium WebDriver.

        Two modes:
          * Attach mode  — if CHROMIUM_DEBUG_PORT is set, connect to an
                           already-running Chromium on that debug port.
          * Launch mode  — otherwise, launch a fresh Chromium with a
                           unique temporary profile.
        """
        options = webdriver.ChromeOptions()

        # ---------------- Attach mode ---------------- #
        if Config.CHROMIUM_DEBUG_PORT:
            options.add_experimental_option(
                "debuggerAddress", f"127.0.0.1:{Config.CHROMIUM_DEBUG_PORT}"
            )
            service = Service(
                Config.CHROME_DRIVER_PATH if Config.CHROME_DRIVER_PATH else None,
                log_path="/tmp/chromedriver.log",
            )
            service.service_args = ["--verbose"]
            try:
                self.driver = webdriver.Chrome(service=service, options=options)
                self.driver.set_page_load_timeout(Config.PAGE_LOAD_TIMEOUT)
                self.driver.implicitly_wait(Config.IMPLICIT_WAIT)
                logger.info(
                    f"Attached to existing Chromium at port {Config.CHROMIUM_DEBUG_PORT}"
                )
            except Exception as e:
                logger.error(f"Failed to attach to Chromium: {e}")
                raise
            return

        # ---------------- Launch mode ---------------- #

        # 1. Binary location
        if Config.CHROME_BINARY_PATH:
            if os.path.exists(Config.CHROME_BINARY_PATH):
                options.binary_location = Config.CHROME_BINARY_PATH
                logger.info(f"Using Chromium binary: {Config.CHROME_BINARY_PATH}")
            else:
                logger.warning(
                    f"CHROME_BINARY_PATH does not exist: {Config.CHROME_BINARY_PATH}"
                )

        # 2. Profile — always use a unique temporary directory.
        self._temp_profile_dir = tempfile.mkdtemp(prefix="chrome-profile-")
        options.add_argument(f"--user-data-dir={self._temp_profile_dir}")
        logger.info(f"Using temporary Chromium profile: {self._temp_profile_dir}")

        # 3. Headless / GPU
        if Config.HEADLESS_BROWSER:
            options.add_argument("--headless=new")
            options.add_argument("--disable-gpu")

        # 4. Stability flags for Linux
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--window-size=1920,1080")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--remote-allow-origins=*")
        # NOTE: do NOT set --remote-debugging-port here.

        # 5. Exclude automation markers
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option("useAutomationExtension", False)

        # 6. Service with verbose logging
        service = Service(
            Config.CHROME_DRIVER_PATH if Config.CHROME_DRIVER_PATH else None,
            log_path="/tmp/chromedriver.log",
        )
        service.service_args = ["--verbose"]

        if Config.CHROME_DRIVER_PATH:
            logger.info(f"Using chromedriver at: {Config.CHROME_DRIVER_PATH}")
        else:
            logger.info("CHROME_DRIVER_PATH not set — using Selenium Manager")

        # 7. Launch
        try:
            self.driver = webdriver.Chrome(service=service, options=options)
            self.driver.set_page_load_timeout(Config.PAGE_LOAD_TIMEOUT)
            self.driver.implicitly_wait(Config.IMPLICIT_WAIT)
        except SessionNotCreatedException as e:
            logger.error(
                "Chromium failed to start. Common causes on Ubuntu:\n"
                "  * profile already in use by another Chrome process\n"
                "  * chromedriver/Chrome version mismatch\n"
                "  * missing system libraries\n"
                f"Underlying error: {e}\n"
                "See /tmp/chromedriver.log for verbose details."
            )
            raise

        # 8. Anti-detection JS
        try:
            self.driver.execute_script(
                "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
            )
        except Exception:
            pass

        logger.info("ChromeDriver started successfully")

    # ------------------------------------------------------------------
    # Scrape wrapper
    # ------------------------------------------------------------------
    def scrape(self, max_posts: Optional[int] = None) -> List[Dict[str, Any]]:
        try:
            self.login()
        except Exception as e:
            logger.error(f"Login step failed before scraping: {e}", exc_info=True)

        return self.scrape_feed(limit=max_posts)

    # ------------------------------------------------------------------
    # Login
    # ------------------------------------------------------------------
    def login(self) -> bool:
        if Config.CHROMIUM_DEBUG_PORT:
            logger.info("Using existing Chromium session; already logged in")
            return True
        if not Config.THREADS_USERNAME or not Config.THREADS_PASSWORD:
            logger.info("No credentials provided; skipping login")
            return False

        try:
            self.driver.get(f"{self.base_url}/login")
            time.sleep(3)

            username_field = WebDriverWait(self.driver, Config.LOGIN_ELEMENT_TIMEOUT).until(
                EC.presence_of_element_located((By.NAME, "username"))
            )
            username_field.send_keys(Config.THREADS_USERNAME)

            password_field = self.driver.find_element(By.NAME, "password")
            password_field.send_keys(Config.THREADS_PASSWORD)

            login_button = self.driver.find_element(
                By.XPATH, "//button[@type='submit']"
            )
            login_button.click()
            time.sleep(3)

            logger.info("Login successful")
            return True
        except Exception as e:
            logger.error(f"Login failed: {e}")
            return False

    # ------------------------------------------------------------------
    # Feed scraping
    # ------------------------------------------------------------------
    def scrape_feed(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        posts: List[Dict[str, Any]] = []
        try:
            self.driver.get(f"{self.base_url}/")
            time.sleep(Config.POST_LOAD_WAIT)

            scroll_count = 0
            no_new_count = 0
            max_scrolls = (limit // 5) if limit else Config.MAX_SCROLLS

            while scroll_count < max_scrolls:
                before = len(posts)
                post_elements = self.driver.find_elements(By.XPATH, "//article")

                for element in post_elements[len(posts):]:
                    post_data = self.extract_post_data(element)
                    if post_data and post_data.get("text"):
                        posts.append(post_data)
                        if limit and len(posts) >= limit:
                            logger.info(f"Reached limit of {limit} posts")
                            return posts

                self.driver.execute_script(
                    "window.scrollBy(0, window.innerHeight);"
                )
                time.sleep(Config.SCROLL_DELAY)
                scroll_count += 1

                if len(posts) == before:
                    no_new_count += 1
                    if no_new_count >= Config.NO_NEW_POSTS_LIMIT:
                        logger.info(
                            f"No new posts after {no_new_count} scrolls; stopping."
                        )
                        break
                else:
                    no_new_count = 0

        except Exception as e:
            logger.error(f"Error scraping feed: {e}", exc_info=True)

        logger.info(f"Scraped {len(posts)} posts")
        return posts

    # ------------------------------------------------------------------
    # Post extraction
    # ------------------------------------------------------------------
    def extract_post_data(self, post_element) -> Dict[str, Any]:
        try:
            author = ""
            try:
                author_elem = post_element.find_element(
                    By.XPATH, ".//span[contains(@class,'author')]"
                )
                author = author_elem.text
            except NoSuchElementException:
                pass

            text = ""
            try:
                text_elem = post_element.find_element(By.XPATH, ".//div[@dir='auto']")
                text = text_elem.text
            except NoSuchElementException:
                pass

            images = []
            try:
                img_elements = post_element.find_elements(By.XPATH, ".//img")
                for img in img_elements:
                    src = img.get_attribute("src")
                    if src and "profile" not in src.lower():
                        images.append(src)
            except Exception:
                pass

            likes, replies = 0, 0
            try:
                metrics = post_element.text.split("\n")
                for line in metrics:
                    if "like" in line.lower() or "❤" in line:
                        likes = self.extract_number(line)
                    elif "reply" in line.lower() or "💬" in line:
                        replies = self.extract_number(line)
            except Exception:
                pass

            timestamp = ""
            try:
                time_elem = post_element.find_element(By.XPATH, ".//time")
                timestamp = time_elem.get_attribute("datetime")
            except Exception:
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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
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
        except Exception:
            pass
        return 0

    @staticmethod
    def generate_post_id(text: str, author: str) -> str:
        unique_string = f"{author}{text}{time.time()}"
        return hashlib.md5(unique_string.encode()).hexdigest()

    def close(self):
        # In attach mode we must NOT quit the browser — that would kill
        # the manually-launched Chrome that has your logged-in session.
        if Config.CHROMIUM_DEBUG_PORT:
            logger.info("Attach mode — leaving Chromium running")
            self.driver = None
            return

        if self.driver:
            try:
                self.driver.quit()
                logger.info("ChromeDriver closed successfully")
            except Exception:
                pass
            self.driver = None

        if self._temp_profile_dir and os.path.isdir(self._temp_profile_dir):
            try:
                shutil.rmtree(self._temp_profile_dir, ignore_errors=True)
            except Exception:
                pass
            self._temp_profile_dir = None


# -------------------------
# Usage example
# -------------------------
if __name__ == "__main__":
    scraper = ThreadsScraper()
    scraper.login()
    posts = scraper.scrape_feed(limit=10)
    for i, p in enumerate(posts, 1):
        logger.info(f"Post {i}: {p['author']} - {p['text'][:50]}...")
    scraper.close()
