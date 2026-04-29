from selenium.webdriver.remote.webdriver import WebDriver
from selenium.common.exceptions import JavascriptException
import time
from typing import Optional


class ScrollHandler:
    """Handles infinite scroll interactions"""
    
    def __init__(self, driver: WebDriver):
        self.driver = driver
        self.last_height = self.get_scroll_height()
        self.scroll_count = 0
        
    def get_scroll_height(self) -> int:
        """Get current scroll height"""
        try:
            return self.driver.execute_script("return document.body.scrollHeight")
        except JavascriptException:
            return 0
    
    def get_scroll_position(self) -> int:
        """Get current scroll position"""
        try:
            return self.driver.execute_script("return window.pageYOffset")
        except JavascriptException:
            return 0
    
    def scroll_to_bottom(self, smooth: bool = True):
        """Scroll to bottom of page"""
        if smooth:
            script = "window.scrollTo({top: document.body.scrollHeight, behavior: 'smooth'})"
        else:
            script = "window.scrollTo(0, document.body.scrollHeight)"
        
        try:
            self.driver.execute_script(script)
        except JavascriptException as e:
            logger.error(f"Scroll failed: {e}")
    
    def scroll_down(self, pixels: Optional[int] = None):
        """Scroll down by specified pixels or to bottom"""
        if pixels:
            script = f"window.scrollBy(0, {pixels})"
        else:
            script = "window.scrollTo(0, document.body.scrollHeight)"
        
        try:
            self.driver.execute_script(script)
            self.scroll_count += 1
            logger.debug(f"Scrolled down ({self.scroll_count})")
        except JavascriptException as e:
            logger.error(f"Scroll failed: {e}")
    
    def scroll_up(self, pixels: int):
        """Scroll up by specified pixels"""
        try:
            self.driver.execute_script(f"window.scrollBy(0, -{pixels})")
        except JavascriptException as e:
            logger.error(f"Scroll up failed: {e}")
    
    def scroll_to_element(self, element):
        """Scroll to specific element"""
        try:
            self.driver.execute_script(
                "arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});",
                element
            )
        except JavascriptException as e:
            logger.error(f"Scroll to element failed: {e}")
    
    def has_more_content(self) -> bool:
        """Check if there's more content to load"""
        new_height = self.get_scroll_height()
        return new_height > self.last_height
    
    def wait_for_content(self, timeout: int = 5):
        """Wait for new content to load after scroll"""
        start_time = time.time()
        current_height = self.get_scroll_height()
        
        while time.time() - start_time < timeout:
            time.sleep(0.5)
            new_height = self.get_scroll_height()
            if new_height > current_height:
                self.last_height = new_height
                return True
        
        return False
    
    def reset(self):
        """Reset scroll handler state"""
        self.last_height = self.get_scroll_height()
        self.scroll_count = 0


from app.utils.logger import setup_logger
logger = setup_logger(__name__)