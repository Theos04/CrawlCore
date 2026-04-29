from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.devtools import v85 as devtools
from typing import Dict, Any, Callable, Optional
import json

from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class NetworkInterceptor:
    """Intercepts network requests for API data capture"""
    
    def __init__(self, driver: WebDriver):
        self.driver = driver
        self.request_log = []
        self.response_log = []
        self.enabled = False
        
        # Try to enable DevTools
        try:
            self.driver.execute_cdp_cmd('Network.enable', {})
            self.enabled = True
            logger.info("Network interception enabled")
        except Exception as e:
            logger.warning(f"Could not enable network interception: {e}")
    
    def add_request_listener(self, callback: Callable):
        """Add listener for network requests"""
        if not self.enabled:
            return
            
        def listener(request):
            callback(request)
            self.request_log.append(request)
        
        self.driver.request_interceptor = listener
    
    def add_response_listener(self, callback: Callable):
        """Add listener for network responses"""
        if not self.enabled:
            return
            
        def listener(response):
            callback(response)
            self.response_log.append(response)
        
        self.driver.response_interceptor = listener
    
    def capture_api_responses(self, url_pattern: str) -> list:
        """Capture responses matching URL pattern"""
        captured = []
        
        def response_listener(response):
            if url_pattern in response['url']:
                try:
                    body = self.driver.execute_cdp_cmd(
                        'Network.getResponseBody',
                        {'requestId': response['requestId']}
                    )
                    captured.append({
                        'url': response['url'],
                        'body': body['body'],
                        'type': body.get('base64Encoded', False)
                    })
                except:
                    pass
        
        self.add_response_listener(response_listener)
        return captured
    
    def block_requests(self, patterns: list):
        """Block requests matching patterns"""
        if not self.enabled:
            return
            
        def request_listener(request):
            for pattern in patterns:
                if pattern in request['url']:
                    request['blocked'] = True
                    logger.debug(f"Blocked request: {request['url']}")
                    break
        
        self.add_request_listener(request_listener)
    
    def get_requests(self, filter_pattern: Optional[str] = None) -> list:
        """Get captured requests, optionally filtered"""
        if filter_pattern:
            return [r for r in self.request_log if filter_pattern in r.get('url', '')]
        return self.request_log.copy()
    
    def get_responses(self, filter_pattern: Optional[str] = None) -> list:
        """Get captured responses, optionally filtered"""
        if filter_pattern:
            return [r for r in self.response_log if filter_pattern in r.get('url', '')]
        return self.response_log.copy()
    
    def clear_logs(self):
        """Clear request and response logs"""
        self.request_log.clear()
        self.response_log.clear()
        logger.debug("Network logs cleared")
    
    def disable(self):
        """Disable network interception"""
        if self.enabled:
            try:
                self.driver.execute_cdp_cmd('Network.disable', {})
                self.enabled = False
                logger.info("Network interception disabled")
            except Exception as e:
                logger.error(f"Error disabling network interception: {e}")