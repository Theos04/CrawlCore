import re
import emoji
from typing import Dict, Any, List, Tuple
from bs4 import BeautifulSoup


class TextCleaner:
    """Cleans and normalizes text content"""
    
    def __init__(self):
        # Compile regex patterns for efficiency
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+')
        self.mention_pattern = re.compile(r'@\w+')
        self.hashtag_pattern = re.compile(r'#\w+')
        self.html_tag_pattern = re.compile(r'<[^>]+>')
        self.whitespace_pattern = re.compile(r'\s+')
        self.special_chars_pattern = re.compile(r'[^\w\s@#]')
        
    def clean(self, text: str, options: Dict[str, bool] = None) -> str:
        """Clean text based on options"""
        if not text or not isinstance(text, str):
            return ""
        
        default_options = {
            'remove_urls': True,
            'remove_mentions': False,  # Keep mentions by default
            'remove_hashtags': False,  # Keep hashtags by default
            'remove_html': True,
            'remove_emojis': False,    # Keep emojis by default
            'normalize_whitespace': True,
            'lowercase': False,
            'strip_punctuation': False,
            'min_word_length': 1
        }
        
        if options:
            default_options.update(options)
        
        cleaned = text
        
        # Remove HTML tags
        if default_options['remove_html']:
            cleaned = self.html_tag_pattern.sub(' ', cleaned)
        
        # Remove URLs
        if default_options['remove_urls']:
            cleaned = self.url_pattern.sub(' ', cleaned)
        
        # Remove mentions if requested
        if default_options['remove_mentions']:
            cleaned = self.mention_pattern.sub(' ', cleaned)
        
        # Remove hashtags if requested
        if default_options['remove_hashtags']:
            cleaned = self.hashtag_pattern.sub(' ', cleaned)
        
        # Remove emojis if requested
        if default_options['remove_emojis']:
            cleaned = emoji.replace_emoji(cleaned, '')
        
        # Remove punctuation if requested
        if default_options['strip_punctuation']:
            cleaned = self.special_chars_pattern.sub(' ', cleaned)
        
        # Normalize whitespace
        if default_options['normalize_whitespace']:
            cleaned = self.whitespace_pattern.sub(' ', cleaned)
        
        # Convert to lowercase
        if default_options['lowercase']:
            cleaned = cleaned.lower()
        
        # Remove very short words
        if default_options['min_word_length'] > 1:
            words = cleaned.split()
            words = [w for w in words if len(w) >= default_options['min_word_length']]
            cleaned = ' '.join(words)
        
        return cleaned.strip()
    
    def extract_clean_version(self, post: Dict[str, Any]) -> Dict[str, Any]:
        """Extract cleaned version of post text"""
        text = post.get('text', '')
        
        # Create different cleaning versions for different purposes
        cleaned_versions = {
            'cleaned_basic': self.clean(text, {'remove_html': True, 'normalize_whitespace': True}),
            'cleaned_no_mentions': self.clean(text, {'remove_mentions': True}),
            'cleaned_no_hashtags': self.clean(text, {'remove_hashtags': True}),
            'cleaned_minimal': self.clean(text, {
                'remove_urls': True,
                'remove_mentions': True,
                'remove_hashtags': True,
                'remove_emojis': True,
                'strip_punctuation': True,
                'lowercase': True
            })
        }
        
        return cleaned_versions
    
    def extract_components(self, text: str) -> Dict[str, List[str]]:
        """Extract different components from text"""
        return {
            'urls': self.url_pattern.findall(text),
            'mentions': self.mention_pattern.findall(text),
            'hashtags': self.hashtag_pattern.findall(text),
            'emojis': [c for c in text if c in emoji.EMOJI_DATA]
        }
    
    def calculate_readability(self, text: str) -> Dict[str, float]:
        """Calculate readability metrics"""
        if not text:
            return {'flesch_score': 0, 'avg_word_length': 0, 'avg_sentence_length': 0}
        
        # Simple readability metrics
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s for s in sentences if s.strip()]
        
        if not words or not sentences:
            return {'flesch_score': 0, 'avg_word_length': 0, 'avg_sentence_length': 0}
        
        # Average word length
        avg_word_length = sum(len(w) for w in words) / len(words)
        
        # Average sentence length (in words)
        avg_sentence_length = len(words) / len(sentences)
        
        # Flesch Reading Ease (simplified)
        # 206.835 - 1.015 * (words/sentences) - 84.6 * (syllables/words)
        # Approximate syllables by counting vowels
        syllable_count = 0
        for word in words:
            syllable_count += len(re.findall(r'[aeiouy]+', word.lower()))
        
        flesch_score = 206.835 - 1.015 * avg_sentence_length - 84.6 * (syllable_count / len(words))
        flesch_score = max(0, min(100, flesch_score))  # Clamp between 0-100
        
        return {
            'flesch_score': round(flesch_score, 2),
            'avg_word_length': round(avg_word_length, 2),
            'avg_sentence_length': round(avg_sentence_length, 2),
            'word_count': len(words),
            'sentence_count': len(sentences)
        }


from app.utils.logger import setup_logger
logger = setup_logger(__name__)