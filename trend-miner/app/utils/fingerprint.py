import hashlib
import re
from typing import Optional, Dict, Any
from datetime import datetime


def generate_fingerprint(text: str, author: str = "", timestamp: Optional[str] = None) -> str:
    """
    Generate a unique fingerprint for content to detect duplicates.

    Args:
        text: The main text content
        author: Author/source of the content
        timestamp: Optional timestamp to include

    Returns:
        MD5 hash fingerprint
    """
    if not text:
        text = ""

    # Normalize text: remove extra whitespace, lowercase
    text = re.sub(r'\s+', ' ', text.strip().lower())

    # Remove common variations that don't affect meaning
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation

    # Create unique string components
    components = [text]

    if author:
        components.append(author.lower().strip())

    if timestamp:
        # Normalize timestamp to hour granularity
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            components.append(dt.strftime('%Y%m%d%H'))  # Hour precision
        except Exception:
            components.append(timestamp[:13] if len(timestamp) >= 13 else timestamp)

    # Generate fingerprint
    unique_str = '|'.join(components)
    return hashlib.md5(unique_str.encode()).hexdigest()


def generate_content_hash(content: Dict[str, Any]) -> str:
    """
    Generate a hash for a complete content object.

    Args:
        content: Dictionary containing content data

    Returns:
        SHA-256 hash of the content
    """
    import json
    content_str = json.dumps(content, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(content_str.encode()).hexdigest()


def is_duplicate(existing_hash: str, new_content: Dict[str, Any], threshold: float = 0.95) -> bool:
    """
    Check if new content is a duplicate based on hash comparison.

    Args:
        existing_hash: Hash of existing content
        new_content: New content to check
        threshold: Similarity threshold (for fuzzy matching)

    Returns:
        True if content is considered duplicate
    """
    new_hash = generate_content_hash(new_content)
    if existing_hash == new_hash:
        return True
    return False


def generate_batch_fingerprints(posts: list) -> list:
    """
    Generate fingerprints for a batch of posts.

    Args:
        posts: List of post dictionaries

    Returns:
        List of posts with fingerprints added
    """
    for post in posts:
        if 'fingerprint' not in post:
            post['fingerprint'] = generate_fingerprint(
                post.get('text', ''),
                post.get('author', ''),
                post.get('timestamp')
            )
    return posts
