import json
import hashlib
import re
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import csv
import pandas as pd

from app.config import EXPORTS_DIR
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

def generate_fingerprint(text: str, author: str = "") -> str:
    """Generate a unique fingerprint for content"""
    if not text:
        text = ""
    
    # Normalize text
    text = re.sub(r'\s+', ' ', text.strip().lower())
    
    # Create unique string
    unique_str = f"{author}|{text}"
    
    # Generate hash
    return hashlib.md5(unique_str.encode()).hexdigest()

def sanitize_filename(filename: str) -> str:
    """Sanitize filename by removing invalid characters"""
    # Remove invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Limit length
    if len(filename) > 200:
        name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
        filename = name[:195] + '.' + ext if ext else name[:200]
    return filename

def format_timestamp(timestamp: str) -> str:
    """Format timestamp consistently"""
    try:
        if isinstance(timestamp, str):
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        else:
            dt = timestamp
        return dt.isoformat()
    except:
        return datetime.utcnow().isoformat()

def extract_numbers(text: str) -> List[int]:
    """Extract all numbers from text"""
    return [int(n) for n in re.findall(r'\d+', text)]

def parse_engagement_count(text: str) -> int:
    """Parse engagement counts like '1.2K' or '5M'"""
    try:
        match = re.search(r'([\d.]+)\s*([KM])?', text)
        if match:
            num = float(match.group(1))
            suffix = match.group(2)
            if suffix == 'K':
                num *= 1000
            elif suffix == 'M':
                num *= 1000000
            return int(num)
    except:
        pass
    return 0

def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split a list into chunks"""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def export_to_jsonl(data: List[Dict[str, Any]], filename: str) -> str:
    """Export data to JSONL format"""
    # Ensure .jsonl extension
    if not filename.endswith('.jsonl'):
        filename += '.jsonl'
    
    # Sanitize filename
    filename = sanitize_filename(filename)
    
    # Create full path
    filepath = Path(EXPORTS_DIR) / filename
    
    # Write data
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    logger.info(f"Exported {len(data)} items to {filepath}")
    return str(filepath)

def export_to_csv(data: List[Dict[str, Any]], filename: str) -> str:
    """Export data to CSV format"""
    if not filename.endswith('.csv'):
        filename += '.csv'
    
    filename = sanitize_filename(filename)
    filepath = Path(EXPORTS_DIR) / filename
    
    if not data:
        logger.warning("No data to export")
        return str(filepath)
    
    # Get all possible field names
    fieldnames = set()
    for item in data:
        fieldnames.update(item.keys())
    
    fieldnames = sorted(list(fieldnames))
    
    # Write CSV
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for item in data:
            # Convert complex types to strings
            row = {}
            for key, value in item.items():
                if isinstance(value, (list, dict)):
                    row[key] = json.dumps(value, ensure_ascii=False)
                else:
                    row[key] = value
            writer.writerow(row)
    
    logger.info(f"Exported {len(data)} items to {filepath}")
    return str(filepath)

def export_to_parquet(data: List[Dict[str, Any]], filename: str) -> str:
    """Export data to Parquet format"""
    try:
        if not filename.endswith('.parquet'):
            filename += '.parquet'
        
        filename = sanitize_filename(filename)
        filepath = Path(EXPORTS_DIR) / filename
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Write parquet
        df.to_parquet(filepath, index=False)
        
        logger.info(f"Exported {len(data)} items to {filepath}")
        return str(filepath)
    except Exception as e:
        logger.error(f"Parquet export failed: {e}")
        return None

def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """Load data from JSONL file"""
    data = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        logger.info(f"Loaded {len(data)} items from {filepath}")
    except Exception as e:
        logger.error(f"Error loading {filepath}: {e}")
    
    return data

def safe_get(data: Dict[str, Any], *keys, default=None):
    """Safely get nested dictionary value"""
    for key in keys:
        try:
            data = data[key]
        except (KeyError, TypeError, IndexError):
            return default
    return data

def truncate_text(text: str, max_length: int = 200, suffix: str = "...") -> str:
    """Truncate text to maximum length"""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix

def calculate_age(timestamp: str) -> float:
    """Calculate age in hours of a timestamp"""
    try:
        if isinstance(timestamp, str):
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        else:
            dt = timestamp
        
        now = datetime.utcnow().replace(tzinfo=dt.tzinfo)
        delta = now - dt
        return delta.total_seconds() / 3600  # Hours
    except:
        return 0

def merge_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two dictionaries"""
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    
    return result