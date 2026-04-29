import json
import os

def extract_sessionid_from_har(har_file_path):
    """Extract Instagram sessionid from HAR file"""
    
    if not os.path.exists(har_file_path):
        print(f"❌ File not found: {har_file_path}")
        return None
    
    try:
        with open(har_file_path, 'r', encoding='utf-8') as f:
            har_data = json.load(f)
    except Exception as e:
        print(f"❌ Error reading HAR file: {e}")
        return None
    
    session_ids = []
    
    # Navigate through HAR structure
    entries = har_data.get('log', {}).get('entries', [])
    
    for entry in entries:
        # Check request cookies
        request = entry.get('request', {})
        cookies = request.get('cookies', [])
        
        for cookie in cookies:
            if cookie.get('name') == 'sessionid':
                session_ids.append({
                    'url': request.get('url', ''),
                    'value': cookie.get('value', ''),
                    'source': 'request'
                })
        
        # Check response cookies (more reliable)
        response = entry.get('response', {})
        cookies = response.get('cookies', [])
        
        for cookie in cookies:
            if cookie.get('name') == 'sessionid':
                session_ids.append({
                    'url': request.get('url', ''),
                    'value': cookie.get('value', ''),
                    'source': 'response'
                })
        
        # Also check Set-Cookie headers
        headers = response.get('headers', [])
        for header in headers:
            if header.get('name', '').lower() == 'set-cookie':
                cookie_str = header.get('value', '')
                if 'sessionid=' in cookie_str:
                    # Extract sessionid from cookie string
                    import re
                    match = re.search(r'sessionid=([^;]+)', cookie_str)
                    if match:
                        session_ids.append({
                            'url': request.get('url', ''),
                            'value': match.group(1),
                            'source': 'set-cookie header'
                        })
    
    # Remove duplicates and return the most recent/likely valid one
    if session_ids:
        # Get unique values
        unique_values = list(set([s['value'] for s in session_ids]))
        
        print(f"\n✅ Found {len(unique_values)} unique session ID(s):\n")
        
        for i, sid in enumerate(unique_values, 1):
            print(f"Option {i}:")
            print("-" * 60)
            print(sid)
            print("-" * 60)
            print()
        
        # Return the first one (usually the most valid)
        return unique_values[0]
    else:
        print("❌ No sessionid found in HAR file")
        return None

def extract_all_cookies_from_har(har_file_path):
    """Extract all Instagram cookies from HAR file"""
    
    if not os.path.exists(har_file_path):
        print(f"❌ File not found: {har_file_path}")
        return None
    
    try:
        with open(har_file_path, 'r', encoding='utf-8') as f:
            har_data = json.load(f)
    except Exception as e:
        print(f"❌ Error reading HAR file: {e}")
        return None
    
    all_cookies = {}
    entries = har_data.get('log', {}).get('entries', [])
    
    for entry in entries:
        request = entry.get('request', {})
        
        # Only look at Instagram requests
        url = request.get('url', '')
        if 'instagram.com' in url:
            # Check request cookies
            cookies = request.get('cookies', [])
            for cookie in cookies:
                name = cookie.get('name')
                value = cookie.get('value')
                if name and value:
                    all_cookies[name] = value
    
    return all_cookies

if __name__ == "__main__":
    har_file = r"E:\www.instagram.com.har"
    
    print("🔍 Extracting session ID from HAR file...")
    print("=" * 60)
    
    # Method 1: Just get sessionid
    sessionid = extract_sessionid_from_har(har_file)
    
    if sessionid:
        print("\n📋 Copy this to your config file:")
        print('INSTAGRAM_SESSIONID = "' + sessionid + '"')
    
    # Method 2: Show all cookies (optional)
    print("\n" + "=" * 60)
    print("🔍 Do you want to see all Instagram cookies? (y/n)")
    choice = input().lower()
    
    if choice == 'y':
        all_cookies = extract_all_cookies_from_har(har_file)
        if all_cookies:
            print("\n📋 All Instagram cookies found:")
            print("=" * 60)
            for name, value in all_cookies.items():
                print(f"{name}: {value[:50]}..." if len(value) > 50 else f"{name}: {value}")