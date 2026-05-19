#!/usr/bin/env python3
"""
Setup script for Google Maps Business Scraper
Downloads and bundles Chrome + ChromeDriver
Updated with better error handling, latest Chrome version, and improved platform support
"""

import os
import sys
import platform
import subprocess
import urllib.request
import zipfile
import tarfile
import json
import shutil
import ssl
import time
from pathlib import Path
from urllib.error import URLError, HTTPError

# Latest stable Chrome version (update this periodically)
CHROME_VERSION = "126.0.6478.126"  # Updated to latest stable
CHROME_VERSION_MAJOR = "126"

# Download URLs by platform with fallback mirrors
URLS = {
    "Windows": {
        "chrome": f"https://storage.googleapis.com/chrome-for-testing-public/{CHROME_VERSION}/win64/chrome-win64.zip",
        "driver": f"https://storage.googleapis.com/chrome-for-testing-public/{CHROME_VERSION}/win64/chromedriver-win64.zip",
        "fallback": "https://edgedl.me.gvt1.com/edgedl/chrome/chrome-for-testing"
    },
    "Windows32": {
        "chrome": f"https://storage.googleapis.com/chrome-for-testing-public/{CHROME_VERSION}/win32/chrome-win32.zip",
        "driver": f"https://storage.googleapis.com/chrome-for-testing-public/{CHROME_VERSION}/win32/chromedriver-win32.zip"
    },
    "Linux": {
        "chrome": f"https://storage.googleapis.com/chrome-for-testing-public/{CHROME_VERSION}/linux64/chrome-linux64.zip",
        "driver": f"https://storage.googleapis.com/chrome-for-testing-public/{CHROME_VERSION}/linux64/chromedriver-linux64.zip"
    },
    "MacIntel": {
        "chrome": f"https://storage.googleapis.com/chrome-for-testing-public/{CHROME_VERSION}/mac-x64/chrome-mac-x64.zip",
        "driver": f"https://storage.googleapis.com/chrome-for-testing-public/{CHROME_VERSION}/mac-x64/chromedriver-mac-x64.zip"
    },
    "MacSilicon": {
        "chrome": f"https://storage.googleapis.com/chrome-for-testing-public/{CHROME_VERSION}/mac-arm64/chrome-mac-arm64.zip",
        "driver": f"https://storage.googleapis.com/chrome-for-testing-public/{CHROME_VERSION}/mac-arm64/chromedriver-mac-arm64.zip"
    }
}

def download_file(url, dest, description="Downloading", max_retries=3):
    """Download a file with progress indicator and retry logic"""
    print(f"{description}: {os.path.basename(dest)}")
    print(f"From: {url}")
    
    # Create SSL context that doesn't verify certificates (for problematic networks)
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    
    def report_progress(block_num, block_size, total_size):
        if total_size > 0:
            downloaded = block_num * block_size
            percent = min(int(downloaded * 100 / total_size), 100)
            bar_length = 50
            filled_length = int(bar_length * percent / 100)
            bar = '=' * filled_length + ' ' * (bar_length - filled_length)
            print(f'\r[{bar}] {percent}% ({downloaded // 1024 // 1024}MB/{total_size // 1024 // 1024}MB)', 
                  end='', flush=True)
    
    for attempt in range(max_retries):
        try:
            opener = urllib.request.build_opener()
            opener.addheaders = [('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')]
            urllib.request.install_opener(opener)
            
            urllib.request.urlretrieve(url, dest, reporthook=report_progress)
            print(f"\n✅ Downloaded: {os.path.basename(dest)}")
            return True
        except (URLError, HTTPError, Exception) as e:
            print(f"\n⚠️ Download attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                wait_time = 3 * (attempt + 1)
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"\n❌ Download failed after {max_retries} attempts: {e}")
                raise
    
    return False

def setup_chrome_windows(is_64bit=True):
    """Setup Chrome on Windows with improved error handling"""
    chrome_dir = Path("chrome")
    chrome_dir.mkdir(exist_ok=True)
    
    platform_key = "Windows" if is_64bit else "Windows32"
    urls = URLS[platform_key]
    
    # Download Chrome
    chrome_zip = chrome_dir / "chrome.zip"
    download_file(urls["chrome"], chrome_zip, "Downloading Chrome")
    
    print("Extracting Chrome...")
    try:
        with zipfile.ZipFile(chrome_zip, 'r') as zip_ref:
            zip_ref.extractall(chrome_dir)
    except Exception as e:
        print(f"Error extracting Chrome: {e}")
        raise
    
    # Download ChromeDriver
    driver_zip = chrome_dir / "chromedriver.zip"
    download_file(urls["driver"], driver_zip, "Downloading ChromeDriver")
    
    print("Extracting ChromeDriver...")
    try:
        with zipfile.ZipFile(driver_zip, 'r') as zip_ref:
            zip_ref.extractall(chrome_dir)
    except Exception as e:
        print(f"Error extracting ChromeDriver: {e}")
        raise
    
    # Clean up zip files
    chrome_zip.unlink()
    driver_zip.unlink()
    
    # Find paths with better pattern matching
    chrome_pattern = "chrome-win64" if is_64bit else "chrome-win32"
    driver_pattern = "chromedriver-win64" if is_64bit else "chromedriver-win32"
    
    chrome_path = None
    driver_path = None
    
    # Search for actual paths
    for item in chrome_dir.iterdir():
        if item.is_dir():
            if "chrome" in item.name.lower() and item.name.endswith("64" if is_64bit else "32"):
                chrome_path = item / "chrome.exe"
            if "chromedriver" in item.name.lower() and item.name.endswith("64" if is_64bit else "32"):
                driver_path = item / "chromedriver.exe"
    
    if not chrome_path or not chrome_path.exists():
        # Try alternative naming
        possible_chrome = list(chrome_dir.glob("**/chrome.exe"))
        if possible_chrome:
            chrome_path = possible_chrome[0]
    
    if not driver_path or not driver_path.exists():
        possible_driver = list(chrome_dir.glob("**/chromedriver.exe"))
        if possible_driver:
            driver_path = possible_driver[0]
    
    if not chrome_path or not chrome_path.exists():
        raise FileNotFoundError(f"Chrome not found in {chrome_dir}")
    if not driver_path or not driver_path.exists():
        raise FileNotFoundError(f"ChromeDriver not found in {chrome_dir}")
    
    return str(chrome_path.absolute()), str(driver_path.absolute())

def setup_chrome_linux():
    """Setup Chrome on Linux with dependency checking"""
    chrome_dir = Path("chrome")
    chrome_dir.mkdir(exist_ok=True)
    
    urls = URLS["Linux"]
    
    # Download Chrome
    chrome_zip = chrome_dir / "chrome.zip"
    download_file(urls["chrome"], chrome_zip, "Downloading Chrome")
    
    print("Extracting Chrome...")
    with zipfile.ZipFile(chrome_zip, 'r') as zip_ref:
        zip_ref.extractall(chrome_dir)
    
    # Download ChromeDriver
    driver_zip = chrome_dir / "chromedriver.zip"
    download_file(urls["driver"], driver_zip, "Downloading ChromeDriver")
    
    print("Extracting ChromeDriver...")
    with zipfile.ZipFile(driver_zip, 'r') as zip_ref:
        zip_ref.extractall(chrome_dir)
    
    chrome_zip.unlink()
    driver_zip.unlink()
    
    # Find paths
    chrome_path = None
    driver_path = None
    
    for item in chrome_dir.iterdir():
        if item.is_dir():
            if "chrome-linux64" in item.name:
                chrome_path = item / "chrome"
            if "chromedriver-linux64" in item.name:
                driver_path = item / "chromedriver"
    
    if not chrome_path or not chrome_path.exists():
        raise FileNotFoundError(f"Chrome not found in {chrome_dir}")
    if not driver_path or not driver_path.exists():
        raise FileNotFoundError(f"ChromeDriver not found in {chrome_dir}")
    
    # Make executable
    os.chmod(chrome_path, 0o755)
    os.chmod(driver_path, 0o755)
    
    # Install dependencies for headless mode (check package manager)
    print("\n📦 Checking system dependencies for Chrome...")
    
    # Detect package manager
    package_manager = None
    if shutil.which("apt-get"):
        package_manager = "apt-get"
    elif shutil.which("yum"):
        package_manager = "yum"
    elif shutil.which("dnf"):
        package_manager = "dnf"
    elif shutil.which("pacman"):
        package_manager = "pacman"
    
    if package_manager:
        print(f"Detected package manager: {package_manager}")
        try:
            if package_manager == "apt-get":
                subprocess.run(["sudo", "apt-get", "update"], check=False, capture_output=True)
                subprocess.run(["sudo", "apt-get", "install", "-y", 
                               "libx11-6", "libxcb1", "libxext6", "libxcomposite1",
                               "libxrender1", "libxrandr2", "libxi6", "libgl1-mesa-glx",
                               "libgconf-2-4", "libnss3", "libxss1", "libasound2", 
                               "libatk-bridge2.0-0", "libgtk-3-0"], check=False, capture_output=True)
            elif package_manager in ["yum", "dnf"]:
                subprocess.run(["sudo", package_manager, "install", "-y",
                               "libX11", "libxcb", "libXext", "libXcomposite",
                               "libXrender", "libXrandr", "libXi", "mesa-libGL",
                               "nss", "libXScrnSaver", "alsa-lib"], check=False, capture_output=True)
        except Exception as e:
            print(f"⚠️ Could not install all dependencies: {e}")
    else:
        print("⚠️ Could not detect package manager. Chrome may not run headlessly.")
    
    return str(chrome_path.absolute()), str(driver_path.absolute())

def setup_chrome_mac():
    """Setup Chrome on Mac with improved path detection"""
    chrome_dir = Path("chrome")
    chrome_dir.mkdir(exist_ok=True)
    
    # Detect Apple Silicon vs Intel
    is_silicon = platform.processor() == "arm" or platform.machine() == "arm64"
    platform_key = "MacSilicon" if is_silicon else "MacIntel"
    urls = URLS[platform_key]
    
    # Download Chrome
    chrome_zip = chrome_dir / "chrome.zip"
    download_file(urls["chrome"], chrome_zip, "Downloading Chrome")
    
    print("Extracting Chrome...")
    with zipfile.ZipFile(chrome_zip, 'r') as zip_ref:
        zip_ref.extractall(chrome_dir)
    
    # Download ChromeDriver
    driver_zip = chrome_dir / "chromedriver.zip"
    download_file(urls["driver"], driver_zip, "Downloading ChromeDriver")
    
    print("Extracting ChromeDriver...")
    with zipfile.ZipFile(driver_zip, 'r') as zip_ref:
        zip_ref.extractall(chrome_dir)
    
    chrome_zip.unlink()
    driver_zip.unlink()
    
    # Find paths with better pattern matching
    chrome_path = None
    driver_path = None
    
    for item in chrome_dir.iterdir():
        if item.is_dir():
            if "chrome-mac" in item.name:
                chrome_app = item / "Google Chrome for Testing.app"
                if chrome_app.exists():
                    chrome_path = chrome_app / "Contents" / "MacOS" / "Google Chrome for Testing"
            if "chromedriver-mac" in item.name:
                driver_path = item / "chromedriver"
    
    if not chrome_path or not chrome_path.exists():
        # Try alternative naming
        possible_chrome = list(chrome_dir.glob("**/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing"))
        if possible_chrome:
            chrome_path = possible_chrome[0]
    
    if not driver_path or not driver_path.exists():
        possible_driver = list(chrome_dir.glob("**/chromedriver"))
        if possible_driver:
            driver_path = possible_driver[0]
    
    if not chrome_path or not chrome_path.exists():
        raise FileNotFoundError(f"Chrome not found in {chrome_dir}")
    if not driver_path or not driver_path.exists():
        raise FileNotFoundError(f"ChromeDriver not found in {chrome_dir}")
    
    # Make executable
    os.chmod(chrome_path, 0o755)
    os.chmod(driver_path, 0o755)
    
    # Remove quarantine attribute (macOS security)
    try:
        subprocess.run(["xattr", "-d", "com.apple.quarantine", str(chrome_path)], 
                      check=False, capture_output=True)
        subprocess.run(["xattr", "-d", "com.apple.quarantine", str(driver_path)], 
                      check=False, capture_output=True)
        print("✅ Removed macOS quarantine attributes")
    except Exception:
        pass
    
    return str(chrome_path.absolute()), str(driver_path.absolute())

def create_default_config(chrome_path, driver_path):
    """Create default configuration file with bundled paths"""
    config = {
        "business_type": "",
        "search_query": "",
        "mode": "single",
        "max_workers": 5,
        "scroll_delay": 2,
        "max_scrolls": 15,
        "page_load_delay": 5,
        "delay_between_pincodes": 2,
        "output_dir": str(Path("output").absolute()),
        "csv_file": "",
        "chrome_binary": chrome_path,
        "chromedriver_path": driver_path,
        "selected_states": [],
        "selected_districts": [],
        "filter_mode": "all",
        "window_width": 1400,
        "window_height": 900,
        "window_position_x": 100,
        "window_position_y": 50,
        "postgres_enabled": False,
        "postgres_host": "localhost",
        "postgres_port": 5432,
        "postgres_db": "business_scraper",
        "postgres_user": "postgres",
        "postgres_password": "",
        "auto_export_csv": True,
        "export_format": "both",
        "keep_all_runs": True,
        "headless_mode": False,  # Added option for headless mode
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    
    # Preserve existing config if it exists
    config_path = Path("scraper_config.json")
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                existing = json.load(f)
            # Update only necessary fields
            existing["chrome_binary"] = chrome_path
            existing["chromedriver_path"] = driver_path
            config = existing
            print("✅ Updated existing configuration")
        except Exception:
            print("⚠️ Could not read existing config, creating new one")
    
    with open("scraper_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("✅ Created/Updated configuration")

def create_launcher_scripts():
    """Create platform-specific launcher scripts with better error handling"""
    
    # Windows batch file
    with open("run_scraper.bat", "w") as f:
        f.write("""@echo off
title Google Maps Business Scraper
echo ========================================
echo Google Maps Business Scraper
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not installed or not in PATH
    echo Please install Python 3.8 or higher
    pause
    exit /b 1
)

echo Starting web server...
echo The interface will open in your browser
echo Press Ctrl+C to stop the server
echo.

python app.py
if errorlevel 1 (
    echo.
    echo An error occurred while running the scraper
    echo Check the error messages above
    pause
)
""")
    
    # Linux/Mac shell script
    with open("run_scraper.sh", "w") as f:
        f.write("""#!/bin/bash

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed or not in PATH"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

echo "========================================"
echo "Google Maps Business Scraper"
echo "========================================"
echo ""

echo "Starting web server..."
echo "The interface will open in your browser"
echo "Press Ctrl+C to stop the server"
echo ""

python3 app.py

if [ $? -ne 0 ]; then
    echo ""
    echo "An error occurred while running the scraper"
    echo "Check the error messages above"
    exit 1
fi
""")
    
    # Make shell script executable
    if sys.platform != "win32":
        os.chmod("run_scraper.sh", 0o755)
    
    print("✅ Created launcher scripts")

def verify_installation(chrome_path, driver_path):
    """Verify Chrome and ChromeDriver work with better diagnostics"""
    print("\n🔍 Verifying installation...")
    issues = []
    
    # Check Chrome exists
    if not Path(chrome_path).exists():
        print(f"❌ Chrome not found at: {chrome_path}")
        issues.append("Chrome not found")
    else:
        print(f"✅ Chrome found at: {chrome_path}")
    
    # Check ChromeDriver exists
    if not Path(driver_path).exists():
        print(f"❌ ChromeDriver not found at: {driver_path}")
        issues.append("ChromeDriver not found")
    else:
        print(f"✅ ChromeDriver found at: {driver_path}")
    
    if issues:
        return False
    
    # Test Chrome version
    try:
        result = subprocess.run([chrome_path, "--version"], 
                               capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"✅ Chrome version: {version}")
            # Check if version matches expected
            if CHROME_VERSION_MAJOR not in version:
                print(f"⚠️ Warning: Expected Chrome {CHROME_VERSION_MAJOR}.x, got {version}")
        else:
            print(f"⚠️ Chrome version check returned: {result.stderr}")
    except subprocess.TimeoutExpired:
        print("⚠️ Chrome version check timed out")
    except Exception as e:
        print(f"⚠️ Could not check Chrome version: {e}")
    
    # Test ChromeDriver version
    try:
        result = subprocess.run([driver_path, "--version"], 
                               capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"✅ ChromeDriver version: {version}")
        else:
            print(f"⚠️ ChromeDriver check returned: {result.stderr}")
    except subprocess.TimeoutExpired:
        print("⚠️ ChromeDriver version check timed out")
    except Exception as e:
        print(f"⚠️ Could not check ChromeDriver version: {e}")
    
    return True

def check_requirements_file():
    """Check if requirements.txt exists and create if missing"""
    req_file = Path("requirements.txt")
    if not req_file.exists():
        print("📝 Creating requirements.txt...")
        requirements = """
flask>=2.0.0
selenium>=4.10.0
pandas>=1.5.0
psycopg2-binary>=2.9.0
requests>=2.28.0
beautifulsoup4>=4.11.0
lxml>=4.9.0
openpyxl>=3.0.0
python-dotenv>=0.19.0
"""
        with open(req_file, "w") as f:
            f.write(requirements.strip())
        print("✅ Created requirements.txt")
    return True

def main():
    """Main setup function with improved error handling"""
    print("=" * 60)
    print("Google Maps Business Scraper - Setup")
    print("=" * 60)
    print(f"\nThis will download Chrome {CHROME_VERSION} (bundled version)")
    print("Total download size: ~200-300MB\n")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher required")
        print(f"Current version: {sys.version}")
        print("Please upgrade Python and try again")
        sys.exit(1)
    
    print(f"✅ Python version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    
    # Check if requirements.txt exists
    check_requirements_file()
    
    # Install Python dependencies
    print("\n📦 Installing Python dependencies...")
    try:
        # Upgrade pip first
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                      capture_output=True, check=False)
        
        # Install requirements
        result = subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Dependencies installed successfully")
        else:
            print("⚠️ Some dependencies may have failed to install")
            print(result.stderr[-500:])  # Show last 500 chars of error
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                sys.exit(1)
    except Exception as e:
        print(f"⚠️ Error installing dependencies: {e}")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Detect OS and setup Chrome
    system = platform.system()
    print(f"\n🖥️  Detected OS: {system}")
    print(f"Architecture: {platform.machine()}")
    
    chrome_path = ""
    driver_path = ""
    
    try:
        if system == "Windows":
            # Check if 64-bit
            is_64bit = platform.machine().endswith('64') or '64' in platform.machine()
            print(f"Architecture: {'64-bit' if is_64bit else '32-bit'}")
            chrome_path, driver_path = setup_chrome_windows(is_64bit)
        elif system == "Linux":
            chrome_path, driver_path = setup_chrome_linux()
        elif system == "Darwin":  # macOS
            chrome_path, driver_path = setup_chrome_mac()
        else:
            print(f"❌ Unsupported OS: {system}")
            print("\nPlease install Chrome and ChromeDriver manually:")
            print("1. Download Chrome for Testing from https://googlechromelabs.github.io/chrome-for-testing/")
            print("2. Extract to 'chrome' folder")
            print("3. Update scraper_config.json with paths")
            response = input("\nContinue with manual setup? (y/n): ")
            if response.lower() != 'y':
                sys.exit(1)
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        print("\nTroubleshooting:")
        print("1. Check your internet connection")
        print("2. Try running as administrator")
        print("3. Disable antivirus/firewall temporarily")
        print("4. Install Chrome manually from https://googlechromelabs.github.io/chrome-for-testing/")
        print("5. Update CHROME_VERSION in this script to a newer version if 120 is unavailable")
        
        response = input("\nContinue with manual configuration? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Create run_history directory
    history_dir = Path("run_history")
    history_dir.mkdir(exist_ok=True)
    
    # Create config with bundled paths
    create_default_config(chrome_path, driver_path)
    
    # Create launcher scripts
    create_launcher_scripts()
    
    # Verify installation
    if chrome_path and driver_path:
        verify_installation(chrome_path, driver_path)
    
    # Create or update .gitignore
    gitignore_path = Path(".gitignore")
    if not gitignore_path.exists():
        with open(".gitignore", "w") as f:
            f.write("""# Chrome bundle (too large for git)
chrome/
chromedriver*
chrome.exe
chromedriver.exe

# Output data
output/
run_history/
*.csv
*.json
!scraper_config.json

# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
*.so
*.dll
*.dylib
venv/
env/
ENV/
.venv/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store

# Logs
*.log
scraper.log

# Temporary files
*.tmp
*.temp
temp/
tmp/

# Environment variables
.env
.env.local
""")
        print("✅ Created .gitignore")
    
    print("\n" + "=" * 60)
    print("✅ Setup Complete!")
    print("=" * 60)
    
    if chrome_path and driver_path:
        print(f"\n📁 Chrome installed to: {chrome_path}")
        print(f"🔧 ChromeDriver installed to: {driver_path}")
    
    print("\n📝 Next steps:")
    print("1. Edit 'scraper_config.json' to configure:")
    print("   - Add your CSV file path (csv_file)")
    print("   - Adjust scraping parameters if needed")
    print("   - Configure database settings (optional)")
    print("\n2. Run the scraper:")
    if system == "Windows":
        print("   double-click 'run_scraper.bat'")
        print("   OR run: python app.py")
    else:
        print("   ./run_scraper.sh")
        print("   OR run: python3 app.py")
    print("\n3. Open http://127.0.0.1:5000 in your browser")
    print("\n📊 Bundled Chrome version: " + CHROME_VERSION)
    print("   This version is tested and known to work with the scraper")
    
    # Ask about adding CSV
    print("\n" + "=" * 60)
    response = input("Do you have a CSV file with pincodes ready? (y/n): ")
    if response.lower() == 'y':
        csv_path = input("Enter full path to your CSV file: ").strip()
        if csv_path and Path(csv_path).exists():
            try:
                with open("scraper_config.json", "r") as f:
                    config = json.load(f)
                config["csv_file"] = csv_path
                with open("scraper_config.json", "w") as f:
                    json.dump(config, f, indent=2)
                print("✅ CSV path saved to config")
            except Exception as e:
                print(f"⚠️ Could not save CSV path: {e}")
        elif csv_path:
            print("⚠️ File not found. You can add the path manually to scraper_config.json")
    
    print("\n🎉 Setup complete! You can now run the scraper.")
    print("\n💡 Tip: If you encounter any issues, check the troubleshooting section in the documentation.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)