"""
Complete build script for Google Maps Business Scraper
Includes all data files with proper error handling
"""

import os
import sys
import subprocess
import shutil
import time
from pathlib import Path

def kill_existing_process():
    """Kill any running instance of the scraper"""
    print("\n🔍 Checking for running instances...")
    try:
        # Kill any running GoogleMapsScraper processes
        result = subprocess.run(
            ["taskkill", "/F", "/IM", "GoogleMapsScraper.exe"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("   ✅ Stopped existing GoogleMapsScraper.exe")
        else:
            print("   ℹ️ No running instances found")
    except Exception as e:
        print(f"   ⚠️ Could not check for running instances: {e}")

def clean_build_dirs():
    """Clean build directories"""
    print("\n🧹 Cleaning build directories...")
    
    dirs_to_clean = ['build_temp', 'dist', '__pycache__']
    files_to_clean = ['*.spec']
    
    for dir_name in dirs_to_clean:
        dir_path = Path(dir_name)
        if dir_path.exists():
            try:
                shutil.rmtree(dir_path)
                print(f"   ✅ Removed {dir_name}/")
            except Exception as e:
                print(f"   ⚠️ Could not remove {dir_name}/: {e}")
    
    for pattern in files_to_clean:
        for file in Path('.').glob(pattern):
            try:
                file.unlink()
                print(f"   ✅ Removed {file}")
            except Exception as e:
                print(f"   ⚠️ Could not remove {file}: {e}")
    
    time.sleep(1)  # Wait for files to be released

def clean_build():
    """Build executable including all data files"""
    
    print("=" * 60)
    print("Google Maps Business Scraper - Complete Build")
    print("=" * 60)
    
    # Kill any running instances first
    kill_existing_process()
    
    # Clean build directories
    clean_build_dirs()
    
    # Check for data directory
    data_dir = Path("data")
    if not data_dir.exists():
        print("\n❌ ERROR: 'data' directory not found!")
        print("   This directory contains required pincode files.")
        print("   Please ensure the data directory exists with all JSON/CSV files.")
        return False
    
    # Count data files
    data_files = list(data_dir.rglob("*"))
    data_file_count = sum(1 for f in data_files if f.is_file())
    data_size_mb = sum(f.stat().st_size for f in data_files if f.is_file()) / (1024 * 1024)
    
    print(f"\n📁 Found data directory with:")
    print(f"   - {data_file_count} files")
    print(f"   - Total size: {data_size_mb:.2f} MB")
    
    # Check for large files
    large_files = []
    for file in data_dir.rglob("*"):
        if file.is_file():
            size_mb = file.stat().st_size / (1024 * 1024)
            if size_mb > 50:
                large_files.append((file, size_mb))
    
    if large_files:
        print("\n⚠️ Found large files that will increase EXE size:")
        for file, size_mb in large_files:
            print(f"   - {file.relative_to(data_dir)}: {size_mb:.2f} MB")
        
        print("\n   Suggestions:")
        print("   1. Consider compressing these files")
        print("   2. Download on first run instead of bundling")
        print("   3. Use a more efficient format (Parquet instead of CSV)")
        
        response = input("\n   Continue with bundling? (y/n): ")
        if response.lower() != 'y':
            print("   Build cancelled.")
            return False
    
    # Build command with exclusions
    print("\n🔨 Building executable (this may take 5-15 minutes depending on data size)...")
    print("   Including all data files from 'data/' directory...")
    print("   Excluding heavy packages to reduce size...\n")
    
    cmd = [
        "pyinstaller",
        "--clean",
        "--noconfirm",
        "--onedir",
        "--name", "GoogleMapsScraper",
        "--workpath", "build_temp",
        "--distpath", "dist",
        "--specpath", ".",
        # Exclude heavy modules
        "--exclude-module", "PyQt5",
        "--exclude-module", "PyQt6", 
        "--exclude-module", "PySide2",
        "--exclude-module", "PySide6",
        "--exclude-module", "matplotlib",
        "--exclude-module", "tensorflow",
        "--exclude-module", "torch",
        "--exclude-module", "pygame",
        "--exclude-module", "scipy",
        "--exclude-module", "sklearn",
        "--exclude-module", "notebook",
        "--exclude-module", "jupyter",
        "--exclude-module", "IPython",
        "--exclude-module", "tornado",
        "--exclude-module", "zmq",
        "--exclude-module", "tkinter",
        "--exclude-module", "pytest",
        "--exclude-module", "numba",
        "--exclude-module", "llvmlite",
        # Include data directories
        "--add-data", f"templates{os.pathsep}templates",
        "--add-data", f"static{os.pathsep}static",
        "--add-data", f"data{os.pathsep}data",
        "--add-data", f"chrome{os.pathsep}chrome",
        # Check for config file
        *(["--add-data", f"scraper_config.json{os.pathsep}."] if Path("scraper_config.json").exists() else []),
        # Hidden imports
        "--hidden-import", "selenium",
        "--hidden-import", "selenium.webdriver",
        "--hidden-import", "selenium.webdriver.common",
        "--hidden-import", "selenium.webdriver.chrome",
        "--hidden-import", "flask",
        "--hidden-import", "pandas",
        "--hidden-import", "openpyxl",
        "--hidden-import", "bs4",
        "--hidden-import", "lxml",
        "--hidden-import", "requests",
        "--hidden-import", "json",
        "--hidden-import", "csv",
        "--hidden-import", "datetime",
        "--hidden-import", "threading",
        "--hidden-import", "queue",
        "--hidden-import", "time",
        "--hidden-import", "os",
        "--hidden-import", "sys",
        "--hidden-import", "pathlib",
        # Main script
        "app.py"
    ]
    
    try:
        # Run pyinstaller
        result = subprocess.run(cmd, capture_output=False, text=True, check=False)
        
        if result.returncode == 0:
            print("\n" + "=" * 60)
            print("✅ Build Successful!")
            print("=" * 60)
            
            # Check file size
            dist_dir = Path("dist") / "GoogleMapsScraper"
            exe_path = dist_dir / "GoogleMapsScraper.exe"
            if exe_path.exists():
                size_mb = sum(f.stat().st_size for f in dist_dir.rglob('*') if f.is_file()) / (1024 * 1024)
                print(f"\n📁 Distribution folder created: {dist_dir}")
                print(f"📊 Total size: {size_mb:.2f} MB")
                
                # Create launcher and helper scripts
                create_launcher(dist_dir)
                create_data_extractor_check(dist_dir)
                
                print("\n📦 Zipping the distribution folder...")
                shutil.make_archive("GoogleMapsScraper_Portable", 'zip', "dist", "GoogleMapsScraper")
                print(f"✅ Created GoogleMapsScraper_Portable.zip")
                
                return True
            else:
                print("❌ Executable not found after build")
                return False
        else:
            print(f"\n❌ Build failed with return code: {result.returncode}")
            return False
            
    except PermissionError as e:
        print(f"\n❌ Permission Error: {e}")
        print("\n   This usually means the EXE file is locked.")
        print("   Try these solutions:")
        print("   1. Close any running GoogleMapsScraper.exe")
        print("   2. Run: taskkill /F /IM GoogleMapsScraper.exe")
        print("   3. Restart your computer if the file is still locked")
        return False
        
    except Exception as e:
        print(f"\n❌ Build error: {e}")
        return False

def create_launcher(dist_dir):
    """Create enhanced launcher script inside dist folder"""
    launcher = '''@echo off
title Google Maps Business Scraper
color 0A

echo ========================================
echo    Google Maps Business Scraper
echo ========================================
echo.

REM Check if Chrome exists
if not exist "chrome\\" (
    echo [WARNING] Chrome browser bundle not found!
    echo.
    echo Please run setup.py first to download Chrome:
    echo   python setup.py
    echo.
    pause
    exit /b 1
)

REM Check if executable exists
if not exist "GoogleMapsScraper.exe" (
    echo [ERROR] Executable not found!
    echo Please run build_exe.py first.
    echo.
    pause
    exit /b 1
)

echo Starting Google Maps Business Scraper...
echo.
echo The application will open in your browser at:
echo   http://127.0.0.1:5000
echo.
echo DO NOT close this window while the app is running!
echo.
echo Press Ctrl+C to stop the application
echo.

REM Run the executable
"GoogleMapsScraper.exe"

echo.
echo Application has stopped.
pause
'''
    
    bat_path = Path(dist_dir) / "run.bat"
    with open(bat_path, "w", encoding='utf-8') as f:
        f.write(launcher)
    
    print(f"✅ Created launcher: {bat_path}")

def create_data_extractor_check(dist_dir):
    """Create a script to check extracted data files"""
    checker = '''# data_checker.py
# Run this to verify data files are properly extracted

import os
import sys
from pathlib import Path

def check_data_files():
    """Check if data files are accessible in the bundled app"""
    
    print("Checking data files accessibility...")
    print("-" * 40)
    
    # Get the directory where the executable is running from
    if getattr(sys, 'frozen', False):
        # Running as bundled executable
        base_dir = sys._MEIPASS
        print(f"Running from: {base_dir}")
    else:
        # Running as script
        base_dir = os.path.dirname(os.path.abspath(__file__))
        print(f"Running from: {base_dir}")
    
    data_dir = Path(base_dir) / "data"
    
    if not data_dir.exists():
        print("❌ ERROR: data directory not found in bundled files!")
        return False
    
    print(f"✅ data directory found at: {data_dir}")
    
    # Count files
    files = list(data_dir.rglob("*"))
    json_files = [f for f in files if f.suffix == '.json']
    csv_files = [f for f in files if f.suffix == '.csv']
    
    print(f"\\n📊 Data files summary:")
    print(f"   - Total files: {len(files)}")
    print(f"   - JSON files: {len(json_files)}")
    print(f"   - CSV files: {len(csv_files)}")
    
    # Check for specific files
    india_csv = data_dir / "pincodes-india" / "5c2f62fe-5afa-4119-a499-fec9d604d5bd.csv"
    if india_csv.exists():
        size_mb = india_csv.stat().st_size / (1024 * 1024)
        print(f"   ✅ India pincode CSV found ({size_mb:.2f} MB)")
    else:
        print(f"   ⚠️ India pincode CSV not found")
    
    # Show some sample files
    print(f"\\n📁 Sample of included files:")
    for f in list(files)[:10]:
        if f.is_file():
            print(f"   - {f.relative_to(data_dir)}")
    
    if len(files) > 10:
        print(f"   ... and {len(files) - 10} more files")
    
    return True

if __name__ == "__main__":
    success = check_data_files()
    sys.exit(0 if success else 1)
'''
    
    checker_path = Path(dist_dir) / "check_data_files.py"
    with open(checker_path, "w", encoding='utf-8') as f:
        f.write(checker)
    
    print(f"✅ Created data checker: {checker_path}")

def check_requirements():
    """Check if all required files and directories exist"""
    
    print("\n🔍 Checking build requirements...")
    print("-" * 40)
    
    # Required files
    required_files = [
        ('app.py', "Main application file"),
        ('templates/index.html', "Web template"),
        ('static/', "Static assets directory"),
        ('data/', "Pincode data directory"),
    ]
    
    missing = []
    warnings = []
    
    for path, description in required_files:
        p = Path(path)
        if p.exists():
            print(f"✅ {description}: {path}")
        else:
            if path.endswith('/'):
                print(f"⚠️ {description}: {path} - Not found (creating...)")
                Path(path).mkdir(exist_ok=True)
                warnings.append(path)
            else:
                print(f"❌ {description}: {path} - NOT FOUND")
                missing.append(path)
    
    # Check data directory contents
    data_dir = Path("data")
    if data_dir.exists():
        data_files = list(data_dir.rglob("*"))
        json_count = sum(1 for f in data_files if f.is_file() and f.suffix == '.json')
        csv_count = sum(1 for f in data_files if f.is_file() and f.suffix == '.csv')
        
        if json_count > 0 or csv_count > 0:
            print(f"✅ Data files: {json_count} JSON, {csv_count} CSV files found")
        else:
            print(f"⚠️ Warning: No JSON or CSV files found in data/ directory")
            print(f"   This may cause scraping to fail for some countries")
            warnings.append("No data files")
    
    if missing:
        print("\n" + "=" * 60)
        print("❌ CRITICAL: Missing required files!")
        print("=" * 60)
        for m in missing:
            print(f"   - {m}")
        return False
    
    if warnings:
        print("\n⚠️ Warnings:")
        for w in warnings:
            print(f"   - {w}")
    
    print("\n✅ All critical requirements satisfied!")
    return True

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Google Maps Business Scraper - Complete Build Script")
    print("=" * 60)
    
    # Check requirements
    if not check_requirements():
        print("\nPlease fix the missing requirements and try again.")
        sys.exit(1)
    
    # Build the executable
    success = clean_build()
    
    if success:
        print("\n" + "=" * 60)
        print("🎉 BUILD COMPLETE! 🎉")
        print("=" * 60)
        print("\nNext Steps:")
        print("-" * 40)
        print("\n1. Distribute the zip file:")
        print("   GoogleMapsScraper_Portable.zip")
        print("\n2. Run the application from the extracted folder:")
        print("   run.bat")
        print("\n3. Verify data files are bundled correctly:")
        print("   python check_data_files.py")
        print("\n4. The portable folder is located at:")
        print("   dist\\GoogleMapsScraper\\")
        print("\n" + "=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ BUILD FAILED")
        print("=" * 60)
        print("\nTroubleshooting Steps:")
        print("-" * 40)
        print("\n1. Make sure no instance is running:")
        print("   taskkill /F /IM GoogleMapsScraper.exe")
        print("\n2. Try a clean build:")
        print("   Remove-Item -Path build_temp, dist -Recurse -Force")
        print("   py build_exe.py")
        print("\n3. Run as Administrator (if permission issues)")
        print("\n4. Check disk space (need at least 2GB free)")
        print("\n5. If data files are too large, consider:")
        print("   - Excluding large CSV files")
        print("   - Using --onedir instead of --onefile")
        print("\n6. For more details, check:")
        print("   build_temp\\warn-*.txt")
        print("=" * 60)
        sys.exit(1)