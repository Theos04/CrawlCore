# Google Maps Business Scraper

A powerful, production-ready web scraper for extracting business data from Google Maps. Features persistent storage, run history, PostgreSQL support, and an intuitive web interface.

## 🚀 Features

- **No Chrome installation needed** - Bundles Chrome for Testing automatically
- **Persistent runs** - Each scraping session is saved separately with full history
- **Web interface** - User-friendly dashboard to control and monitor scraping
- **Filter by location** - Select specific states/districts to scrape
- **Dual mode** - Single instance (stable) or multi-instance (fast)
- **Multiple exports** - JSON, CSV, and PostgreSQL support
- **Resume capability** - Continue interrupted runs from where they stopped
- **Run history** - View, export, or delete any past run
- **Real-time logs** - Live console output during scraping

## 📋 Requirements

- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended for multi-instance)
- 500MB free disk space (for Chrome bundle)
- Internet connection

## 🛠️ Quick Installation

### Windows
```batch
python setup.py
run_scraper.bat