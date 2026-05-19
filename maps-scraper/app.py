"""
app.py — Google Maps Business Scraper (Persistent + Versioned)
Features: 
- Each run saves to a timestamped folder
- View history of all runs
- Compare results across runs
- Persistent PostgreSQL storage with run tracking
- Resume capability within the same run

Run: python app.py  →  http://127.0.0.1:5000
"""

import csv
import json
import os
import queue
import re
import shutil
import sys
import tempfile
import threading
import time
import webbrowser
from collections import OrderedDict, defaultdict
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from flask import Flask, Response, abort, jsonify, render_template_string, request, send_file

try:
    import psycopg2
    from psycopg2.pool import ThreadedConnectionPool
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False
    print("⚠️  psycopg2 not installed. PostgreSQL disabled. pip install psycopg2-binary")

app = Flask(__name__)
CONFIG_PATH = Path("scraper_config.json")
RUNS_DB_PATH = Path("scraping_runs.json")  # Track all runs


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

BASE_DIR = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))

_DEFAULTS: Dict = {
    "business_type": "",
    "search_query": "",
    "mode": "single",
    "max_workers": 5,
    "scroll_delay": 2.0,
    "max_scrolls": 15,
    "page_load_delay": 5.0,
    "delay_between_pincodes": 2.0,
    "output_dir": os.path.join(BASE_DIR, "exports"),
    "csv_file": "",
    "chrome_binary": os.path.join(BASE_DIR, "chrome", "chrome-win", "chrome.exe"),
    "chromedriver_path": os.path.join(BASE_DIR, "chrome", "chromedriver_win32", "chromedriver.exe"),
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
    "keep_all_runs": True,  # NEW: Keep all runs instead of overwriting
}


def load_config() -> Dict:
    cfg = _DEFAULTS.copy()
    if CONFIG_PATH.exists():
        try:
            saved = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            normalized = {k.lower(): v for k, v in saved.items()}
            cfg.update(normalized)
        except Exception as e:
            print(f"⚠️ Config load error: {e}")
    return cfg


def save_config(cfg: Dict) -> None:
    try:
        CONFIG_PATH.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        print(f"⚠️ Config save error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  RUNS DATABASE (Track all scraping runs)
# ══════════════════════════════════════════════════════════════════════════════

def load_runs_db() -> Dict:
    """Load the runs database that tracks all scraping runs"""
    if RUNS_DB_PATH.exists():
        try:
            return json.loads(RUNS_DB_PATH.read_text(encoding="utf-8"))
        except:
            return {"runs": [], "last_run_id": 0}
    return {"runs": [], "last_run_id": 0}


def save_runs_db(runs_db: Dict) -> None:
    """Save the runs database"""
    try:
        RUNS_DB_PATH.write_text(json.dumps(runs_db, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        print(f"⚠️ Cannot save runs DB: {e}")


def create_run_record(cfg: Dict) -> str:
    """Create a new run record and return run_id"""
    runs_db = load_runs_db()
    run_id = runs_db["last_run_id"] + 1
    runs_db["last_run_id"] = run_id
    
    # Create timestamped folder for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder_name = f"run_{run_id:04d}_{timestamp}_{cfg['business_type'].replace(' ', '_')}"
    run_folder = Path(cfg["output_dir"]) / "run_history" / run_folder_name
    run_folder.mkdir(parents=True, exist_ok=True)
    
    run_record = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "business_type": cfg["business_type"],
        "search_query": cfg.get("search_query", cfg["business_type"]),
        "mode": cfg.get("mode", "single"),
        "filter_mode": cfg.get("filter_mode", "all"),
        "selected_states": cfg.get("selected_states", []),
        "selected_districts": cfg.get("selected_districts", []),
        "total_pincodes": 0,
        "businesses_found": 0,
        "status": "running",
        "folder": str(run_folder),
        "json_path": str(run_folder / "results.json"),
        "csv_path": str(run_folder / "all_results.csv"),
        "config_snapshot": cfg.copy()
    }
    
    runs_db["runs"].insert(0, run_record)  # Newest first
    save_runs_db(runs_db)
    
    return run_id, run_record


def update_run_status(run_id: int, status: str, businesses_found: int = None, total_pincodes: int = None):
    """Update run status in the database"""
    runs_db = load_runs_db()
    for run in runs_db["runs"]:
        if run["run_id"] == run_id:
            run["status"] = status
            if businesses_found is not None:
                run["businesses_found"] = businesses_found
            if total_pincodes is not None:
                run["total_pincodes"] = total_pincodes
            if status in ["completed", "stopped", "failed"]:
                run["completed_at"] = datetime.now().isoformat()
            break
    save_runs_db(runs_db)


def get_all_runs() -> List[Dict]:
    """Get all scraping runs"""
    runs_db = load_runs_db()
    return runs_db["runs"]


def get_run_by_id(run_id: int) -> Optional[Dict]:
    """Get a specific run by ID"""
    runs_db = load_runs_db()
    for run in runs_db["runs"]:
        if run["run_id"] == run_id:
            return run
    return None


def delete_run(run_id: int) -> bool:
    """Delete a run and its files"""
    runs_db = load_runs_db()
    run_to_delete = None
    
    for i, run in enumerate(runs_db["runs"]):
        if run["run_id"] == run_id:
            run_to_delete = runs_db["runs"].pop(i)
            break
    
    if run_to_delete:
        save_runs_db(runs_db)
        # Optionally delete files
        try:
            folder = Path(run_to_delete["folder"])
            if folder.exists():
                shutil.rmtree(folder)
        except:
            pass
        return True
    return False


# ══════════════════════════════════════════════════════════════════════════════
#  RUNTIME STATE
# ══════════════════════════════════════════════════════════════════════════════

_log_queue: queue.Queue = queue.Queue()
_scraper_thread: Optional[threading.Thread] = None
_stop_event = threading.Event()
_write_lock = threading.Lock()
_stats: Dict = {
    "pincodes_done": 0,
    "businesses_found": 0,
    "total_pincodes": 0,
    "filtered_pincodes": 0,
    "running": False,
    "current_run_id": None,
}
_current_run_id: Optional[int] = None
_db_pool: Optional[ThreadedConnectionPool] = None


def _log(msg: str) -> None:
    _log_queue.put({"type": "log", "msg": msg})
    print(msg)


def _push_stats() -> None:
    _log_queue.put({"type": "stats", **_stats})


# ══════════════════════════════════════════════════════════════════════════════
#  DATABASE (PostgreSQL - Persistent)
# ══════════════════════════════════════════════════════════════════════════════

def init_db(cfg: Dict) -> bool:
    global _db_pool
    if not PSYCOPG2_AVAILABLE or not cfg.get("postgres_enabled"):
        return False
    try:
        _db_pool = ThreadedConnectionPool(
            minconn=1,
            maxconn=max(cfg.get("max_workers", 5) + 2, 4),
            host=cfg["postgres_host"],
            port=cfg["postgres_port"],
            dbname=cfg["postgres_db"],
            user=cfg["postgres_user"],
            password=cfg["postgres_password"],
        )
        with _db_conn() as conn:
            _create_tables(conn)
        _log("✅ PostgreSQL connected and tables ready")
        return True
    except Exception as e:
        _log(f"❌ PostgreSQL connection failed: {e}")
        _db_pool = None
        return False


@contextmanager
def _db_conn():
    if not _db_pool:
        yield None
        return
    conn = _db_pool.getconn()
    try:
        yield conn
    except Exception:
        conn.rollback()
        raise
    finally:
        _db_pool.putconn(conn)


def _create_tables(conn) -> None:
    if conn is None:
        return
    cur = conn.cursor()
    
    # Add run_id column to businesses table if not exists
    cur.execute("""
        DO $$ 
        BEGIN 
            IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                          WHERE table_name='businesses' AND column_name='run_id') THEN
                ALTER TABLE businesses ADD COLUMN run_id INTEGER;
            END IF;
        END $$;
    """)
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS businesses (
            id            SERIAL PRIMARY KEY,
            name          VARCHAR(500) NOT NULL,
            category      VARCHAR(200),
            rating        DECIMAL(3,1),
            reviews       INTEGER,
            phone         VARCHAR(50),
            address       TEXT,
            website       VARCHAR(500),
            pincode       VARCHAR(10),
            district      VARCHAR(100),
            state         VARCHAR(100),
            business_type VARCHAR(200),
            search_query  VARCHAR(500),
            run_id        INTEGER,
            scraped_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    cur.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS uidx_biz_with_phone
        ON businesses (name, pincode, phone, run_id)
        WHERE phone IS NOT NULL AND phone <> ''
    """)
    cur.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS uidx_biz_no_phone
        ON businesses (name, pincode, run_id)
        WHERE phone IS NULL OR phone = ''
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_pincode   ON businesses(pincode)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_state     ON businesses(state)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_district  ON businesses(district)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_run_id    ON businesses(run_id)")
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS scraping_sessions (
            id               SERIAL PRIMARY KEY,
            run_id           INTEGER,
            session_id       VARCHAR(100) UNIQUE NOT NULL,
            business_type    VARCHAR(200),
            search_query     VARCHAR(500),
            businesses_found INTEGER DEFAULT 0,
            started_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at     TIMESTAMP,
            status           VARCHAR(50) DEFAULT 'running'
        )
    """)
    conn.commit()
    cur.close()


def _db_save_businesses(businesses: List[Dict], run_id: int, cfg: Dict) -> int:
    """Save businesses to PostgreSQL with run_id"""
    if not _db_pool or not businesses:
        return 0
    saved = 0
    try:
        with _db_conn() as conn:
            if conn is None:
                return 0
            cur = conn.cursor()
            for biz in businesses:
                phone = (biz.get("phone") or "").strip() or None
                try:
                    if phone:
                        cur.execute("""
                            INSERT INTO businesses
                                (name, category, rating, reviews, phone, address, website,
                                 pincode, district, state, business_type, search_query, run_id, scraped_at)
                            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                            ON CONFLICT (name, pincode, phone, run_id)
                                WHERE phone IS NOT NULL AND phone <> ''
                            DO UPDATE SET
                                category = EXCLUDED.category,
                                rating = EXCLUDED.rating,
                                scraped_at = EXCLUDED.scraped_at
                            RETURNING id
                        """, (
                            biz.get('name', '')[:500],
                            biz.get('category', '')[:200],
                            float(biz['rating']) if biz.get('rating') else None,
                            int(str(biz['reviews']).replace(",","")) if biz.get('reviews') else None,
                            phone[:50],
                            (biz.get('address') or '')[:1000],
                            (biz.get('website') or '')[:500],
                            biz.get('pincode', '')[:10],
                            biz.get('district', '')[:100],
                            biz.get('state', '')[:100],
                            cfg.get('business_type', '')[:200],
                            cfg.get('_search_query', '')[:500],
                            run_id,
                            datetime.now()
                        ))
                    else:
                        cur.execute("""
                            INSERT INTO businesses
                                (name, category, rating, reviews, address, website,
                                 pincode, district, state, business_type, search_query, run_id, scraped_at)
                            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                            ON CONFLICT (name, pincode, run_id)
                                WHERE phone IS NULL OR phone = ''
                            DO NOTHING
                            RETURNING id
                        """, (
                            biz.get('name', '')[:500],
                            biz.get('category', '')[:200],
                            float(biz['rating']) if biz.get('rating') else None,
                            int(str(biz['reviews']).replace(",","")) if biz.get('reviews') else None,
                            (biz.get('address') or '')[:1000],
                            (biz.get('website') or '')[:500],
                            biz.get('pincode', '')[:10],
                            biz.get('district', '')[:100],
                            biz.get('state', '')[:100],
                            cfg.get('business_type', '')[:200],
                            cfg.get('_search_query', '')[:500],
                            run_id,
                            datetime.now()
                        ))
                    if cur.fetchone():
                        saved += 1
                except Exception as e:
                    _log(f"  ⚠️ DB insert ({biz.get('name','')}): {e}")
                    conn.rollback()
                    continue
            conn.commit()
            cur.close()
    except Exception as e:
        _log(f"❌ DB batch save error: {e}")
    return saved


# ══════════════════════════════════════════════════════════════════════════════
#  CSV EXPORT
# ══════════════════════════════════════════════════════════════════════════════

_CSV_FIELDS = [
    "name","category","rating","reviews","phone","address",
    "website","pincode","district","state","business_type",
    "scraped_at","search_query","run_id",
]


def export_to_csv(businesses: List[Dict], path: Path, run_id: int, mode: str = "append") -> bool:
    if not businesses:
        return False
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_mode = "a" if mode == "append" and path.exists() else "w"
        with path.open(write_mode, newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS, extrasaction="ignore")
            if write_mode == "w":
                writer.writeheader()
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for biz in businesses:
                biz.setdefault("scraped_at", ts)
                biz["run_id"] = run_id
                writer.writerow(biz)
        return True
    except Exception as e:
        _log(f"❌ CSV export error: {e}")
        return False


# ══════════════════════════════════════════════════════════════════════════════
#  PINCODE LOADING & FILTERING
# ══════════════════════════════════════════════════════════════════════════════

def load_pincodes(csv_file: str) -> "OrderedDict[str, Dict]":
    pincodes: OrderedDict = OrderedDict()
    for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            with open(csv_file, "r", encoding=encoding, newline="") as f:
                sample = f.read(65536)
                f.seek(0)
                try:
                    dialect = csv.Sniffer().sniff(sample)
                except Exception:
                    dialect = "excel"
                reader = csv.DictReader(f, dialect=dialect)
                if not reader.fieldnames:
                    continue

                pincode_col = district_col = state_col = office_col = None
                for col in reader.fieldnames:
                    cl = col.lower().strip()
                    if any(x in cl for x in ("pincode","pin code","postal","postcode")) or cl == "pin":
                        pincode_col = col
                    elif "district" in cl:
                        district_col = col
                    elif "state" in cl:
                        state_col = col
                    elif "office" in cl:
                        office_col = col

                if not pincode_col:
                    raise ValueError(f"No pincode column in: {reader.fieldnames}")

                for row in reader:
                    pc = re.sub(r"\D", "", str(row[pincode_col]).strip())
                    if pc and len(pc) >= 4 and pc not in pincodes:
                        pincodes[pc] = {
                            "pincode":  pc,
                            "district": (row.get(district_col) or "").strip() if district_col else "",
                            "state":    (row.get(state_col)    or "").strip() if state_col    else "",
                            "office":   (row.get(office_col)   or "").strip() if office_col   else "",
                        }
            if pincodes:
                return pincodes
        except Exception as e:
            print(f"⚠️ Encoding {encoding} failed: {e}")
            continue
    raise ValueError(f"Could not load pincodes from {csv_file}")


def get_csv_summary(csv_file: str) -> Dict:
    try:
        pincodes = load_pincodes(csv_file)
        agg: Dict = defaultdict(lambda: {"count": 0, "districts": defaultdict(int)})
        for info in pincodes.values():
            st   = info["state"]    or "Unknown"
            dist = info["district"] or "Unknown"
            agg[st]["count"] += 1
            agg[st]["districts"][dist] += 1
        return {
            "total_pincodes": len(pincodes),
            "states": {
                st: {
                    "count": data["count"],
                    "districts": [
                        {"name": d, "count": c}
                        for d, c in sorted(data["districts"].items())
                    ],
                }
                for st, data in sorted(agg.items())
            },
        }
    except Exception as e:
        return {"error": str(e)}


def filter_pincodes(
    pincodes: "OrderedDict[str, Dict]",
    selected_states: List[str],
    selected_districts: List[str],
    mode: str,
) -> "OrderedDict[str, Dict]":
    if mode == "all" or (not selected_states and not selected_districts):
        return pincodes
    state_set = set(selected_states)
    dist_set  = set(selected_districts)
    out: OrderedDict = OrderedDict()
    for pc, info in pincodes.items():
        if mode == "states"    and info["state"]    in state_set:  out[pc] = info
        elif mode == "districts" and info["district"] in dist_set: out[pc] = info
        elif mode == "custom"  and (
            info["state"] in state_set or info["district"] in dist_set
        ): out[pc] = info
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  CHROME DRIVER
# ══════════════════════════════════════════════════════════════════════════════

def setup_driver(cfg: Dict, instance_id: int):
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service

    opts = Options()
    opts.binary_location = cfg["chrome_binary"]
    for arg in (
        "--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu",
        "--disable-software-rasterizer",
        "--disable-blink-features=AutomationControlled",
        "--log-level=3", "--disable-logging",
    ):
        opts.add_argument(arg)
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)

    profile = tempfile.mkdtemp(prefix=f"chrome_scraper_{instance_id}_")
    import atexit
    atexit.register(shutil.rmtree, profile, True)
    opts.add_argument(f"--user-data-dir={profile}")

    return webdriver.Chrome(service=Service(cfg["chromedriver_path"]), options=opts)


# ══════════════════════════════════════════════════════════════════════════════
#  BUSINESS EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_businesses(driver, pincode: str, loc: Dict, cfg: Dict) -> List[Dict]:
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.support.ui import WebDriverWait

    bt = cfg["business_type"]
    businesses, seen = [], set()
    no_new = 0
    max_no_new = cfg["max_scrolls"]

    try:
        WebDriverWait(driver, cfg["page_load_delay"]).until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, '[role="feed"], [role="main"]')
            )
        )
    except Exception:
        pass

    while no_new < max_no_new and len(businesses) < 200:
        if _stop_event.is_set():
            break

        cards = driver.find_elements(
            By.CSS_SELECTOR, '[role="article"], .Nv2PK, .bfdHYd'
        )
        prev = len(businesses)

        for card in cards:
            try:
                if not card.is_displayed():
                    continue
                text = card.text
                if not text or len(text) <= 10:
                    continue
                lines = [l.strip() for l in text.split("\n") if l.strip()]
                if not lines:
                    continue
                name = lines[0]
                if len(name) <= 2 or name in seen:
                    continue
                if any(x in name for x in ("Collapse","Results","Directions","Sponsored")):
                    continue
                seen.add(name)

                phone = None
                m = re.search(r"(?:\+91[\s\-]?)?[6-9]\d{9}", text)
                if m:
                    phone = re.sub(r"[^\d+]", "", m.group())

                rating = reviews = None
                m = re.search(r"(\d+\.?\d*)\s*\((\d[\d,]*)\)", text)
                if m:
                    rating  = m.group(1)
                    reviews = m.group(2).replace(",", "")

                address = None
                for line in lines[1:]:
                    if "·" in line and len(line) > 10:
                        parts = [p.strip() for p in line.split("·") if p.strip()]
                        address = parts[-1] if parts else None
                        break
                if not address:
                    for line in lines[1:]:
                        if len(line) > 20 and "," in line and "www." not in line.lower():
                            address = line
                            break

                website = None
                for line in lines:
                    ll = line.lower()
                    if (
                        "." in line and " " not in line and len(line) > 5
                        and any(x in ll for x in ("www.", "http", ".com", ".in", ".org"))
                    ):
                        website = line
                        break

                businesses.append({
                    "name":          name,
                    "category":      lines[1] if len(lines) > 1 else "",
                    "rating":        rating,
                    "reviews":       reviews,
                    "phone":         phone,
                    "address":       address,
                    "website":       website,
                    "pincode":       pincode,
                    "district":      loc.get("district", ""),
                    "state":         loc.get("state", ""),
                    "business_type": bt,
                    "search_query":  cfg.get("_search_query", bt),
                })
            except Exception:
                continue

        if len(businesses) > prev:
            no_new = 0
            _log(f"  📊 {pincode}: {len(businesses)} found …")
        else:
            no_new += 1

        try:
            feed = driver.find_element(By.CSS_SELECTOR, '[role="feed"]')
            driver.execute_script(
                "arguments[0].scrollTop = arguments[0].scrollHeight", feed
            )
        except Exception:
            driver.execute_script("window.scrollBy(0, window.innerHeight)")
        time.sleep(cfg["scroll_delay"])

    return businesses


# ══════════════════════════════════════════════════════════════════════════════
#  SHARED SCRAPER HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _scrape_setup(cfg: Dict, run_id: int, run_record: Dict) -> Tuple:
    """Load pincodes, apply filters, setup paths for this run"""
    global _current_run_id
    
    _current_run_id = run_id
    
    json_path = Path(run_record["json_path"])
    csv_path = Path(run_record["csv_path"])
    
    # For this run, start fresh (no resume across runs)
    all_results: Dict = {}
    
    # Load + filter pincodes
    pincode_info = load_pincodes(cfg["csv_file"])
    orig = len(pincode_info)
    pincode_info = filter_pincodes(
        pincode_info,
        cfg.get("selected_states", []),
        cfg.get("selected_districts", []),
        cfg.get("filter_mode", "all"),
    )
    filtered = len(pincode_info)
    if filtered < orig:
        _log(f"🔍 Filtered {orig:,} → {filtered:,} pincodes")
    
    # Update run record with total pincodes
    update_run_status(run_id, "running", total_pincodes=filtered)
    
    _stats["total_pincodes"] = orig
    _stats["filtered_pincodes"] = filtered
    _stats["pincodes_done"] = 0
    _stats["businesses_found"] = 0
    _push_stats()
    
    _log(f"   Total {orig:,} | Filtered {filtered:,}")
    
    pending = [(p, pincode_info[p]) for p in pincode_info]
    return json_path, csv_path, all_results, pending


def _save_result(
    pincode: str,
    info: Dict,
    biz: List[Dict],
    cfg: Dict,
    all_results: Dict,
    json_path: Path,
    csv_path: Path,
    run_id: int,
) -> None:
    """Persist one pincode's results for this run"""
    entry = {
        "pincode":       pincode,
        "district":      info["district"],
        "state":         info["state"],
        "business_type": cfg["business_type"],
        "timestamp":     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "count":         len(biz),
        "businesses":    biz,
    }
    with _write_lock:
        all_results[pincode] = entry
        tmp = json_path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        tmp.replace(json_path)
        if cfg.get("auto_export_csv") and biz:
            export_to_csv(biz, csv_path, run_id, mode="append")
    
    if cfg.get("postgres_enabled") and biz:
        _db_save_businesses(biz, run_id, cfg)
    
    with _write_lock:
        _stats["pincodes_done"] = len(all_results)
        _stats["businesses_found"] = sum(d.get("count", 0) for d in all_results.values())
        # Update run record
        update_run_status(run_id, "running", businesses_found=_stats["businesses_found"])
    _push_stats()


def _scrape_finish(run_id: int, all_results: Dict, json_path: Path, csv_path: Path, cfg: Dict) -> None:
    total = sum(d.get("count", 0) for d in all_results.values())
    status = "stopped" if _stop_event.is_set() else "completed"
    update_run_status(run_id, status, businesses_found=total)
    
    _log(f'\n{"=" * 55}')
    _log(f"🎉 RUN #{run_id} DONE | pincodes: {len(all_results):,} | businesses: {total:,}")
    _log(f"💾 JSON: {json_path}")
    if cfg.get("auto_export_csv"):
        _log(f"💾 CSV:  {csv_path}")
    _log_queue.put({"type": "done"})
    _stats["running"] = False
    _stats["current_run_id"] = None


# ══════════════════════════════════════════════════════════════════════════════
#  SINGLE-INSTANCE SCRAPER
# ══════════════════════════════════════════════════════════════════════════════

def _run_single(cfg: Dict, run_id: int, run_record: Dict) -> None:
    try:
        json_path, csv_path, all_results, pending = _scrape_setup(cfg, run_id, run_record)
    except Exception as e:
        _log(f"❌ Setup failed: {e}")
        _log_queue.put({"type": "done"})
        _stats["running"] = False
        update_run_status(run_id, "failed")
        return

    if not pending:
        _log("🎉 No pincodes to scrape!")
        _scrape_finish(run_id, all_results, json_path, csv_path, cfg)
        return

    sq = cfg["_search_query"]
    _log(f"\n🔄 Single-instance — {cfg['business_type']}")
    driver = setup_driver(cfg, 1)
    try:
        for i, (pincode, info) in enumerate(pending, 1):
            if _stop_event.is_set():
                _log("⚠️ Stopped by user.")
                break
            _log(f'{"=" * 55}')
            _log(f"📍 [{i}/{len(pending)}]  {pincode}  —  {info['district']}, {info['state']}")

            driver.get(
                f"https://www.google.co.in/maps/search/{sq.replace(' ', '+')}+in+{pincode}/"
            )
            biz = extract_businesses(driver, pincode, info, cfg)
            _save_result(pincode, info, biz, cfg, all_results, json_path, csv_path, run_id)
            _log(f"✅ {pincode}: {len(biz)} entries saved")
            if biz:
                _log(f"   Sample: {biz[0]['name']}")
            if i < len(pending) and not _stop_event.is_set():
                time.sleep(cfg["delay_between_pincodes"])
    except Exception as e:
        import traceback
        _log(f"❌ Fatal: {e}")
        _log(traceback.format_exc())
        update_run_status(run_id, "failed")
    finally:
        driver.quit()
        _scrape_finish(run_id, all_results, json_path, csv_path, cfg)


# ══════════════════════════════════════════════════════════════════════════════
#  MULTI-INSTANCE SCRAPER
# ══════════════════════════════════════════════════════════════════════════════

def _run_multi(cfg: Dict, run_id: int, run_record: Dict) -> None:
    try:
        json_path, csv_path, all_results, pending = _scrape_setup(cfg, run_id, run_record)
    except Exception as e:
        _log(f"❌ Setup failed: {e}")
        _log_queue.put({"type": "done"})
        _stats["running"] = False
        update_run_status(run_id, "failed")
        return

    if not pending:
        _log("🎉 No pincodes to scrape!")
        _scrape_finish(run_id, all_results, json_path, csv_path, cfg)
        return

    num = min(cfg.get("max_workers", 3), len(pending), 10)
    sq  = cfg["_search_query"]
    _log(f"🚀 Multi-instance — {num} workers — {cfg['business_type']}")

    def worker(inst_id: int, chunk: List[Tuple]) -> None:
        driver = None
        try:
            driver = setup_driver(cfg, inst_id)
            for pincode, info in chunk:
                if _stop_event.is_set():
                    break
                driver.get(
                    f"https://www.google.co.in/maps/search/{sq.replace(' ', '+')}+in+{pincode}/"
                )
                biz = extract_businesses(driver, pincode, info, cfg)
                _save_result(pincode, info, biz, cfg, all_results, json_path, csv_path, run_id)
                _log(f"✅ Worker {inst_id}: {pincode} → {len(biz)} results")
                if not _stop_event.is_set():
                    time.sleep(cfg["delay_between_pincodes"])
        except Exception as e:
            _log(f"❌ Worker {inst_id}: {e}")
        finally:
            if driver:
                driver.quit()

    chunk_size = max(1, (len(pending) + num - 1) // num)
    chunks  = [pending[i : i + chunk_size] for i in range(0, len(pending), chunk_size)]
    threads = [
        threading.Thread(target=worker, args=(i + 1, c), daemon=True)
        for i, c in enumerate(chunks)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    _scrape_finish(run_id, all_results, json_path, csv_path, cfg)


# ══════════════════════════════════════════════════════════════════════════════
#  SCRAPER ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def _run_scraper(cfg: Dict) -> None:
    global _current_run_id
    
    _stats["running"] = True
    _stop_event.clear()
    cfg["_search_query"] = cfg.get("search_query") or cfg["business_type"]
    
    # Create a new run record
    run_id, run_record = create_run_record(cfg)
    _stats["current_run_id"] = run_id
    _current_run_id = run_id
    
    _log(f"🔍 RUN #{run_id} - Business: {cfg['business_type']}")
    _log(f"🔍 Query: {cfg['_search_query']}")
    _log(f"🔍 Mode: {cfg.get('mode', 'single')}")
    _log(f"📁 Output: {run_record['folder']}")

    if cfg.get("postgres_enabled"):
        init_db(cfg)

    try:
        if cfg.get("mode") == "multi":
            _run_multi(cfg, run_id, run_record)
        else:
            _run_single(cfg, run_id, run_record)
    except Exception as e:
        _log(f"❌ Unhandled: {e}")
        _log_queue.put({"type": "done"})
        _stats["running"] = False
        update_run_status(run_id, "failed")


# ══════════════════════════════════════════════════════════════════════════════
#  FLASK ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/csv_summary", methods=["POST"])
def csv_summary():
    data = request.get_json(force=True) or {}
    csv_file = data.get("csv_file", "").strip()
    if not csv_file:
        return jsonify({"error": "No CSV file provided"})
    return jsonify(get_csv_summary(csv_file))


@app.route("/load_config")
def load_config_route():
    return jsonify(load_config())


@app.route("/save_config", methods=["POST"])
def save_config_route():
    cfg = request.get_json(force=True) or {}
    save_config(cfg)
    return jsonify({"status": "ok"})


@app.route("/get_runs")
def get_runs():
    """Get all scraping runs"""
    return jsonify({"runs": get_all_runs()})


@app.route("/get_run/<int:run_id>")
def get_run(run_id):
    """Get details of a specific run"""
    run = get_run_by_id(run_id)
    if not run:
        return jsonify({"error": "Run not found"}), 404
    
    # Load results from the run's JSON file
    results = {}
    json_path = Path(run["json_path"])
    if json_path.exists():
        try:
            results = json.loads(json_path.read_text(encoding="utf-8"))
        except:
            pass
    
    return jsonify({
        "run": run,
        "results": results,
        "total_businesses": sum(d.get("count", 0) for d in results.values())
    })


@app.route("/delete_run/<int:run_id>", methods=["DELETE"])
def delete_run_route(run_id):
    """Delete a run and its files"""
    if delete_run(run_id):
        return jsonify({"status": "ok"})
    return jsonify({"error": "Run not found"}), 404


@app.route("/export_run/<int:run_id>")
def export_run(run_id):
    """Export a run's CSV file"""
    run = get_run_by_id(run_id)
    if not run:
        return jsonify({"error": "Run not found"}), 404
    
    csv_path = Path(run["csv_path"])
    if not csv_path.exists():
        return jsonify({"error": "CSV file not found"}), 404
    
    return send_file(str(csv_path), as_attachment=True, download_name=f"run_{run_id}_{run['business_type']}.csv")


@app.route("/start", methods=["POST"])
def start():
    global _scraper_thread
    if _stats.get("running"):
        return jsonify({"status": "error", "error": "Already running"})

    cfg = request.get_json(force=True) or {}

    missing = [k for k in ("business_type", "csv_file", "chrome_binary", "chromedriver_path") if not cfg.get(k)]
    if missing:
        return jsonify({"status": "error", "error": f"Missing fields: {', '.join(missing)}"})

    try:
        all_pincodes = load_pincodes(cfg["csv_file"])
        filtered = filter_pincodes(
            all_pincodes,
            cfg.get("selected_states", []),
            cfg.get("selected_districts", []),
            cfg.get("filter_mode", "all"),
        )
        if not filtered:
            return jsonify({"status": "error", "error": "No pincodes match the active filter"})
        _log(f"✅ CSV valid: {len(all_pincodes):,} total | {len(filtered):,} filtered")
    except Exception as e:
        return jsonify({"status": "error", "error": f"CSV error: {e}"})

    save_config(cfg)
    _scraper_thread = threading.Thread(target=_run_scraper, args=(cfg,), daemon=True)
    _scraper_thread.start()
    return jsonify({"status": "ok"})


@app.route("/stop", methods=["POST"])
def stop():
    _stop_event.set()
    return jsonify({"status": "ok"})


@app.route("/stats")
def stats_route():
    return jsonify(_stats)


@app.route("/stream")
def stream():
    def gen():
        try:
            while True:
                try:
                    msg = _log_queue.get(timeout=25)
                    yield f"data: {json.dumps(msg)}\n\n"
                    if msg.get("type") == "done":
                        break
                except queue.Empty:
                    yield 'data: {"type":"ping"}\n\n'
        except GeneratorExit:
            pass
    return Response(
        gen(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ══════════════════════════════════════════════════════════════════════════════
#  HTML TEMPLATE (with Runs History)
# ══════════════════════════════════════════════════════════════════════════════

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Maps Scraper | Persistent Runs</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,400;14..32,500;14..32,600;14..32,700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet"/>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #0b0a09; font-family: 'Inter', sans-serif; padding: 32px 40px; color: #ece8e0; }
  .bento-grid { display: grid; grid-template-columns: repeat(12, 1fr); gap: 28px; max-width: 1600px; margin: 0 auto; }
  .solid-card { background: #171614; border-radius: 32px; border: 1px solid #2a2722; box-shadow: 0 12px 28px -8px rgba(0,0,0,0.6); overflow: hidden; }
  .header-bento { grid-column: span 12; display: flex; justify-content: space-between; align-items: center; background: #171614; border-radius: 32px; padding: 0.9rem 2rem; border: 1px solid #2a2722; }
  .logo-area { display: flex; align-items: center; gap: 14px; }
  .solid-icon { width: 48px; height: 48px; background: #25221c; border-radius: 24px; display: flex; align-items: center; justify-content: center; font-size: 1.7rem; color: #e0b285; border: 1px solid #3a3530; }
  h1 { font-weight: 700; font-size: 1.6rem; letter-spacing: -0.02em; color: #f2ede4; }
  h1 span { color: #e0b285; }
  .badge-solid { background: #25221c; border-radius: 40px; padding: 6px 20px; font-size: 0.75rem; font-weight: 600; color: #e0b285; border: 1px solid #3a3530; }
  .stats-bento { grid-column: span 3; display: flex; flex-direction: column; gap: 20px; }
  .stat-block { background: #171614; border-radius: 28px; padding: 1.4rem 1.2rem; border: 1px solid #2a2722; }
  .stat-label { font-size: 0.7rem; text-transform: uppercase; font-weight: 700; letter-spacing: 0.08em; color: #a6a094; margin-bottom: 10px; }
  .stat-value { font-family: 'JetBrains Mono', monospace; font-size: 2.2rem; font-weight: 700; color: #e0b285; line-height: 1.1; }
  .stat-value.accent-green { color: #9fc088; }
  .progress-solid { margin-top: 16px; background: #2a2722; border-radius: 40px; height: 6px; overflow: hidden; }
  .progress-fill { width: 0%; height: 100%; background: #e0b285; border-radius: 40px; transition: width 0.35s ease; }
  .status-indicator { display: flex; align-items: center; gap: 10px; margin-top: 8px; }
  .status-led { width: 10px; height: 10px; border-radius: 50%; background: #4a4540; }
  .status-led.running { background: #9fc088; box-shadow: 0 0 8px #b5d693; }
  .status-led.stopped { background: #e08f7c; }
  #status-text { font-weight: 600; font-size: 0.85rem; color: #cec9c0; }
  .config-bento { grid-column: span 9; padding: 1.8rem 2rem; display: flex; flex-direction: column; gap: 28px; }
  .filter-bento { grid-column: span 12; padding: 1.6rem 1.8rem; display: flex; flex-direction: column; gap: 24px; }
  .log-bento { grid-column: span 6; display: flex; flex-direction: column; max-height: 500px; }
  .runs-bento { grid-column: span 6; display: flex; flex-direction: column; max-height: 500px; }
  .field-group { display: flex; flex-direction: column; gap: 8px; }
  .field-group label { font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: #b6b0a2; }
  input, select, textarea { background: #0f0e0c; border: 1px solid #2e2a25; border-radius: 20px; padding: 12px 16px; color: #f0ebe2; font-family: 'Inter', monospace; font-size: 0.85rem; transition: 0.2s; }
  input:focus, select:focus, textarea:focus { outline: none; border-color: #e0b285; box-shadow: 0 0 0 3px rgba(224,178,133,0.15); background: #141210; }
  .double-col { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
  .triple-col { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; }
  .mode-selector { display: flex; gap: 12px; background: #0f0e0c; border-radius: 48px; padding: 6px; border: 1px solid #2e2a25; }
  .mode-opt { flex: 1; text-align: center; padding: 10px 0; border-radius: 40px; cursor: pointer; font-weight: 600; font-size: 0.8rem; transition: 0.2s; color: #a6a094; }
  .mode-opt.active { background: #25221c; color: #e0b285; box-shadow: 0 2px 6px rgba(0,0,0,0.3); border: 1px solid #3e3a33; }
  .btn-premium { background: #25221c; border: 1px solid #3a3530; padding: 12px 0; border-radius: 40px; font-weight: 700; font-size: 0.85rem; color: #ece4d9; cursor: pointer; transition: 0.15s; text-transform: uppercase; letter-spacing: 0.03em; }
  .btn-premium.primary { background: #2e2a24; border: 1px solid #e0b285; color: #e0b285; }
  .btn-premium.primary:hover:not(:disabled) { background: #3d382f; transform: scale(0.98); }
  .btn-premium.danger { background: #251d1b; border-color: #7e5a4f; color: #e08f7c; }
  button:disabled { opacity: 0.4; cursor: not-allowed; }
  .filter-scroll { max-height: 240px; overflow-y: auto; display: flex; flex-direction: column; gap: 10px; margin-top: 12px; padding-right: 6px; }
  .filter-row { display: flex; align-items: center; gap: 14px; background: #12110e; border-radius: 48px; padding: 10px 18px; font-size: 0.8rem; cursor: pointer; border: 1px solid #2a2722; }
  .filter-row:hover { background: #1b1916; border-color: #4a4540; }
  .filter-row input { width: 18px; height: 18px; margin: 0; accent-color: #e0b285; }
  .filter-row label { flex: 1; cursor: pointer; font-weight: 500; color: #ddd6cc; }
  .filter-count { color: #8e887c; font-family: monospace; font-size: 0.75rem; }
  .section-title-sm { font-weight: 700; font-size: 0.75rem; letter-spacing: 0.08em; color: #e0b285; padding-left: 10px; border-left: 3px solid #e0b285; margin-bottom: 16px; text-transform: uppercase; }
  .log-container, .runs-container { padding: 1.2rem 1.6rem; font-family: 'JetBrains Mono', monospace; font-size: 0.73rem; overflow-y: auto; height: 380px; background: #0f0e0c; }
  .log-line { display: block; padding: 6px 0; border-bottom: 1px solid #22201c; white-space: pre-wrap; color: #cdc6bb; }
  .run-item { background: #12110e; border: 1px solid #2a2722; border-radius: 16px; padding: 12px 16px; margin-bottom: 12px; cursor: pointer; transition: 0.1s; }
  .run-item:hover { background: #1b1916; border-color: #e0b285; }
  .run-item.completed { border-left: 3px solid #9fc088; }
  .run-item.running { border-left: 3px solid #e0b285; }
  .run-item.stopped { border-left: 3px solid #e08f7c; }
  .run-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
  .run-id { font-weight: 700; color: #e0b285; font-family: monospace; }
  .run-status { font-size: 0.7rem; padding: 2px 8px; border-radius: 12px; background: #25221c; }
  .run-status.completed { color: #9fc088; }
  .run-status.running { color: #e0b285; }
  .run-status.stopped { color: #e08f7c; }
  .run-details { font-size: 0.7rem; color: #8e887c; display: flex; gap: 16px; flex-wrap: wrap; }
  .run-actions { display: flex; gap: 8px; margin-top: 8px; }
  .btn-small { background: #25221c; border: 1px solid #3a3530; padding: 4px 12px; border-radius: 20px; font-size: 0.7rem; cursor: pointer; color: #ece4d9; }
  .btn-small:hover { background: #3d382f; }
  ::-webkit-scrollbar { width: 5px; }
  ::-webkit-scrollbar-track { background: #1f1d1a; border-radius: 8px; }
  ::-webkit-scrollbar-thumb { background: #4a4540; border-radius: 8px; }
  @media (max-width: 1200px) {
    body { padding: 24px 28px; }
    .stats-bento { grid-column: span 12; flex-direction: row; flex-wrap: wrap; }
    .stats-bento .stat-block { flex: 1; min-width: 170px; }
    .config-bento { grid-column: span 12; }
    .log-bento, .runs-bento { grid-column: span 12; }
  }
</style>
</head>
<body>
<div class="bento-grid">
  <div class="header-bento">
    <div class="logo-area"><div class="solid-icon">🗺️</div><h1>MAPS <span>SCRAPER</span></h1></div>
    <div class="badge-solid">✦ Persistent Runs · History ✦</div>
  </div>

  <div class="stats-bento">
    <div class="stat-block"><div class="stat-label">PINCODES DONE</div><div class="stat-value" id="stat-done">0</div><div class="progress-solid"><div class="progress-fill" id="progress-fill-bar"></div></div></div>
    <div class="stat-block"><div class="stat-label">TOTAL PINCODES</div><div class="stat-value" id="stat-total">—</div></div>
    <div class="stat-block"><div class="stat-label">FILTERED (ACTIVE)</div><div class="stat-value accent-green" id="stat-filtered">—</div></div>
    <div class="stat-block"><div class="stat-label">BUSINESSES FOUND</div><div class="stat-value accent-green" id="stat-biz">0</div></div>
    <div class="stat-block"><div class="stat-label">CURRENT RUN</div><div class="stat-value" id="current-run">—</div><div class="status-indicator"><div class="status-led" id="status-led"></div><span id="status-text">Idle</span></div></div>
  </div>

  <div class="config-bento solid-card">
    <div><div class="section-title-sm">⚙️ TARGET & DATA PATHS</div>
      <div class="field-group" style="margin-bottom: 14px;"><label>Business Type *</label><input type="text" id="business_type" placeholder="e.g., hospital, clinic"/></div>
      <div class="field-group" style="margin-bottom: 14px;"><label>Custom Search Query</label><input type="text" id="search_query" placeholder="Leave blank → uses Business Type"/></div>
      <div class="double-col">
        <div class="field-group"><label>Pincode CSV File</label><input type="text" id="csv_file"/><button class="btn-premium" style="margin-top:12px; padding:8px 0; font-size:0.7rem;" onclick="loadCSVSummary()">⟳ Load Summary</button></div>
        <div class="field-group"><label>Output Directory</label><input type="text" id="output_dir" placeholder="D:/GSTCSV"/></div>
      </div>
      <div class="double-col">
        <div class="field-group"><label>Chrome Binary</label><input type="text" id="chrome_binary"/></div>
        <div class="field-group"><label>ChromeDriver Path</label><input type="text" id="chromedriver_path"/></div>
      </div>
    </div>
    
    <div><div class="section-title-sm">🗄️ DATABASE & EXPORT</div>
      <div class="triple-col">
        <div class="field-group"><label><input type="checkbox" id="postgres_enabled"> Enable PostgreSQL</label></div>
        <div class="field-group"><label><input type="checkbox" id="auto_export_csv" checked> Auto-Export CSV</label></div>
      </div>
      <div id="postgres-config" style="display:none; margin-top:16px;">
        <div class="double-col">
          <div class="field-group"><label>PostgreSQL Host</label><input type="text" id="pg_host" placeholder="localhost"/></div>
          <div class="field-group"><label>Port</label><input type="text" id="pg_port" placeholder="5432"/></div>
          <div class="field-group"><label>Database</label><input type="text" id="pg_db" placeholder="business_scraper"/></div>
          <div class="field-group"><label>Username</label><input type="text" id="pg_user" placeholder="postgres"/></div>
          <div class="field-group"><label>Password</label><input type="password" id="pg_password"/></div>
        </div>
      </div>
    </div>
    
    <div><div class="section-title-sm">⏱️ PERFORMANCE</div>
      <div class="double-col">
        <div class="field-group"><label>Scroll Delay (s)</label><input type="number" id="scroll_delay" value="2" step="0.5"/></div>
        <div class="field-group"><label>Max Scrolls</label><input type="number" id="max_scrolls" value="15"/></div>
        <div class="field-group"><label>Page Load Delay (s)</label><input type="number" id="page_load_delay" value="5"/></div>
        <div class="field-group"><label>Pincode Delay (s)</label><input type="number" id="delay_between_pincodes" value="2"/></div>
      </div>
    </div>
    
    <div><div class="section-title-sm">⚡ MODE</div>
      <div class="mode-selector" id="modeSwitchWrapper">
        <div class="mode-opt" data-mode="single">SINGLE</div>
        <div class="mode-opt" data-mode="multi">MULTI</div>
      </div>
      <div id="workers-wrap" style="margin-top:18px; display:none;"><div class="field-group"><label>Workers (max 10)</label><input type="number" id="max_workers" value="5" min="2" max="10"/></div></div>
    </div>
    
    <div class="double-col" style="margin-top: 4px;">
      <button class="btn-premium primary" id="btn-start" onclick="startScraper()">▶ START NEW RUN</button>
      <button class="btn-premium danger" id="btn-stop" onclick="stopScraper()" disabled>■ STOP</button>
    </div>
  </div>

  <div class="filter-bento solid-card">
    <div><div class="section-title-sm">🎯 FILTER BY REGION</div>
      <div class="mode-selector" id="filterModeToggle">
        <div class="mode-opt" data-filtermode="all">ALL</div>
        <div class="mode-opt" data-filtermode="states">STATE</div>
        <div class="mode-opt" data-filtermode="districts">DISTRICT</div>
      </div>
    </div>
    <div id="states-filter-container" style="display:none;"><div class="section-title-sm">🏛️ STATES <span id="selected-states-count"></span></div><div class="filter-scroll" id="state-list"></div><div style="display:flex; gap:12px; margin-top:16px;"><button class="btn-premium" style="padding:8px 0; font-size:0.75rem;" onclick="selectAllStates()">✓ All</button><button class="btn-premium danger" style="padding:8px 0; font-size:0.75rem;" onclick="clearAllStates()">✗ Clear</button></div></div>
    <div id="districts-filter-container" style="display:none;"><div class="section-title-sm">🏙️ DISTRICTS <span id="selected-districts-count"></span></div><div class="filter-scroll" id="district-list"></div><div style="display:flex; gap:12px; margin-top:16px;"><button class="btn-premium" style="padding:8px 0; font-size:0.75rem;" onclick="selectAllDistricts()">✓ All</button><button class="btn-premium danger" style="padding:8px 0; font-size:0.75rem;" onclick="clearAllDistricts()">✗ Clear</button></div></div>
  </div>

  <div class="log-bento solid-card">
    <div style="padding:1rem 1.8rem; border-bottom:1px solid #2a2722; display:flex; justify-content:space-between;"><span class="section-title-sm" style="margin-bottom:0;">📡 LIVE CONSOLE</span><div><button class="btn-premium" style="padding:6px 20px; font-size:0.7rem;" onclick="clearLog()">Clear</button></div></div>
    <div id="log" class="log-container"><span class="log-line">◆ Persistent Runs Ready - Each run saved separately</span></div>
  </div>

  <div class="runs-bento solid-card">
    <div style="padding:1rem 1.8rem; border-bottom:1px solid #2a2722;"><span class="section-title-sm" style="margin-bottom:0;">📚 RUN HISTORY</span></div>
    <div id="runs-list" class="runs-container"><span class="log-line">Loading runs...</span></div>
  </div>
</div>

<script>
  let evtSource = null;
  let csvSummary = null;
  let selectedStates = new Set();
  let selectedDistricts = new Set();
  let currentFilterMode = "all";
  let currentMode = "single";

  function appendLog(msg) {
    const logDiv = document.getElementById('log');
    const line = document.createElement('span');
    line.className = 'log-line';
    if(msg.includes("✅")) line.style.color = "#9fc088";
    else if(msg.includes("❌")) line.style.color = "#e08f7c";
    else if(msg.includes("⚠️")) line.style.color = "#e0b285";
    line.textContent = msg;
    logDiv.appendChild(line);
    line.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }
  function clearLog() { document.getElementById('log').innerHTML = '<span class="log-line">◆ Console cleared.</span>'; }
  
  function setStatus(status) {
    const led = document.getElementById('status-led');
    const txt = document.getElementById('status-text');
    if(status === 'running') { led.className = "status-led running"; txt.innerText = "RUNNING"; }
    else if(status === 'stopped') { led.className = "status-led stopped"; txt.innerText = "STOPPED"; }
    else { led.className = "status-led"; txt.innerText = "IDLE"; }
  }
  
  function updateStats(data) {
    if(data.pincodes_done !== undefined) document.getElementById('stat-done').innerText = data.pincodes_done;
    if(data.businesses_found !== undefined) document.getElementById('stat-biz').innerText = data.businesses_found;
    if(data.total_pincodes > 0) {
      document.getElementById('stat-total').innerText = data.total_pincodes;
      let percent = (data.pincodes_done / data.total_pincodes) * 100;
      document.getElementById('progress-fill-bar').style.width = percent + "%";
    }
    if(data.filtered_pincodes !== undefined) document.getElementById('stat-filtered').innerText = data.filtered_pincodes;
    if(data.current_run_id !== undefined && data.current_run_id !== null) {
      document.getElementById('current-run').innerText = "#" + data.current_run_id;
    }
  }
  
  async function loadRuns() {
    try {
      const resp = await fetch('/get_runs');
      const data = await resp.json();
      const container = document.getElementById('runs-list');
      if(data.runs.length === 0) {
        container.innerHTML = '<span class="log-line">No runs yet. Start a new run!</span>';
        return;
      }
      container.innerHTML = data.runs.map(run => `
        <div class="run-item ${run.status}" onclick="viewRun(${run.run_id})">
          <div class="run-header">
            <span class="run-id">Run #${run.run_id}</span>
            <span class="run-status ${run.status}">${run.status.toUpperCase()}</span>
          </div>
          <div class="run-details">
            <span>📌 ${run.business_type}</span>
            <span>📍 ${run.total_pincodes || 0} pincodes</span>
            <span>🏢 ${run.businesses_found || 0} businesses</span>
            <span>📅 ${new Date(run.timestamp).toLocaleString()}</span>
          </div>
          <div class="run-actions">
            <button class="btn-small" onclick="event.stopPropagation(); exportRun(${run.run_id})">📥 Export CSV</button>
            <button class="btn-small" onclick="event.stopPropagation(); deleteRun(${run.run_id})">🗑️ Delete</button>
          </div>
        </div>
      `).join('');
    } catch(e) { console.error(e); }
  }
  
  async function viewRun(runId) {
    try {
      const resp = await fetch(`/get_run/${runId}`);
      const data = await resp.json();
      if(data.error) { appendLog(`❌ ${data.error}`); return; }
      appendLog(`📊 Run #${runId}: ${data.total_businesses} businesses from ${Object.keys(data.results).length} pincodes`);
      // Show summary in console
      let summary = Object.entries(data.results).slice(0, 5).map(([pc, info]) => `  ${pc}: ${info.count} businesses`).join('\n');
      if(Object.keys(data.results).length > 5) summary += `\n  ... and ${Object.keys(data.results).length - 5} more`;
      appendLog(`📋 Sample:\n${summary}`);
    } catch(e) { appendLog(`❌ ${e.message}`); }
  }
  
  async function exportRun(runId) {
    window.location.href = `/export_run/${runId}`;
  }
  
  async function deleteRun(runId) {
    if(confirm(`Delete Run #${runId} and all its files?`)) {
      try {
        const resp = await fetch(`/delete_run/${runId}`, { method: 'DELETE' });
        if(resp.ok) {
          appendLog(`✅ Run #${runId} deleted`);
          loadRuns();
        } else {
          appendLog(`❌ Failed to delete run`);
        }
      } catch(e) { appendLog(`❌ ${e.message}`); }
    }
  }
  
  async function loadCSVSummary() {
    const csvFile = document.getElementById('csv_file').value.trim();
    if(!csvFile) { appendLog("⚠️ Enter CSV file path"); return; }
    appendLog(`📂 Loading ${csvFile} ...`);
    try {
      const resp = await fetch('/csv_summary', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({csv_file: csvFile}) });
      const data = await resp.json();
      if(data.error) { appendLog(`❌ ${data.error}`); return; }
      csvSummary = data;
      appendLog(`✅ ${data.total_pincodes} pincodes from ${Object.keys(data.states).length} states`);
      renderStateList(data.states);
      renderDistrictList(data.states);
    } catch(e) { appendLog(`❌ ${e.message}`); }
  }

  function renderStateList(statesMap) {
    const container = document.getElementById('state-list');
    container.innerHTML = '';
    for(let [stateName, stateData] of Object.entries(statesMap).sort()) {
      const row = document.createElement('div'); row.className = 'filter-row';
      row.innerHTML = `<input type="checkbox" value="${stateName}"><label>${stateName}</label><span class="filter-count">(${stateData.count})</span>`;
      const cb = row.querySelector('input');
      cb.addEventListener('change', () => {
        if(cb.checked) selectedStates.add(stateName); else selectedStates.delete(stateName);
        document.getElementById('selected-states-count').innerText = `(${selectedStates.size})`;
      });
      container.appendChild(row);
    }
  }

  function renderDistrictList(statesMap) {
    const container = document.getElementById('district-list');
    container.innerHTML = '';
    let allDistricts = [];
    for(let [st, val] of Object.entries(statesMap))
      for(let d of val.districts) allDistricts.push({state: st, name: d.name, count: d.count});
    allDistricts.sort((a,b)=>a.name.localeCompare(b.name));
    for(let d of allDistricts) {
      const row = document.createElement('div'); row.className = 'filter-row';
      row.innerHTML = `<input type="checkbox" value="${d.name}"><label>${d.name} <span style="opacity:0.7;">(${d.state})</span></label><span class="filter-count">(${d.count})</span>`;
      const cb = row.querySelector('input');
      cb.addEventListener('change', () => {
        if(cb.checked) selectedDistricts.add(d.name); else selectedDistricts.delete(d.name);
        document.getElementById('selected-districts-count').innerText = `(${selectedDistricts.size})`;
      });
      container.appendChild(row);
    }
  }

  function selectAllStates() { document.querySelectorAll('#state-list input').forEach(cb => { cb.checked=true; selectedStates.add(cb.value); }); document.getElementById('selected-states-count').innerText = `(${selectedStates.size})`; }
  function clearAllStates() { document.querySelectorAll('#state-list input').forEach(cb => { cb.checked=false; selectedStates.delete(cb.value); }); document.getElementById('selected-states-count').innerText = `(0)`; }
  function selectAllDistricts() { document.querySelectorAll('#district-list input').forEach(cb => { cb.checked=true; selectedDistricts.add(cb.value); }); document.getElementById('selected-districts-count').innerText = `(${selectedDistricts.size})`; }
  function clearAllDistricts() { document.querySelectorAll('#district-list input').forEach(cb => { cb.checked=false; selectedDistricts.delete(cb.value); }); document.getElementById('selected-districts-count').innerText = `(0)`; }

  document.getElementById('postgres_enabled').addEventListener('change', (e) => { document.getElementById('postgres-config').style.display = e.target.checked ? 'block' : 'none'; });
  document.querySelectorAll('#modeSwitchWrapper .mode-opt').forEach(el => {
    el.addEventListener('click', () => {
      document.querySelectorAll('#modeSwitchWrapper .mode-opt').forEach(o=>o.classList.remove('active'));
      el.classList.add('active');
      currentMode = el.getAttribute('data-mode');
      document.getElementById('workers-wrap').style.display = currentMode === 'multi' ? 'block' : 'none';
    });
  });
  document.querySelectorAll('#filterModeToggle .mode-opt').forEach(el => {
    el.addEventListener('click', () => {
      document.querySelectorAll('#filterModeToggle .mode-opt').forEach(o=>o.classList.remove('active'));
      el.classList.add('active');
      currentFilterMode = el.getAttribute('data-filtermode');
      document.getElementById('states-filter-container').style.display = currentFilterMode === 'states' ? 'block' : 'none';
      document.getElementById('districts-filter-container').style.display = currentFilterMode === 'districts' ? 'block' : 'none';
    });
  });
  
  document.querySelector('#modeSwitchWrapper .mode-opt[data-mode="single"]').classList.add('active');
  document.querySelector('#filterModeToggle .mode-opt[data-filtermode="all"]').classList.add('active');

  async function startScraper() {
    const config = {
      business_type: document.getElementById('business_type').value.trim(),
      search_query: document.getElementById('search_query').value.trim(),
      csv_file: document.getElementById('csv_file').value.trim(),
      output_dir: document.getElementById('output_dir').value.trim() || 'D:/GSTCSV',
      chrome_binary: document.getElementById('chrome_binary').value.trim(),
      chromedriver_path: document.getElementById('chromedriver_path').value.trim(),
      scroll_delay: +document.getElementById('scroll_delay').value,
      max_scrolls: +document.getElementById('max_scrolls').value,
      page_load_delay: +document.getElementById('page_load_delay').value,
      delay_between_pincodes: +document.getElementById('delay_between_pincodes').value,
      max_workers: +document.getElementById('max_workers').value,
      mode: currentMode,
      filter_mode: currentFilterMode,
      selected_states: Array.from(selectedStates),
      selected_districts: Array.from(selectedDistricts),
      postgres_enabled: document.getElementById('postgres_enabled').checked,
      postgres_host: document.getElementById('pg_host').value,
      postgres_port: parseInt(document.getElementById('pg_port').value) || 5432,
      postgres_db: document.getElementById('pg_db').value,
      postgres_user: document.getElementById('pg_user').value,
      postgres_password: document.getElementById('pg_password').value,
      auto_export_csv: document.getElementById('auto_export_csv').checked,
    };
    if(!config.business_type) { appendLog("⚠️ Business Type required"); return; }
    if(!config.csv_file) { appendLog("⚠️ CSV File required"); return; }
    if(!config.chrome_binary || !config.chromedriver_path) { appendLog("⚠️ Chrome paths required"); return; }
    
    document.getElementById('btn-start').disabled = true;
    document.getElementById('btn-stop').disabled = false;
    setStatus('running');
    if(evtSource) evtSource.close();
    evtSource = new EventSource('/stream');
    evtSource.onmessage = e => {
      const data = JSON.parse(e.data);
      if(data.type === 'log') appendLog(data.msg);
      if(data.type === 'stats') updateStats(data);
      if(data.type === 'done') {
        setStatus('idle');
        document.getElementById('btn-start').disabled = false;
        document.getElementById('btn-stop').disabled = true;
        evtSource.close();
        loadRuns();
      }
    };
    const resp = await fetch('/start', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(config)});
    const json = await resp.json();
    if(json.status !== 'ok') appendLog("❌ " + (json.error || "start failed"));
    loadRuns();
  }

  function stopScraper() {
    fetch('/stop', {method:'POST'}).then(() => {
      setStatus('stopped');
      appendLog("⏹️ Stopping...");
      document.getElementById('btn-stop').disabled = true;
      document.getElementById('btn-start').disabled = false;
    });
  }

  setInterval(async () => {
    try{
      const resp = await fetch('/stats');
      const stats = await resp.json();
      updateStats(stats);
    } catch(e){}
  }, 2200);
  
  setInterval(() => { loadRuns(); }, 10000);

  async function loadSavedConfig() {
    try{
      const resp = await fetch('/load_config');
      const cfg = await resp.json();
      document.getElementById('business_type').value = cfg.business_type || '';
      document.getElementById('search_query').value = cfg.search_query || '';
      document.getElementById('csv_file').value = cfg.csv_file || '';
      document.getElementById('output_dir').value = cfg.output_dir || '';
      document.getElementById('chrome_binary').value = cfg.chrome_binary || '';
      document.getElementById('chromedriver_path').value = cfg.chromedriver_path || '';
      document.getElementById('scroll_delay').value = cfg.scroll_delay || 2;
      document.getElementById('max_scrolls').value = cfg.max_scrolls || 15;
      document.getElementById('page_load_delay').value = cfg.page_load_delay || 5;
      document.getElementById('delay_between_pincodes').value = cfg.delay_between_pincodes || 2;
      document.getElementById('max_workers').value = cfg.max_workers || 5;
      document.getElementById('postgres_enabled').checked = cfg.postgres_enabled || false;
      document.getElementById('pg_host').value = cfg.postgres_host || 'localhost';
      document.getElementById('pg_port').value = cfg.postgres_port || 5432;
      document.getElementById('pg_db').value = cfg.postgres_db || 'business_scraper';
      document.getElementById('pg_user').value = cfg.postgres_user || 'postgres';
      document.getElementById('pg_password').value = cfg.postgres_password || '';
      document.getElementById('auto_export_csv').checked = cfg.auto_export_csv !== false;
      if(cfg.postgres_enabled) document.getElementById('postgres-config').style.display = 'block';
      if(cfg.csv_file) loadCSVSummary();
      appendLog("✅ Config loaded");
    } catch(e) { console.log("no saved config"); }
    loadRuns();
  }
  loadSavedConfig();
</script>
</body>
</html>
"""


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def _open_browser() -> None:
    import urllib.request
    url = "http://127.0.0.1:5000"
    for _ in range(30):
        try:
            urllib.request.urlopen(url, timeout=0.5)
            break
        except Exception:
            time.sleep(0.4)
    webbrowser.open(url)


if __name__ == "__main__":
    load_config()
    threading.Thread(target=_open_browser, daemon=True).start()
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False, threaded=True)