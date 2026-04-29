"""
Instagram Reels Downloader
--------------------------
1. pip install yt-dlp
2. python download_reels.py reels_350_xxxx.json
   (or it will auto-find the JSON in the current folder)
"""

import json
import sys
import os
import glob
import subprocess
import time

# ── Config ────────────────────────────────────────────────────

LIKES_THRESHOLD = 9000       # only download reels above this
OUTPUT_DIR      = "downloads"
DELAY_BETWEEN   = 3          # seconds between downloads (avoid rate limiting)

# ── Load JSON ─────────────────────────────────────────────────

def load_json(path=None):
    if path:
        with open(path) as f:
            return json.load(f)
    # Auto-find the most recent reels JSON in current directory
    files = sorted(glob.glob("reels_*.json"), key=os.path.getmtime, reverse=True)
    if not files:
        print("❌ No reels JSON found. Pass the file as an argument:")
        print("   python download_reels.py reels_350_xxxx.json")
        sys.exit(1)
    print(f"📂 Using: {files[0]}")
    with open(files[0]) as f:
        return json.load(f)

# ── Download via yt-dlp ───────────────────────────────────────

def download_reel(reel_id, likes, output_dir):
    url = f"https://www.instagram.com/reels/{reel_id}/"
    filename = f"{output_dir}/reel_{likes}likes_{reel_id}.mp4"

    if os.path.exists(filename):
        print(f"  ⏭️  Already exists: {filename}")
        return True

    cmd = [
        "yt-dlp",
        "--quiet",
        "--no-warnings",
        "-o", filename,
        url
    ]

    print(f"  ⬇️  Downloading {reel_id} ({likes:,} likes)...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"  ✅ Saved: {filename}")
        return True
    else:
        print(f"  ❌ Failed: {result.stderr.strip()[:120]}")
        return False

# ── Main ──────────────────────────────────────────────────────

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    reels = load_json(path)

    viral = [r for r in reels if r.get("likes", 0) > LIKES_THRESHOLD]
    viral.sort(key=lambda r: r["likes"], reverse=True)

    print(f"\n🔥 Found {len(viral)} viral reels (>{LIKES_THRESHOLD:,} likes) out of {len(reels)} total\n")

    if not viral:
        print("No viral reels to download.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    success = 0
    fail    = 0

    for i, reel in enumerate(viral, 1):
        reel_id = reel["reelId"]
        likes   = reel["likes"]
        print(f"[{i}/{len(viral)}]", end=" ")
        ok = download_reel(reel_id, likes, OUTPUT_DIR)
        if ok:
            success += 1
        else:
            fail += 1
        if i < len(viral):
            time.sleep(DELAY_BETWEEN)

    print(f"\n🏁 Done — {success} downloaded, {fail} failed")
    print(f"📁 Saved to: {os.path.abspath(OUTPUT_DIR)}/")

if __name__ == "__main__":
    main()