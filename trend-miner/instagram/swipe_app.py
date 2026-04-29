"""
Reels Swipe App
---------------
pip install flask requests

python swipe_app.py

Then open http://localhost:5000 in your browser.

Swipe RIGHT  → posts to platforms + deletes .mp4 (keeps JSON)
Swipe LEFT   → marks as "review" in JSON, keeps .mp4

── Platform Setup ──────────────────────────────────────────────
Fill in your API keys below. Leave empty to skip that platform.

INSTAGRAM: Requires Facebook Graph API + Instagram Business account
  https://developers.facebook.com/docs/instagram-api/content-publishing

TIKTOK: Requires TikTok Content Posting API
  https://developers.tiktok.com/doc/content-posting-api-get-started

YOUTUBE: Requires Google API + youtube.googleapis.com enabled
  https://developers.google.com/youtube/v3/guides/uploading_a_video
"""

import os, json, glob
from flask import Flask, jsonify, send_file, request, Response

# ── CONFIGURE THESE ───────────────────────────────────────────

DOWNLOADS_DIR   = "downloads"    # folder with your .mp4 files
JSON_GLOB       = "reels_*.json" # pattern to find your metrics JSON

# Platform credentials (leave as "" to skip)
INSTAGRAM_ACCESS_TOKEN  = "chaosinclips"     # Facebook Graph API long-lived token
INSTAGRAM_USER_ID       = "Sakal8983!"     # Instagram Business Account ID

TIKTOK_ACCESS_TOKEN     = ""     # TikTok content posting access token

YOUTUBE_CLIENT_SECRETS  = "D:\trend_miner\instagram\Tejas\client_secrets.json"     # path to client_secrets.json

# ── Flask app ─────────────────────────────────────────────────

app = Flask(__name__)

def load_metrics():
    files = sorted(glob.glob(JSON_GLOB), key=os.path.getmtime, reverse=True)
    if not files:
        return {}
    with open(files[0]) as f:
        data = json.load(f)
    return {r["reelId"]: r for r in data}

def save_metrics(metrics_dict):
    files = sorted(glob.glob(JSON_GLOB), key=os.path.getmtime, reverse=True)
    if not files:
        return
    data = list(metrics_dict.values())
    with open(files[0], "w") as f:
        json.dump(data, f, indent=2)

def get_video_list():
    mp4s = glob.glob(os.path.join(DOWNLOADS_DIR, "*.mp4"))
    metrics = load_metrics()
    videos = []
    for path in mp4s:
        fname = os.path.basename(path)
        # extract reelId from filename like reel_88700likes_DUz7qj6Df2Y.mp4
        parts = fname.replace(".mp4", "").split("_")
        reel_id = parts[-1] if len(parts) >= 3 else fname
        m = metrics.get(reel_id, {})
        # skip already processed
        if m.get("status") in ("posted", "review"):
            continue
        videos.append({
            "reelId":   reel_id,
            "filename": fname,
            "likes":    m.get("likes", 0),
            "comments": m.get("comments", 0),
            "time":     m.get("time", ""),
        })
    videos.sort(key=lambda v: v["likes"], reverse=True)
    return videos

# ── Platform posting ──────────────────────────────────────────

def post_to_instagram(video_path, reel_id):
    if not INSTAGRAM_ACCESS_TOKEN or not INSTAGRAM_USER_ID:
        return False, "Instagram credentials not set"
    try:
        import requests
        # Step 1: Create container
        r = requests.post(
            f"https://graph.facebook.com/v19.0/{INSTAGRAM_USER_ID}/media",
            data={
                "media_type":  "REELS",
                "video_url":   f"file://{os.path.abspath(video_path)}",  # must be public URL in prod
                "access_token": INSTAGRAM_ACCESS_TOKEN,
            }
        )
        container_id = r.json().get("id")
        if not container_id:
            return False, r.text
        # Step 2: Publish
        r2 = requests.post(
            f"https://graph.facebook.com/v19.0/{INSTAGRAM_USER_ID}/media_publish",
            data={"creation_id": container_id, "access_token": INSTAGRAM_ACCESS_TOKEN}
        )
        return r2.status_code == 200, r2.text
    except Exception as e:
        return False, str(e)

def post_to_tiktok(video_path, reel_id):
    if not TIKTOK_ACCESS_TOKEN:
        return False, "TikTok credentials not set"
    try:
        import requests
        with open(video_path, "rb") as f:
            r = requests.post(
                "https://open.tiktokapis.com/v2/post/publish/video/init/",
                headers={"Authorization": f"Bearer {TIKTOK_ACCESS_TOKEN}",
                         "Content-Type": "application/json"},
                json={
                    "post_info": {"title": "", "privacy_level": "PUBLIC_TO_EVERYONE",
                                  "disable_duet": False, "disable_comment": False,
                                  "disable_stitch": False},
                    "source_info": {"source": "FILE_UPLOAD",
                                    "video_size": os.path.getsize(video_path),
                                    "chunk_size": os.path.getsize(video_path),
                                    "total_chunk_count": 1}
                }
            )
        return r.status_code == 200, r.text
    except Exception as e:
        return False, str(e)

def post_to_youtube(video_path, reel_id):
    if not YOUTUBE_CLIENT_SECRETS:
        return False, "YouTube credentials not set"
    try:
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaFileUpload
        from google_auth_oauthlib.flow import InstalledAppFlow
        flow = InstalledAppFlow.from_client_secrets_file(
            YOUTUBE_CLIENT_SECRETS,
            ["https://www.googleapis.com/auth/youtube.upload"]
        )
        creds = flow.run_local_server(port=0)
        yt = build("youtube", "v3", credentials=creds)
        body = {"snippet": {"title": f"Reel {reel_id}", "categoryId": "22"},
                "status":  {"privacyStatus": "public"}}
        media = MediaFileUpload(video_path, mimetype="video/mp4", resumable=True)
        req = yt.videos().insert(part="snippet,status", body=body, media_body=media)
        req.execute()
        return True, "uploaded"
    except Exception as e:
        return False, str(e)

# ── Routes ────────────────────────────────────────────────────

@app.route("/")
def index():
    return Response(HTML, mimetype="text/html")

@app.route("/api/videos")
def api_videos():
    return jsonify(get_video_list())

@app.route("/api/video/<filename>")
def api_serve_video(filename):
    path = os.path.join(DOWNLOADS_DIR, filename)
    if not os.path.exists(path):
        return jsonify({"error": "not found"}), 404
    return send_file(path, mimetype="video/mp4")

@app.route("/api/swipe", methods=["POST"])
def api_swipe():
    data     = request.json
    reel_id  = data["reelId"]
    filename = data["filename"]
    direction = data["direction"]
    categories = data.get("categories", [])  # now an array
    
    metrics = load_metrics()
    
    if direction == "right":
        # Post to platforms (non-blocking)
        video_path = os.path.join(DOWNLOADS_DIR, filename)
        results = {}
        
        if INSTAGRAM_ACCESS_TOKEN:
            ok, msg = post_to_instagram(video_path, reel_id)
            results["instagram"] = "✅" if ok else f"❌ {msg[:60]}"
        
        if TIKTOK_ACCESS_TOKEN:
            ok, msg = post_to_tiktok(video_path, reel_id)
            results["tiktok"] = "✅" if ok else f"❌ {msg[:60]}"
        
        if YOUTUBE_CLIENT_SECRETS:
            ok, msg = post_to_youtube(video_path, reel_id)
            results["youtube"] = "✅" if ok else f"❌ {msg[:60]}"

        # Delete .mp4
        if os.path.exists(video_path):
            os.remove(video_path)
        
        if reel_id in metrics:
            metrics[reel_id]["status"] = "posted"
            metrics[reel_id]["categories"] = categories
            metrics[reel_id]["postResults"] = results
            save_metrics(metrics)
        
        return jsonify({"ok": True, "status": "posted", "results": results, "categories": categories})
    
    elif direction == "left":
        if reel_id in metrics:
            metrics[reel_id]["status"] = "review"
            metrics[reel_id]["categories"] = categories
            save_metrics(metrics)
        return jsonify({"ok": True, "status": "review", "categories": categories})
    
    elif direction == "skip":
        if reel_id in metrics:
            metrics[reel_id]["status"] = "skipped"
            metrics[reel_id]["categories"] = categories
            save_metrics(metrics)
        return jsonify({"ok": True, "status": "skipped", "categories": categories})
    
    elif direction == "redo":
        # Reset status to pending
        if reel_id in metrics and "status" in metrics[reel_id]:
            del metrics[reel_id]["status"]
            save_metrics(metrics)
        return jsonify({"ok": True, "status": "pending"})
    
    return jsonify({"error": "unknown direction"}), 400

@app.route("/api/stats")
def api_stats():
    metrics = load_metrics()
    return jsonify({
        "total":   len(metrics),
        "posted":  sum(1 for r in metrics.values() if r.get("status") == "posted"),
        "review":  sum(1 for r in metrics.values() if r.get("status") == "review"),
        "pending": len(get_video_list()),
    })

# ── Embedded HTML UI ──────────────────────────────────────────

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Reel Swiper</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #0a0a0f;
    --surface: #13131a;
    --border: #1e1e2e;
    --accent-green: #00ff88;
    --accent-red: #ff3366;
    --accent-yellow: #ffcc00;
    --accent-purple: #9d4edd;
    --accent-blue: #3b82f6;
    --text: #e8e8f0;
    --muted: #555570;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'DM Sans', sans-serif;
    height: 100vh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    user-select: none;
  }

  /* Header */
  header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 16px 24px;
    border-bottom: 1px solid var(--border);
    flex-shrink: 0;
  }
  .logo {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 22px;
    letter-spacing: 3px;
    color: var(--accent-yellow);
  }
  .stats {
    display: flex;
    gap: 20px;
    font-size: 12px;
    color: var(--muted);
  }
  .stats span strong {
    color: var(--text);
    font-size: 14px;
  }

  /* Main area - now with flex row */
  main {
    flex: 1;
    display: flex;
    flex-direction: row;
    align-items: center;
    justify-content: center;
    position: relative;
    overflow: hidden;
    gap: 40px;
    padding: 20px;
  }

  /* Card stack - centered */
  .card-stack {
    position: relative;
    width: 380px;
    height: 640px;
    flex-shrink: 0;
  }
  .card {
    position: absolute;
    width: 100%;
    height: 100%;
    background: var(--surface);
    border-radius: 20px;
    overflow: hidden;
    border: 1px solid var(--border);
    cursor: grab;
    transform-origin: bottom center;
    transition: box-shadow 0.2s;
    will-change: transform;
  }
  .card:active { cursor: grabbing; }
  .card.top {
    box-shadow: 0 30px 80px rgba(0,0,0,0.7);
    z-index: 10;
  }
  .card.behind-1 {
    transform: scale(0.95) translateY(16px);
    z-index: 9;
    pointer-events: none;
  }
  .card.behind-2 {
    transform: scale(0.90) translateY(32px);
    z-index: 8;
    pointer-events: none;
  }

  video.card-video {
    width: 100%;
    height: 100%;
    object-fit: cover;
    pointer-events: none;
  }

  .card-overlay {
    position: absolute;
    inset: 0;
    background: linear-gradient(to top, rgba(0,0,0,0.85) 0%, transparent 50%);
    pointer-events: none;
  }
  .card-info {
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    padding: 20px;
  }
  .card-id {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 13px;
    letter-spacing: 2px;
    color: var(--muted);
    margin-bottom: 6px;
  }
  .card-likes {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 36px;
    line-height: 1;
    color: white;
  }
  .card-likes span {
    font-family: 'DM Sans', sans-serif;
    font-size: 13px;
    color: var(--muted);
    font-weight: 300;
    margin-left: 4px;
  }
  .card-comments {
    font-size: 13px;
    color: var(--muted);
    margin-top: 4px;
  }

  /* Swipe indicators */
  .stamp {
    position: absolute;
    top: 30px;
    font-family: 'Bebas Neue', sans-serif;
    font-size: 48px;
    letter-spacing: 4px;
    padding: 6px 18px;
    border-radius: 8px;
    border-width: 4px;
    border-style: solid;
    opacity: 0;
    pointer-events: none;
    z-index: 20;
    transform: rotate(-15deg);
    transition: opacity 0.1s;
  }
  .stamp.post {
    left: 24px;
    color: var(--accent-green);
    border-color: var(--accent-green);
    transform: rotate(-15deg);
  }
  .stamp.review {
    right: 24px;
    color: var(--accent-red);
    border-color: var(--accent-red);
    transform: rotate(15deg);
  }

  /* Right sidebar */
  .sidebar-right {
    display: flex;
    flex-direction: column;
    gap: 24px;
    width: 300px;
    flex-shrink: 0;
  }

  /* Horizontal categories panel */
  .categories-horizontal {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 20px;
  }

  .panel-title {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: var(--muted);
    margin-bottom: 12px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  
  .clear-cats {
    font-size: 10px;
    color: var(--accent-yellow);
    cursor: pointer;
    text-transform: uppercase;
    letter-spacing: 1px;
    opacity: 0.7;
    transition: opacity 0.2s;
  }
  
  .clear-cats:hover {
    opacity: 1;
  }

  .category-row {
    display: flex;
    flex-wrap: wrap;
    gap: 10px 12px;
  }

  .category-btn.multi {
    padding: 10px 14px;
    font-size: 13px;
    min-width: auto;
    border-radius: 999px;
    border: 1px solid var(--border);
    background: transparent;
    color: var(--text);
    cursor: pointer;
    transition: all 0.18s;
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
  }

  .category-btn.multi.active {
    background: rgba(255, 204, 0, 0.14);
    border-color: var(--accent-yellow);
    color: var(--accent-yellow);
    box-shadow: 0 0 0 3px rgba(255, 204, 0, 0.08);
  }

  .category-btn.multi:hover:not(.active) {
    border-color: #444466;
    background: rgba(255,255,255,0.03);
  }

  .category-btn.multi .emoji {
    font-size: 16px;
    margin-right: 6px;
  }

  /* Actions panel (vertical) */
  .panel-section.vertical-actions {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 20px;
  }

  .action-btn {
    background: transparent;
    border: 1px solid var(--border);
    border-radius: 40px;
    padding: 14px 16px;
    color: var(--text);
    font-family: 'DM Sans', sans-serif;
    font-size: 14px;
    font-weight: 500;
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 12px;
    transition: all 0.2s;
    width: 100%;
    margin-bottom: 10px;
  }
  
  .action-btn:last-child {
    margin-bottom: 0;
  }
  
  .action-btn:hover {
    transform: translateX(-4px);
  }
  
  .action-btn .icon {
    font-size: 18px;
    width: 24px;
  }
  
  .action-btn.post {
    background: rgba(0,255,136,0.1);
    border-color: var(--accent-green);
    color: var(--accent-green);
  }
  
  .action-btn.post:hover {
    background: rgba(0,255,136,0.2);
  }
  
  .action-btn.review {
    border-color: var(--accent-red);
    color: var(--accent-red);
  }
  
  .action-btn.review:hover {
    background: rgba(255,51,102,0.1);
  }
  
  .action-btn.skip {
    border-color: var(--accent-yellow);
    color: var(--accent-yellow);
  }
  
  .action-btn.redo {
    border-color: var(--accent-blue);
    color: var(--accent-blue);
  }

  /* Empty state */
  .empty {
    text-align: center;
    color: var(--muted);
  }
  .empty h2 {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 42px;
    letter-spacing: 4px;
    color: var(--accent-yellow);
    margin-bottom: 8px;
  }

  /* Toast */
  .toast {
    position: fixed;
    top: 20px;
    left: 50%;
    transform: translateX(-50%) translateY(-80px);
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px 20px;
    font-size: 13px;
    color: var(--text);
    z-index: 100;
    transition: transform 0.3s cubic-bezier(0.34,1.56,0.64,1);
    white-space: nowrap;
    max-width: 90%;
    white-space: normal;
    word-break: break-word;
  }
  .toast.show { transform: translateX(-50%) translateY(0); }

  /* Play button overlay */
  .play-overlay {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    width: 60px;
    height: 60px;
    background: rgba(0,0,0,0.6);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    z-index: 30;
    opacity: 0;
    transition: opacity 0.3s;
    pointer-events: auto;
    border: 2px solid rgba(255,255,255,0.3);
  }
  
  .play-overlay.visible {
    opacity: 1;
  }
  
  .play-overlay::after {
    content: "▶";
    color: white;
    font-size: 24px;
    margin-left: 4px;
  }
  
  /* Selected categories indicator */
  .selected-count {
    background: var(--accent-yellow);
    color: var(--bg);
    border-radius: 12px;
    padding: 2px 8px;
    font-size: 10px;
    font-weight: bold;
  }

  /* Responsive */
  @media (max-width: 1100px) {
    main {
      flex-direction: column;
      gap: 24px;
      padding: 16px;
      overflow-y: auto;
    }
    .sidebar-right {
      width: 100%;
      max-width: 420px;
      margin: 0 auto;
    }
    .card-stack {
      width: 100%;
      max-width: 420px;
      height: 640px;
      margin: 0 auto;
    }
  }
</style>
</head>
<body>

<header>
  <div class="logo">REEL SWIPER</div>
  <div class="stats">
    <div><strong id="stat-pending">-</strong><br>pending</div>
    <div><strong id="stat-posted">-</strong><br>posted</div>
    <div><strong id="stat-review">-</strong><br>review</div>
  </div>
</header>

<main>
  <div class="card-stack" id="stack">
    <div class="empty" id="empty-state" style="display:none;">
      <h2>ALL DONE</h2>
      <p>Every reel has been sorted.</p>
    </div>
  </div>
  
  <div class="sidebar-right">
    <!-- Horizontal multi-select categories -->
    <div class="categories-horizontal">
      <div class="panel-title">
        📁 CATEGORIES 
        <span class="clear-cats" onclick="clearAllCategories()">Clear all</span>
      </div>
      <div class="category-row">
        <button class="category-btn multi" data-cat="funny" onclick="toggleCategory('funny')">
          <span class="emoji">😂</span> Funny
        </button>
        <button class="category-btn multi" data-cat="educational" onclick="toggleCategory('educational')">
          <span class="emoji">📚</span> Educational
        </button>
        <button class="category-btn multi" data-cat="inspirational" onclick="toggleCategory('inspirational')">
          <span class="emoji">✨</span> Inspirational
        </button>
        <button class="category-btn multi" data-cat="lifestyle" onclick="toggleCategory('lifestyle')">
          <span class="emoji">🏠</span> Lifestyle
        </button>
        <button class="category-btn multi" data-cat="other" onclick="toggleCategory('other')">
          <span class="emoji">📌</span> Other
        </button>
      </div>
      <div style="margin-top: 12px; font-size: 11px; color: var(--muted; display: flex; justify-content: flex-end;">
        <span id="selected-cats-display">Selected: <span id="selected-count">1</span></span>
      </div>
    </div>

    <!-- Actions (vertical) -->
    <div class="panel-section vertical-actions">
      <div class="panel-title">⚡ ACTIONS</div>
      <button class="action-btn post" onclick="performAction('right')">
        <span class="icon">🚀</span> Post Directly
      </button>
      <button class="action-btn review" onclick="performAction('left')">
        <span class="icon">↩</span> Review Later
      </button>
      <button class="action-btn skip" onclick="performAction('skip')">
        <span class="icon">⏭</span> Skip
      </button>
      <button class="action-btn redo" onclick="performAction('redo')">
        <span class="icon">↺</span> Redo
      </button>
    </div>
  </div>
</main>

<div class="toast" id="toast"></div>

<script>
let videos = [];
let current = 0;
let isDragging = false;
let startX = 0, startY = 0;
let currentX = 0;
let topCard = null;
let selectedCategories = ['funny'];  // default selection
let currentVideoElement = null;

// Initialize category buttons
function updateCategoryButtons() {
  document.querySelectorAll('.category-btn.multi').forEach(btn => {
    const cat = btn.dataset.cat;
    if (selectedCategories.includes(cat)) {
      btn.classList.add('active');
    } else {
      btn.classList.remove('active');
    }
  });
  document.getElementById('selected-count').textContent = selectedCategories.length;
}

function toggleCategory(category) {
  if (selectedCategories.includes(category)) {
    // Don't allow empty selection - keep at least one category
    if (selectedCategories.length > 1) {
      selectedCategories = selectedCategories.filter(c => c !== category);
    } else {
      showToast("Keep at least one category selected");
      return;
    }
  } else {
    selectedCategories.push(category);
  }
  
  updateCategoryButtons();
  
  const cats = selectedCategories.map(c => c.charAt(0).toUpperCase() + c.slice(1)).join(", ");
  showToast(`Categories: ${cats}`);
}

function clearAllCategories() {
  // Reset to just 'funny' as default
  selectedCategories = ['funny'];
  updateCategoryButtons();
  showToast("Categories reset to: Funny");
}

async function loadVideos() {
  try {
    const res = await fetch("/api/videos");
    videos = await res.json();
    current = 0;
    renderStack();
    loadStats();
  } catch (error) {
    console.error("Error loading videos:", error);
    showToast("Error loading videos");
  }
}

async function loadStats() {
  try {
    const res = await fetch("/api/stats");
    const s = await res.json();
    document.getElementById("stat-pending").textContent = s.pending;
    document.getElementById("stat-posted").textContent  = s.posted;
    document.getElementById("stat-review").textContent  = s.review;
  } catch (error) {
    console.error("Error loading stats:", error);
  }
}

function renderStack() {
  const stack = document.getElementById("stack");
  // Remove old cards
  stack.querySelectorAll(".card").forEach(c => c.remove());

  if (current >= videos.length) {
    document.getElementById("empty-state").style.display = "flex";
    document.getElementById("empty-state").style.flexDirection = "column";
    document.getElementById("empty-state").style.alignItems = "center";
    document.getElementById("empty-state").style.justifyContent = "center";
    document.getElementById("empty-state").style.height = "100%";
    return;
  }

  document.getElementById("empty-state").style.display = "none";

  // Render up to 3 cards
  for (let i = Math.min(current + 2, videos.length - 1); i >= current; i--) {
    const v = videos[i];
    const card = buildCard(v, i - current);
    stack.appendChild(card);
  }

  topCard = stack.querySelector(".card.top");
  if (topCard) {
    setupVideo(topCard, videos[current]);
    attachDrag(topCard, videos[current]);
  }
}

function buildCard(v, stackPos) {
  const card = document.createElement("div");
  card.className = "card " + (stackPos === 0 ? "top" : stackPos === 1 ? "behind-1" : "behind-2");
  card.dataset.reelId = v.reelId;
  card.dataset.filename = v.filename;

  const video = document.createElement("video");
  video.className = "card-video";
  video.loop = true;
  video.muted = true;
  video.playsInline = true;
  video.preload = stackPos === 0 ? "auto" : "none";
  
  // Add source
  const source = document.createElement("source");
  source.src = `/api/video/${v.filename}`;
  source.type = "video/mp4";
  video.appendChild(source);

  const overlay = document.createElement("div");
  overlay.className = "card-overlay";

  const stampPost = document.createElement("div");
  stampPost.className = "stamp post";
  stampPost.textContent = "POST";

  const stampReview = document.createElement("div");
  stampReview.className = "stamp review";
  stampReview.textContent = "REVIEW";

  // Play button overlay
  const playOverlay = document.createElement("div");
  playOverlay.className = "play-overlay";
  if (stackPos === 0) {
    playOverlay.classList.add("visible");
  }

  const info = document.createElement("div");
  info.className = "card-info";
  info.innerHTML = `
    <div class="card-id">${v.reelId}</div>
    <div class="card-likes">${formatNum(v.likes)}<span>likes</span></div>
    <div class="card-comments">${formatNum(v.comments)} comments</div>
  `;

  card.appendChild(video);
  card.appendChild(overlay);
  card.appendChild(stampPost);
  card.appendChild(stampReview);
  card.appendChild(playOverlay);
  card.appendChild(info);
  
  return card;
}

function setupVideo(card, videoData) {
  const video = card.querySelector('video');
  const playOverlay = card.querySelector('.play-overlay');
  
  if (!video) return;
  
  currentVideoElement = video;
  
  // Remove old event listeners
  video.oncanplay = null;
  video.onplay = null;
  video.onpause = null;
  
  // Set up video
  video.muted = true; // Start muted for autoplay
  video.volume = 0;
  
  // Try to play
  const playPromise = video.play();
  
  if (playPromise !== undefined) {
    playPromise.then(() => {
      // Autoplay started successfully
      console.log("Video playing");
      if (playOverlay) playOverlay.classList.remove('visible');
      
      // Allow unmuting after user interaction
      video.addEventListener('click', () => {
        video.muted = !video.muted;
        video.volume = video.muted ? 0 : 1;
      });
    }).catch(error => {
      // Autoplay was prevented
      console.log("Autoplay prevented:", error);
      if (playOverlay) playOverlay.classList.add('visible');
      
      // Play on overlay click
      playOverlay.addEventListener('click', (e) => {
        e.stopPropagation();
        video.play();
        video.muted = false;
        video.volume = 1;
        playOverlay.classList.remove('visible');
      });
    });
  }
  
  // Handle video errors
  video.onerror = (e) => {
    console.error("Video error:", video.error);
    showToast("Error loading video");
  };
  
  // Handle canplay event
  video.oncanplay = () => {
    console.log("Video can play");
  };
}

function formatNum(n) {
  if (n >= 1_000_000) return (n/1_000_000).toFixed(1) + "M";
  if (n >= 1_000) return (n/1_000).toFixed(1) + "K";
  return n;
}

// ── Drag logic ────────────────────────────────────────────────

function attachDrag(card, videoData) {
  if (!card) return;

  card.addEventListener("mousedown", onStart);
  card.addEventListener("touchstart", onStart, { passive: true });

  function onStart(e) {
    isDragging = true;
    const pt = e.touches ? e.touches[0] : e;
    startX = pt.clientX;
    startY = pt.clientY;
    card.style.transition = "none";

    window.addEventListener("mousemove", onMove);
    window.addEventListener("touchmove", onMove, { passive: true });
    window.addEventListener("mouseup", onEnd);
    window.addEventListener("touchend", onEnd);
  }

  function onMove(e) {
    if (!isDragging) return;
    const pt = e.touches ? e.touches[0] : e;
    currentX = pt.clientX - startX;
    const currentY = pt.clientY - startY;
    const rotate = currentX * 0.08;

    card.style.transform = `translate(${currentX}px, ${currentY * 0.3}px) rotate(${rotate}deg)`;

    const ratio = Math.abs(currentX) / 120;
    const stampPost   = card.querySelector(".stamp.post");
    const stampReview = card.querySelector(".stamp.review");

    if (currentX > 20) {
      stampPost.style.opacity   = Math.min(ratio, 1);
      stampReview.style.opacity = 0;
    } else if (currentX < -20) {
      stampReview.style.opacity = Math.min(ratio, 1);
      stampPost.style.opacity   = 0;
    } else {
      stampPost.style.opacity   = 0;
      stampReview.style.opacity = 0;
    }
  }

  function onEnd() {
    if (!isDragging) return;
    isDragging = false;
    window.removeEventListener("mousemove", onMove);
    window.removeEventListener("touchmove", onMove);
    window.removeEventListener("mouseup", onEnd);
    window.removeEventListener("touchend", onEnd);

    const threshold = 100;
    if (currentX > threshold) {
      flyOut(card, "right", videoData);
    } else if (currentX < -threshold) {
      flyOut(card, "left", videoData);
    } else {
      // Snap back
      card.style.transition = "transform 0.4s cubic-bezier(0.34,1.56,0.64,1)";
      card.style.transform  = "";
      card.querySelector(".stamp.post").style.opacity   = 0;
      card.querySelector(".stamp.review").style.opacity = 0;
    }
    currentX = 0;
  }
}

function flyOut(card, direction, videoData) {
  const x = direction === "right" ? 1000 : -1000;
  card.style.transition = "transform 0.4s ease-in, opacity 0.4s";
  card.style.transform  = `translate(${x}px, -80px) rotate(${direction === "right" ? 30 : -30}deg)`;
  card.style.opacity    = "0";
  setTimeout(() => {
    doSwipe(direction, videoData, selectedCategories);
  }, 350);
}

async function performAction(action) {
  if (current >= videos.length) {
    showToast("No more videos");
    return;
  }
  
  const card = document.querySelector(".card.top");
  if (!card) return;
  
  const videoData = videos[current];
  
  if (action === 'redo') {
    await doSwipe('redo', videoData, selectedCategories);
  } else if (action === 'skip') {
    await doSwipe('skip', videoData, selectedCategories);
  } else {
    // For post and review, trigger flyout animation
    flyOut(card, action === 'right' ? 'right' : 'left', videoData);
  }
}

async function doSwipe(direction, video, categories) {
  try {
    const res = await fetch("/api/swipe", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        reelId: video.reelId,
        filename: video.filename,
        direction: direction,
        categories: categories
      })
    });
    const data = await res.json();

    if (direction === "right") {
      const platforms = Object.entries(data.results || {});
      const catStr = categories.join(', ');
      if (platforms.length === 0) {
        showToast(`🚀 Posted (${catStr}) · .mp4 deleted`);
      } else {
        const summary = platforms.map(([p,r]) => `${p}: ${r}`).join(" · ");
        showToast(`Posted (${catStr}) · ${summary}`);
      }
    } else if (direction === "left") {
      showToast(`↩ Saved for review (${categories.join(', ')})`);
    } else if (direction === "skip") {
      showToast(`⏭ Skipped (${categories.join(', ')})`);
    } else if (direction === "redo") {
      showToast(`↺ Ready to redo`);
    }
  } catch(e) {
    showToast("⚠️ Error: " + e.message);
  }

  if (direction !== "redo") {
    current++;
  }
  renderStack();
  loadStats();
}

function showToast(msg) {
  const t = document.getElementById("toast");
  t.textContent = msg;
  t.classList.add("show");
  setTimeout(() => t.classList.remove("show"), 2800);
}

// Keyboard shortcuts
document.addEventListener("keydown", e => {
  if (e.key === "ArrowRight") performAction("right");
  if (e.key === "ArrowLeft")  performAction("left");
  if (e.key === "ArrowDown")  performAction("skip");
  if (e.key === "r")          performAction("redo");
  if (e.key === "c")          clearAllCategories();
  // Number keys for categories (toggle)
  if (e.key === "1") toggleCategory('funny');
  if (e.key === "2") toggleCategory('educational');
  if (e.key === "3") toggleCategory('inspirational');
  if (e.key === "4") toggleCategory('lifestyle');
  if (e.key === "5") toggleCategory('other');
});

// Initialize
document.addEventListener("DOMContentLoaded", () => {
  updateCategoryButtons();
  loadVideos();
});
</script>
</body>
</html>
"""

if __name__ == "__main__":
    if not os.path.exists(DOWNLOADS_DIR):
        print(f"⚠️  '{DOWNLOADS_DIR}' folder not found. Make sure it's in the same directory.")
    print("🚀  Open http://localhost:5000 in your browser")
    print("    Categories (multi-select):")
    print("    1=Funny, 2=Educational, 3=Inspirational, 4=Lifestyle, 5=Other, C=Clear all")
    print("    Actions:")
    print("    → Right arrow / Click Post = Post & delete .mp4")
    print("    ← Left arrow / Click Review = Save for review")
    print("    ↓ Down arrow / Click Skip = Skip")
    print("    R key / Click Redo = Redo current video")
    print("\n    📹 Video playback:")
    print("    - Videos start muted for autoplay")
    print("    - Click video to toggle mute/unmute")
    print("    - Click play button if autoplay is blocked")
    app.run(host='0.0.0.0', port=5000, debug=False)