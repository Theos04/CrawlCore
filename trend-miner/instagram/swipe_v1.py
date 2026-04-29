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

import os, json, glob, threading
from flask import Flask, jsonify, send_file, request, Response
from poster import post_to_all

# ── CONFIGURE THESE ───────────────────────────────────────────

DOWNLOADS_DIR   = "downloads"    # folder with your .mp4 files
JSON_GLOB       = "reels_*.json" # pattern to find your metrics JSON

# Platform credentials (leave as "" to skip)
INSTAGRAM_ACCESS_TOKEN  = ""     # Facebook Graph API long-lived token
INSTAGRAM_USER_ID       = ""     # Instagram Business Account ID

TIKTOK_ACCESS_TOKEN     = ""     # TikTok content posting access token

YOUTUBE_CLIENT_SECRETS  = ""     # path to client_secrets.json

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
    direction = data["direction"]  # "right" or "left"
    
    metrics = load_metrics()
    
    if direction == "right":
        # Post to platforms (non-blocking)
        video_path = os.path.join(DOWNLOADS_DIR, filename)
        raw     = post_to_all(video_path)
        results = {p: ("✅ " + msg) if ok else ("❌ " + msg[:80]) for p, (ok, msg) in raw.items()}

        # Delete .mp4
        if os.path.exists(video_path):
            os.remove(video_path)
        
        if reel_id in metrics:
            metrics[reel_id]["status"] = "posted"
            metrics[reel_id]["postResults"] = results
            save_metrics(metrics)
        
        return jsonify({"ok": True, "status": "posted", "results": results})
    
    elif direction == "left":
        if reel_id in metrics:
            metrics[reel_id]["status"] = "review"
            save_metrics(metrics)
        return jsonify({"ok": True, "status": "review"})
    
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
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="mobile-web-app-capable" content="yes">
<title>Reel Swiper</title>
<link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #0a0a0f;
    --surface: #13131a;
    --border: #1e1e2e;
    --green: #00ff88;
    --red: #ff3366;
    --yellow: #ffcc00;
    --text: #e8e8f0;
    --muted: #555570;
    --card-w: min(88vw, 340px);
    --card-h: min(76vh, 580px);
  }
  * { margin:0; padding:0; box-sizing:border-box; -webkit-tap-highlight-color:transparent; }
  html, body {
    height: 100%; height: 100dvh;
    background: var(--bg);
    color: var(--text);
    font-family: 'DM Sans', sans-serif;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    touch-action: none;
  }

  header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: clamp(10px,2vh,16px) clamp(14px,4vw,24px);
    border-bottom: 1px solid var(--border);
    flex-shrink: 0;
  }
  .logo {
    font-family: 'Bebas Neue', sans-serif;
    font-size: clamp(16px,4vw,22px);
    letter-spacing: 3px;
    color: var(--yellow);
  }
  .stats {
    display: flex;
    gap: clamp(12px,3vw,20px);
    font-size: clamp(10px,2.5vw,12px);
    color: var(--muted);
  }
  .stats span strong {
    display: block;
    color: var(--text);
    font-size: clamp(12px,3vw,14px);
    text-align: center;
  }

  main {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    position: relative;
    overflow: hidden;
    min-height: 0;
  }

  .card-stack {
    position: relative;
    width: var(--card-w);
    height: var(--card-h);
  }
  .card {
    position: absolute;
    width: 100%; height: 100%;
    background: var(--surface);
    border-radius: 18px;
    overflow: hidden;
    border: 1px solid var(--border);
    transform-origin: bottom center;
    will-change: transform;
    cursor: grab;
  }
  .card:active { cursor: grabbing; }
  .card.top    { z-index:10; box-shadow: 0 20px 60px rgba(0,0,0,0.8); }
  .card.behind-1 { transform: scale(0.95) translateY(14px); z-index:9; pointer-events:none; }
  .card.behind-2 { transform: scale(0.90) translateY(28px); z-index:8; pointer-events:none; }

  video.card-video {
    width:100%; height:100%;
    object-fit: cover;
    pointer-events: none;
  }
  .card-overlay {
    position: absolute; inset:0;
    background: linear-gradient(to top, rgba(0,0,0,0.88) 0%, transparent 55%);
    pointer-events: none;
  }
  .card-info {
    position: absolute;
    bottom:0; left:0; right:0;
    padding: clamp(14px,3vw,20px);
  }
  .card-id {
    font-family: 'Bebas Neue', sans-serif;
    font-size: clamp(10px,2.5vw,13px);
    letter-spacing: 2px;
    color: var(--muted);
    margin-bottom: 4px;
  }
  .card-likes {
    font-family: 'Bebas Neue', sans-serif;
    font-size: clamp(28px,7vw,38px);
    line-height: 1;
    color: white;
  }
  .card-likes span { font-family:'DM Sans',sans-serif; font-size:12px; color:var(--muted); margin-left:3px; }
  .card-comments { font-size: clamp(11px,2.5vw,13px); color: var(--muted); margin-top:3px; }

  .stamp {
    position: absolute;
    top: clamp(18px,4vw,30px);
    font-family: 'Bebas Neue', sans-serif;
    font-size: clamp(32px,8vw,48px);
    letter-spacing: 3px;
    padding: 4px 14px;
    border-radius: 8px;
    border-width: 3px;
    border-style: solid;
    opacity: 0;
    pointer-events: none;
    z-index: 20;
    transition: opacity 0.08s;
  }
  .stamp.post   { left:16px; color:var(--green);  border-color:var(--green);  transform:rotate(-15deg); }
  .stamp.review { right:16px; color:var(--red);   border-color:var(--red);    transform:rotate(15deg); }

  .actions {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: clamp(16px,5vw,28px);
    padding: clamp(10px,2vh,16px) 24px clamp(12px,2vh,20px);
    flex-shrink: 0;
  }
  .btn {
    border-radius: 50%;
    border: none;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: transform .15s, box-shadow .15s;
    flex-shrink: 0;
  }
  .btn:active { transform: scale(0.9) !important; }
  .btn-review {
    width: clamp(50px,13vw,62px); height: clamp(50px,13vw,62px);
    font-size: clamp(18px,5vw,24px);
    background: rgba(255,51,102,0.12);
    border: 2px solid var(--red);
    color: var(--red);
  }
  .btn-post {
    width: clamp(58px,15vw,72px); height: clamp(58px,15vw,72px);
    font-size: clamp(22px,6vw,28px);
    background: rgba(0,255,136,0.12);
    border: 2px solid var(--green);
    color: var(--green);
  }
  .btn-skip {
    width: clamp(50px,13vw,62px); height: clamp(50px,13vw,62px);
    font-size: clamp(14px,4vw,16px);
    background: rgba(255,204,0,0.08);
    border: 2px solid var(--yellow);
    color: var(--yellow);
  }

  .labels {
    display: flex;
    justify-content: center;
    gap: clamp(16px,5vw,28px);
    padding-bottom: clamp(8px,2vh,14px);
    flex-shrink: 0;
  }
  .lbl {
    font-size: 9px;
    letter-spacing: 1.5px;
    color: var(--muted);
    text-align: center;
    width: clamp(50px,13vw,62px);
  }
  .lbl.mid { width: clamp(58px,15vw,72px); }

  .empty {
    display: none;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100%;
    text-align: center;
    color: var(--muted);
  }
  .empty h2 {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 40px; letter-spacing: 4px;
    color: var(--yellow); margin-bottom: 8px;
  }

  .toast {
    position: fixed;
    top: 16px; left: 50%;
    transform: translateX(-50%) translateY(-100px);
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 10px 18px;
    font-size: 13px;
    z-index: 200;
    transition: transform .3s cubic-bezier(.34,1.56,.64,1);
    white-space: nowrap;
    max-width: 90vw;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .toast.show { transform: translateX(-50%) translateY(0); }
</style>
</head>
<body>

<header>
  <div class="logo">REEL SWIPER</div>
  <div class="stats">
    <div><strong id="sp">-</strong>queue</div>
    <div><strong id="ss">-</strong>posted</div>
    <div><strong id="sr">-</strong>review</div>
  </div>
</header>

<main>
  <div class="card-stack" id="stack">
    <div class="empty" id="empty">
      <h2>ALL DONE</h2>
      <p>Every reel has been sorted.</p>
    </div>
  </div>
</main>

<div class="actions">
  <button class="btn btn-review" onclick="swipe('left')">↩</button>
  <button class="btn btn-post"   onclick="swipe('right')">🚀</button>
  <button class="btn btn-skip"   onclick="skipCard()">⏭</button>
</div>
<div class="labels">
  <div class="lbl">REVIEW</div>
  <div class="lbl mid">POST</div>
  <div class="lbl">SKIP</div>
</div>

<div class="toast" id="toast"></div>

<script>
let videos = [], current = 0;
let dragging = false, startX = 0, startY = 0, dx = 0;

async function loadVideos() {
  const r = await fetch("/api/videos");
  videos = await r.json();
  current = 0;
  render();
  loadStats();
}

async function loadStats() {
  const r = await fetch("/api/stats");
  const s = await r.json();
  document.getElementById("sp").textContent = s.pending;
  document.getElementById("ss").textContent = s.posted;
  document.getElementById("sr").textContent = s.review;
}

function render() {
  const stack = document.getElementById("stack");
  stack.querySelectorAll(".card").forEach(c => c.remove());

  if (current >= videos.length) {
    const e = document.getElementById("empty");
    e.style.display = "flex";
    return;
  }
  document.getElementById("empty").style.display = "none";

  for (let i = Math.min(current + 2, videos.length - 1); i >= current; i--) {
    stack.appendChild(buildCard(videos[i], i - current));
  }
  const top = stack.querySelector(".card.top");
  hydrateVideo(top);
  attachDrag(top, videos[current]);
}

function buildCard(v, pos) {
  const card = document.createElement("div");
  card.className = "card " + ["top","behind-1","behind-2"][pos];

  const vid = document.createElement("video");
  vid.dataset.src = `/api/video/${v.filename}`;
  vid.loop = true; vid.muted = true; vid.playsInline = true;
  vid.setAttribute("playsinline", ""); vid.setAttribute("webkit-playsinline", "");
  vid.preload = pos === 0 ? "auto" : "none";
  vid.className = "card-video";

  const ov = document.createElement("div"); ov.className = "card-overlay";

  const sp = document.createElement("div"); sp.className = "stamp post"; sp.textContent = "POST";
  const sr = document.createElement("div"); sr.className = "stamp review"; sr.textContent = "SKIP";

  const info = document.createElement("div"); info.className = "card-info";
  info.innerHTML = `<div class="card-id">${v.reelId}</div>
    <div class="card-likes">${fmt(v.likes)}<span>likes</span></div>
    <div class="card-comments">${fmt(v.comments)} comments</div>`;

  card.append(vid, ov, sp, sr, info);
  return card;
}

function hydrateVideo(card) {
  if (!card) return;
  const vid = card.querySelector("video");
  if (vid && vid.dataset.src) {
    vid.src = vid.dataset.src;
    delete vid.dataset.src;
    vid.play().catch(() => {});
  }
}

function fmt(n) {
  if (n >= 1e6) return (n/1e6).toFixed(1) + "M";
  if (n >= 1e3) return (n/1e3).toFixed(1) + "K";
  return String(n);
}

function attachDrag(card, video) {
  if (!card) return;

  const onStart = e => {
    dragging = true; dx = 0;
    const p = e.touches ? e.touches[0] : e;
    startX = p.clientX; startY = p.clientY;
    card.style.transition = "none";
    window.addEventListener("mousemove", onMove);
    window.addEventListener("touchmove", onMove, {passive:true});
    window.addEventListener("mouseup",   onEnd);
    window.addEventListener("touchend",  onEnd);
  };

  const onMove = e => {
    if (!dragging) return;
    const p  = e.touches ? e.touches[0] : e;
    dx       = p.clientX - startX;
    const dy = (p.clientY - startY) * 0.25;
    card.style.transform = `translate(${dx}px,${dy}px) rotate(${dx*0.07}deg)`;
    const r = Math.min(Math.abs(dx) / 100, 1);
    card.querySelector(".stamp.post").style.opacity   = dx >  20 ? r : 0;
    card.querySelector(".stamp.review").style.opacity = dx < -20 ? r : 0;
  };

  const onEnd = () => {
    if (!dragging) return;
    dragging = false;
    window.removeEventListener("mousemove", onMove);
    window.removeEventListener("touchmove", onMove);
    window.removeEventListener("mouseup",   onEnd);
    window.removeEventListener("touchend",  onEnd);

    if      (dx >  90) flyOut(card, "right", video);
    else if (dx < -90) flyOut(card, "left",  video);
    else {
      card.style.transition = "transform .4s cubic-bezier(.34,1.56,.64,1)";
      card.style.transform  = "";
      card.querySelectorAll(".stamp").forEach(s => s.style.opacity = 0);
    }
    dx = 0;
  };

  card.addEventListener("mousedown",  onStart);
  card.addEventListener("touchstart", onStart, {passive:true});
}

function flyOut(card, dir, video) {
  card.style.transition = "transform .35s ease-in, opacity .35s";
  card.style.transform  = `translate(${dir==="right"?900:-900}px,-60px) rotate(${dir==="right"?25:-25}deg)`;
  card.style.opacity    = "0";
  setTimeout(() => doSwipe(dir, video), 320);
}

async function swipe(dir) {
  if (current >= videos.length) return;
  flyOut(document.querySelector(".card.top"), dir, videos[current]);
}

async function doSwipe(dir, video) {
  const res  = await fetch("/api/swipe", {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify({reelId:video.reelId, filename:video.filename, direction:dir})
  });
  const data = await res.json();
  if (dir === "right") {
    const parts = Object.entries(data.results||{});
    showToast(parts.length ? "🚀 " + parts.map(([p,r])=>`${p}:${r}`).join(" · ") : "🚀 Posted · .mp4 deleted");
  } else {
    showToast("↩ Saved for review");
  }
  current++; render(); loadStats();
}

function skipCard() { if (current < videos.length) { current++; render(); } }

function showToast(msg) {
  const t = document.getElementById("toast");
  t.textContent = msg; t.classList.add("show");
  setTimeout(() => t.classList.remove("show"), 2800);
}

document.addEventListener("keydown", e => {
  if (e.key==="ArrowRight") swipe("right");
  if (e.key==="ArrowLeft")  swipe("left");
  if (e.key==="ArrowDown")  skipCard();
});

loadVideos();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    if not os.path.exists(DOWNLOADS_DIR):
        print(f"⚠️  '{DOWNLOADS_DIR}' folder not found. Make sure it's in the same directory.")
    print("🚀  Open http://localhost:5000 in your browser")
    print("    ← Left arrow / click ↩ = Review")
    print("    → Right arrow / click 🚀 = Post & delete .mp4")
    print("    ↓ Down arrow / click ⏭  = Skip for now")
    app.run(host='0.0.0.0', port=5000, debug=False)  # accessible on Tailscale