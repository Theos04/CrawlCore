"""
Reel Swiper - Complete Working Script
--------------------------------------
A swipe-based interface for reviewing and posting reels/videos to multiple platforms.

Features:
- Swipe right to post to Instagram & YouTube (parallel posting)
- Swipe left to mark as reviewed without posting
- Auto-deletes videos after successful posting
- Tracks metrics and posting status in JSON

Setup:
pip install flask instagrapi google-auth-oauthlib googleapiclient

Platform Setup:
- Instagram: Fill in credentials below
- YouTube: Place client_secrets.json in the path below
"""

import os
import json
import glob
import threading
from flask import Flask, jsonify, request, send_from_directory, Response

# ================= CONFIGURATION =================

# Video and metrics paths
VIDEO_FOLDER = r"D:\trend_miner\instagram\downloads"  # Folder containing your .mp4 files
METRICS_PATTERN = r"E:\reels_1527_1771839673506.json" # Pattern for metrics JSON
YOUTUBE_SECRETS = r"D:\trend_miner\instagram\Tejas\client_secrets.json"  # YouTube OAuth file

# Instagram credentials (replace with your actual credentials)
INSTAGRAM_USERNAME = "chaosinclips"
INSTAGRAM_PASSWORD = "Sakal8983!"

# ================= FLASK APP =================

app = Flask(__name__)

print("🚀 Reel Swiper STARTED")
print(f"Video folder: {VIDEO_FOLDER}")
print(f"Metrics JSON pattern: {METRICS_PATTERN}")
print(f"YouTube secrets: {YOUTUBE_SECRETS}")

# ================= UTILITIES =================

def load_latest_metrics():
    """Load the most recent metrics JSON file."""
    files = glob.glob(METRICS_PATTERN)
    if not files:
        print("⚠️  No metrics files found")
        return []
    latest = max(files, key=os.path.getctime)
    print(f"📊 Loaded metrics from: {os.path.basename(latest)}")
    with open(latest, "r", encoding="utf-8") as f:
        return json.load(f)

def save_metrics(data):
    """Save metrics back to the latest file."""
    files = glob.glob(METRICS_PATTERN)
    if not files:
        print("⚠️  No metrics file to save to")
        return
    latest = max(files, key=os.path.getctime)
    with open(latest, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"💾 Metrics saved to: {os.path.basename(latest)}")

def get_sorted_videos():
    """Get unreviewed videos sorted by likes (highest first)."""
    data = load_latest_metrics()
    unreviewed = [v for v in data if not v.get("reviewed")]
    sorted_videos = sorted(unreviewed, key=lambda x: x.get("likes", 0), reverse=True)
    
    # Also check which videos actually exist on disk
    available = []
    for video in sorted_videos:
        filename = video.get("filename")
        if filename and os.path.exists(os.path.join(VIDEO_FOLDER, filename)):
            available.append(video)
        else:
            print(f"⚠️  Video file missing: {filename}")
    
    return available

# ================= PLATFORM POSTING =================

def post_to_instagram(video_path, filename):
    """Post video to Instagram Reels."""
    try:
        from instagrapi import Client
        
        print(f"📱 Posting to Instagram: {filename}")
        cl = Client()
        cl.login(INSTAGRAM_USERNAME, INSTAGRAM_PASSWORD)
        
        # Upload as reel/clip
        cl.clip_upload(
            video_path,
            caption=f"🔥 Trending Reel\n\n#viral #trending #reels"
        )
        
        print(f"✅ Instagram upload successful: {filename}")
        return True, "Instagram uploaded successfully"
    except Exception as e:
        print(f"❌ Instagram upload failed: {str(e)}")
        return False, f"Instagram failed: {str(e)[:100]}"

def post_to_youtube(video_path, filename):
    """Post video to YouTube Shorts."""
    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaFileUpload
        
        print(f"📱 Posting to YouTube: {filename}")
        
        scopes = ["https://www.googleapis.com/auth/youtube.upload"]
        
        if not os.path.exists(YOUTUBE_SECRETS):
            return False, f"YouTube secrets file not found: {YOUTUBE_SECRETS}"
        
        flow = InstalledAppFlow.from_client_secrets_file(YOUTUBE_SECRETS, scopes)
        creds = flow.run_local_server(port=0)
        
        youtube = build("youtube", "v3", credentials=creds)
        
        # Prepare video metadata
        title = os.path.splitext(filename)[0].replace("_", " ").title()
        request_body = {
            "snippet": {
                "title": title[:100],  # YouTube title limit
                "description": "🔥 Trending Short\n\n#shorts #viral #trending",
                "tags": ["shorts", "viral", "trending"],
                "categoryId": "22"  # "People & Blogs" category
            },
            "status": {
                "privacyStatus": "public"
            }
        }
        
        media = MediaFileUpload(video_path, chunksize=-1, resumable=True)
        
        # Upload video
        insert_request = youtube.videos().insert(
            part="snippet,status",
            body=request_body,
            media_body=media
        )
        
        insert_request.execute()
        print(f"✅ YouTube upload successful: {filename}")
        return True, "YouTube uploaded successfully"
        
    except Exception as e:
        print(f"❌ YouTube upload failed: {str(e)}")
        return False, f"YouTube failed: {str(e)[:100]}"

def post_to_all(video_path):
    """Post to all platforms in parallel."""
    results = {}
    
    def run_platform(name, func):
        ok, msg = func(video_path, os.path.basename(video_path))
        results[name] = (ok, msg)
    
    # Define active platforms
    platforms = {}
    
    if INSTAGRAM_USERNAME != "YOUR_USERNAME" and INSTAGRAM_PASSWORD != "YOUR_PASSWORD":
        platforms["instagram"] = post_to_instagram
    else:
        print("⚠️  Instagram credentials not set - skipping")
        results["instagram"] = (False, "Instagram credentials not configured")
    
    if os.path.exists(YOUTUBE_SECRETS):
        platforms["youtube"] = post_to_youtube
    else:
        print(f"⚠️  YouTube secrets not found - skipping")
        results["youtube"] = (False, "YouTube secrets file missing")
    
    # Run platforms in parallel
    threads = []
    for name, func in platforms.items():
        t = threading.Thread(target=run_platform, args=(name, func))
        t.start()
        threads.append(t)
    
    for t in threads:
        t.join()
    
    return results

# ================= ROUTES =================

@app.route("/")
def home():
    """Serve the main HTML interface."""
    return Response(HTML_TEMPLATE, mimetype="text/html")

@app.route("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok", "video_folder": VIDEO_FOLDER}

@app.route("/api/videos")
def api_videos():
    """Get list of videos to display."""
    return jsonify(get_sorted_videos())

@app.route("/api/video/<path:filename>")
def serve_video(filename):
    """Serve video files."""
    return send_from_directory(VIDEO_FOLDER, filename)

@app.route("/api/swipe", methods=["POST"])
def swipe():
    """Handle swipe actions (left=review, right=post)."""
    data = request.json
    filename = data.get("filename")
    direction = data.get("direction")
    
    if not filename or not direction:
        return jsonify({"error": "Missing filename or direction"}), 400
    
    metrics = load_latest_metrics()
    
    for item in metrics:
        if item.get("filename") == filename:
            item["reviewed"] = True
            
            if direction == "right":  # POST
                video_path = os.path.join(VIDEO_FOLDER, filename)
                
                if not os.path.exists(video_path):
                    return jsonify({"error": "Video file not found"}), 404
                
                # Post to platforms
                raw_results = post_to_all(video_path)
                
                # Format results for display
                formatted_results = {}
                for platform, (ok, msg) in raw_results.items():
                    formatted_results[platform] = "✅ " + msg if ok else "❌ " + msg
                
                # If any platform succeeded, mark as posted and delete video
                any_success = any(ok for ok, _ in raw_results.values())
                
                if any_success:
                    try:
                        os.remove(video_path)
                        print(f"🗑️  Deleted video: {filename}")
                        item["posted"] = True
                        item["post_results"] = formatted_results
                    except Exception as e:
                        print(f"⚠️  Could not delete video: {e}")
                
                save_metrics(metrics)
                return jsonify({
                    "results": formatted_results,
                    "video_deleted": any_success
                })
                
            else:  # LEFT - just mark as reviewed
                save_metrics(metrics)
                return jsonify({"message": "Marked for review"})
    
    return jsonify({"error": "Video not found in metrics"}), 404

@app.route("/api/stats")
def stats():
    """Get statistics about videos."""
    data = load_latest_metrics()
    video_files = set(os.listdir(VIDEO_FOLDER))
    
    return jsonify({
        "total": len(data),
        "reviewed": len([v for v in data if v.get("reviewed")]),
        "posted": len([v for v in data if v.get("posted")]),
        "pending": len(get_sorted_videos()),
        "videos_on_disk": len([f for f in video_files if f.endswith('.mp4')])
    })

@app.route("/api/debug")
def debug():
    """Debug endpoint to check configuration."""
    return jsonify({
        "video_folder_exists": os.path.exists(VIDEO_FOLDER),
        "video_folder": VIDEO_FOLDER,
        "metrics_files": glob.glob(METRICS_PATTERN),
        "youtube_secrets_exists": os.path.exists(YOUTUBE_SECRETS),
        "instagram_configured": INSTAGRAM_USERNAME != "YOUR_USERNAME"
    })

# ================= HTML TEMPLATE =================

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>Reel Swiper</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Inter', sans-serif;
            background: #0a0a0a;
            color: white;
            height: 100vh;
            overflow: hidden;
            touch-action: pan-y pinch-zoom;
        }
        
        .app {
            max-width: 400px;
            margin: 0 auto;
            height: 100vh;
            display: flex;
            flex-direction: column;
            background: #0a0a0a;
            position: relative;
        }
        
        .header {
            padding: 16px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: #0a0a0a;
            border-bottom: 1px solid #1a1a1a;
        }
        
        .logo {
            font-weight: 700;
            font-size: 20px;
            background: linear-gradient(135deg, #ff3366, #ffcc00);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .stats {
            display: flex;
            gap: 16px;
            font-size: 12px;
            color: #666;
        }
        
        .stats div {
            text-align: center;
        }
        
        .stats strong {
            display: block;
            color: white;
            font-size: 16px;
        }
        
        .card-container {
            flex: 1;
            position: relative;
            display: flex;
            justify-content: center;
            align-items: center;
            overflow: hidden;
        }
        
        .card {
            position: absolute;
            width: 90%;
            max-width: 320px;
            height: 70vh;
            max-height: 560px;
            background: #1a1a1a;
            border-radius: 24px;
            overflow: hidden;
            box-shadow: 0 20px 40px rgba(0,0,0,0.8);
            transition: transform 0.3s ease;
            cursor: grab;
            user-select: none;
            touch-action: none;
        }
        
        .card:active {
            cursor: grabbing;
        }
        
        .card video {
            width: 100%;
            height: 100%;
            object-fit: cover;
            pointer-events: none;
        }
        
        .card-overlay {
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            padding: 24px;
            background: linear-gradient(to top, rgba(0,0,0,0.9), transparent);
        }
        
        .card-id {
            font-size: 12px;
            color: #ffcc00;
            margin-bottom: 8px;
            opacity: 0.8;
        }
        
        .card-likes {
            font-size: 32px;
            font-weight: 700;
            margin-bottom: 4px;
        }
        
        .card-likes span {
            font-size: 14px;
            color: #666;
            margin-left: 4px;
        }
        
        .card-comments {
            font-size: 14px;
            color: #999;
        }
        
        .stamp {
            position: absolute;
            top: 40px;
            font-size: 32px;
            font-weight: 800;
            padding: 8px 24px;
            border: 4px solid;
            border-radius: 12px;
            transform: rotate(-15deg);
            opacity: 0;
            pointer-events: none;
            z-index: 10;
        }
        
        .stamp.post {
            right: 30px;
            color: #00ff88;
            border-color: #00ff88;
            transform: rotate(15deg);
        }
        
        .stamp.review {
            left: 30px;
            color: #ff3366;
            border-color: #ff3366;
            transform: rotate(-15deg);
        }
        
        .actions {
            display: flex;
            justify-content: center;
            gap: 24px;
            padding: 20px;
            background: #0a0a0a;
            border-top: 1px solid #1a1a1a;
        }
        
        .btn {
            width: 64px;
            height: 64px;
            border-radius: 50%;
            border: none;
            font-size: 24px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s, background 0.2s;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .btn:active {
            transform: scale(0.9);
        }
        
        .btn-review {
            background: rgba(255, 51, 102, 0.1);
            border: 2px solid #ff3366;
            color: #ff3366;
        }
        
        .btn-post {
            background: rgba(0, 255, 136, 0.1);
            border: 2px solid #00ff88;
            color: #00ff88;
        }
        
        .btn-skip {
            background: rgba(255, 204, 0, 0.1);
            border: 2px solid #ffcc00;
            color: #ffcc00;
        }
        
        .action-labels {
            display: flex;
            justify-content: center;
            gap: 24px;
            padding: 0 20px 16px;
            font-size: 12px;
            color: #666;
        }
        
        .action-labels div {
            width: 64px;
            text-align: center;
        }
        
        .toast {
            position: fixed;
            bottom: 100px;
            left: 50%;
            transform: translateX(-50%);
            background: #1a1a1a;
            color: white;
            padding: 12px 24px;
            border-radius: 30px;
            font-size: 14px;
            border: 1px solid #333;
            box-shadow: 0 10px 30px rgba(0,0,0,0.5);
            z-index: 1000;
            transition: opacity 0.3s;
            opacity: 0;
            pointer-events: none;
            max-width: 80%;
            text-align: center;
        }
        
        .toast.show {
            opacity: 1;
        }
        
        .empty-state {
            display: none;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 100%;
            text-align: center;
            color: #666;
        }
        
        .empty-state h2 {
            color: #ffcc00;
            margin-bottom: 8px;
        }
    </style>
</head>
<body>
    <div class="app">
        <div class="header">
            <div class="logo">REEL SWIPER</div>
            <div class="stats">
                <div><strong id="statPending">0</strong>queue</div>
                <div><strong id="statPosted">0</strong>posted</div>
                <div><strong id="statReviewed">0</strong>review</div>
            </div>
        </div>
        
        <div class="card-container" id="cardContainer">
            <div class="empty-state" id="emptyState">
                <h2>ALL DONE</h2>
                <p>No more videos to review</p>
            </div>
        </div>
        
        <div class="actions">
            <button class="btn btn-review" onclick="swipe('left')">←</button>
            <button class="btn btn-post" onclick="swipe('right')">✓</button>
            <button class="btn btn-skip" onclick="skip()">⏭</button>
        </div>
        <div class="action-labels">
            <div>REVIEW</div>
            <div>POST</div>
            <div>SKIP</div>
        </div>
    </div>
    
    <div class="toast" id="toast"></div>
    
    <script>
        let videos = [];
        let currentIndex = 0;
        let isDragging = false;
        let startX, startY, currentX = 0;
        let currentCard = null;
        
        async function loadVideos() {
            try {
                const response = await fetch('/api/videos');
                videos = await response.json();
                currentIndex = 0;
                render();
                loadStats();
            } catch (error) {
                showToast('Error loading videos');
            }
        }
        
        async function loadStats() {
            try {
                const response = await fetch('/api/stats');
                const stats = await response.json();
                document.getElementById('statPending').textContent = stats.pending || 0;
                document.getElementById('statPosted').textContent = stats.posted || 0;
                document.getElementById('statReviewed').textContent = stats.reviewed || 0;
            } catch (error) {
                console.error('Error loading stats');
            }
        }
        
        function render() {
            const container = document.getElementById('cardContainer');
            const emptyState = document.getElementById('emptyState');
            
            // Remove all cards
            const existingCards = container.querySelectorAll('.card');
            existingCards.forEach(card => card.remove());
            
            if (currentIndex >= videos.length) {
                emptyState.style.display = 'flex';
                return;
            }
            
            emptyState.style.display = 'none';
            
            // Create current card
            const video = videos[currentIndex];
            const card = createCard(video);
            container.appendChild(card);
            
            // Set up drag handlers
            setupDrag(card);
            
            // Start playing video
            const videoElement = card.querySelector('video');
            videoElement.play().catch(e => console.log('Autoplay prevented'));
        }
        
        function createCard(video) {
            const card = document.createElement('div');
            card.className = 'card';
            card.innerHTML = `
                <video src="/api/video/${encodeURIComponent(video.filename)}" loop muted playsinline></video>
                <div class="stamp post">POST</div>
                <div class="stamp review">REVIEW</div>
                <div class="card-overlay">
                    <div class="card-id">${video.reelId || 'Unknown'}</div>
                    <div class="card-likes">${formatNumber(video.likes || 0)}<span>likes</span></div>
                    <div class="card-comments">${formatNumber(video.comments || 0)} comments</div>
                </div>
            `;
            return card;
        }
        
        function setupDrag(card) {
            currentCard = card;
            
            const onDragStart = (e) => {
                e.preventDefault();
                isDragging = true;
                const point = e.touches ? e.touches[0] : e;
                startX = point.clientX;
                startY = point.clientY;
                card.style.transition = 'none';
                
                document.addEventListener('mousemove', onDragMove);
                document.addEventListener('touchmove', onDragMove, { passive: false });
                document.addEventListener('mouseup', onDragEnd);
                document.addEventListener('touchend', onDragEnd);
            };
            
            const onDragMove = (e) => {
                if (!isDragging) return;
                e.preventDefault();
                
                const point = e.touches ? e.touches[0] : e;
                const dx = point.clientX - startX;
                const dy = (point.clientY - startY) * 0.3;
                
                currentX = dx;
                
                card.style.transform = `translate(${dx}px, ${dy}px) rotate(${dx * 0.05}deg)`;
                
                // Show stamps based on direction
                const opacity = Math.min(Math.abs(dx) / 100, 0.8);
                if (dx > 20) {
                    card.querySelector('.stamp.post').style.opacity = opacity;
                    card.querySelector('.stamp.review').style.opacity = 0;
                } else if (dx < -20) {
                    card.querySelector('.stamp.review').style.opacity = opacity;
                    card.querySelector('.stamp.post').style.opacity = 0;
                } else {
                    card.querySelector('.stamp.post').style.opacity = 0;
                    card.querySelector('.stamp.review').style.opacity = 0;
                }
            };
            
            const onDragEnd = () => {
                if (!isDragging) return;
                isDragging = false;
                
                document.removeEventListener('mousemove', onDragMove);
                document.removeEventListener('touchmove', onDragMove);
                document.removeEventListener('mouseup', onDragEnd);
                document.removeEventListener('touchend', onDragEnd);
                
                if (Math.abs(currentX) > 120) {
                    const direction = currentX > 0 ? 'right' : 'left';
                    swipe(direction);
                } else {
                    // Reset position
                    card.style.transition = 'transform 0.3s ease';
                    card.style.transform = '';
                    card.querySelector('.stamp.post').style.opacity = 0;
                    card.querySelector('.stamp.review').style.opacity = 0;
                }
                
                currentX = 0;
            };
            
            card.addEventListener('mousedown', onDragStart);
            card.addEventListener('touchstart', onDragStart, { passive: false });
        }
        
        async function swipe(direction) {
            if (currentIndex >= videos.length) return;
            
            const video = videos[currentIndex];
            const card = document.querySelector('.card');
            
            if (card) {
                // Animate card out
                card.style.transition = 'transform 0.3s ease, opacity 0.3s ease';
                const translateX = direction === 'right' ? 500 : -500;
                card.style.transform = `translate(${translateX}px, -50px) rotate(${direction === 'right' ? 15 : -15}deg)`;
                card.style.opacity = '0';
            }
            
            // Make API call
            try {
                const response = await fetch('/api/swipe', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        filename: video.filename,
                        direction: direction
                    })
                });
                
                const result = await response.json();
                
                if (direction === 'right') {
                    const platforms = result.results || {};
                    const successCount = Object.values(platforms).filter(r => r.includes('✅')).length;
                    const failCount = Object.values(platforms).length - successCount;
                    
                    let message = `Posted to ${successCount} platform(s)`;
                    if (failCount > 0) message += `, ${failCount} failed`;
                    if (result.video_deleted) message += ' · Video deleted';
                    showToast(message);
                } else {
                    showToast('Marked for review');
                }
            } catch (error) {
                showToast('Error processing swipe');
            }
            
            // Move to next video
            setTimeout(() => {
                currentIndex++;
                render();
                loadStats();
            }, 300);
        }
        
        function skip() {
            if (currentIndex < videos.length - 1) {
                currentIndex++;
                render();
                showToast('Skipped');
            } else if (currentIndex === videos.length - 1) {
                currentIndex++;
                render();
            }
        }
        
        function formatNumber(num) {
            if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
            if (num >= 1000) return (num / 1000).toFixed(1) + 'K';
            return num.toString();
        }
        
        function showToast(message) {
            const toast = document.getElementById('toast');
            toast.textContent = message;
            toast.classList.add('show');
            setTimeout(() => toast.classList.remove('show'), 3000);
        }
        
        // Keyboard controls
        document.addEventListener('keydown', (e) => {
            if (e.key === 'ArrowRight') swipe('right');
            if (e.key === 'ArrowLeft') swipe('left');
            if (e.key === 'ArrowDown') skip();
        });
        
        // Initialize
        loadVideos();
        
        // Refresh stats periodically
        setInterval(loadStats, 5000);
    </script>
</body>
</html>
"""

# ================= MAIN =================

if __name__ == "__main__":
    # Check if video folder exists
    if not os.path.exists(VIDEO_FOLDER):
        print(f"⚠️  WARNING: Video folder not found: {VIDEO_FOLDER}")
        print(f"   Creating it for you...")
        os.makedirs(VIDEO_FOLDER, exist_ok=True)
    
    # Check for metrics files
    metrics_files = glob.glob(METRICS_PATTERN)
    if not metrics_files:
        print(f"⚠️  WARNING: No metrics files found matching: {METRICS_PATTERN}")
    
    # Check platform configurations
    if INSTAGRAM_USERNAME == "YOUR_USERNAME":
        print("⚠️  Instagram not configured - update credentials to enable posting")
    
    if not os.path.exists(YOUTUBE_SECRETS):
        print(f"⚠️  YouTube not configured - client_secrets.json not found")
    
    print("\n" + "="*50)
    print("🚀 Server starting...")
    print("📱 Open http://localhost:5000 in your browser")
    print("   or http://[your-tailscale-ip]:5000 from other devices")
    print("="*50 + "\n")
    
    # Run the app
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)