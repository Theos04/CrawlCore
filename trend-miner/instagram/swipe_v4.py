import os
import json
import glob
import threading
import datetime
import random
import time
import traceback
from collections import defaultdict
from flask import Flask, jsonify, request, send_from_directory, Response, redirect, url_for, render_template_string
from instagrapi import Client
from instagrapi.exceptions import LoginRequired, ClientError, ChallengeRequired, TwoFactorRequired
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow, InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import pickle
import subprocess
import time

# ================= CONFIGURATION =================

# Video and metrics paths
VIDEO_FOLDER = r"D:\trend_miner\instagram\downloads"
METRICS_PATTERN = r"E:\reels_1527_1771839673506.json"
YOUTUBE_SECRETS = r"D:\trend_miner\instagram\Tejas\client_secrets.json"

# Instagram credentials - ONLY PASSWORD LOGIN, NO SESSION ID
INSTAGRAM_USERNAME = "chaosinclips"
INSTAGRAM_PASSWORD = "Sakal8983!"

# Session files for auth persistence
INSTAGRAM_SESSION = "instagram_session.pkl"
YOUTUBE_TOKEN = "youtube_token.pickle"

# Thread locks for platform operations
instagram_lock = threading.Lock()
youtube_lock = threading.Lock()

# ================= POSTING STRATEGY (DISABLED FOR TESTING) =================

class PostingStrategy:
    def __init__(self):
        self.daily_limits = {
            'instagram': 10,
            'youtube': 5,
            'youtube_long': 2
        }
        
        self.best_times = {
            'instagram': [7, 8, 9, 12, 13, 17, 18, 19, 20, 21],
            'youtube': [5, 6, 7, 11, 12, 14, 15, 17, 18, 19, 20],
            'youtube_long': [5, 6, 17, 18, 19, 20, 21]
        }
        
        self.scoring_weights = {
            'likes': 0.35,
            'comments': 0.25,
            'recency': 0.25,
            'engagement_rate': 0.15
        }
        
        self.posting_history = self.load_posting_history()
        self.post_queue = []
        self.currently_processing = False
        
    def load_posting_history(self):
        history_file = 'posting_history.json'
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r') as f:
                    return json.load(f)
            except:
                return {
                    'posts': [],
                    'daily_counts': {},
                    'platform_performance': {}
                }
        return {
            'posts': [],
            'daily_counts': {},
            'platform_performance': {}
        }
    
    def save_posting_history(self):
        try:
            with open('posting_history.json', 'w') as f:
                json.dump(self.posting_history, f, indent=2)
        except:
            pass
    
    def calculate_content_score(self, reel_data):
        likes = reel_data.get('likes', 0)
        comments = reel_data.get('comments', 0)
        
        engagement_rate = (comments / max(likes, 1)) * 1000
        
        if 'time' in reel_data and reel_data['time']:
            try:
                time_str = reel_data['time'].replace('Z', '+00:00')
                post_time = datetime.datetime.fromisoformat(time_str)
                days_old = (datetime.datetime.now(datetime.timezone.utc) - post_time).days
                recency_score = max(0, 1 - (days_old / 30))
            except:
                recency_score = 0.7
        else:
            recency_score = 0.7
        
        likes_score = min(1, likes / 50000)
        comments_score = min(1, comments / 2500)
        engagement_score = min(1, engagement_rate / 50)
        
        score = (
            likes_score * self.scoring_weights['likes'] +
            comments_score * self.scoring_weights['comments'] +
            recency_score * self.scoring_weights['recency'] +
            engagement_score * self.scoring_weights['engagement_rate']
        )
        
        return round(min(1, score), 3)
    
    def should_post_now(self, platform, content_score):
        # DISABLED FOR TESTING - Always return True to post
        return True, "Ready to post (strategy disabled for testing)"
    
    def get_last_post_time(self, platform):
        platform_posts = [p for p in self.posting_history.get('posts', []) 
                         if p['platform'] == platform]
        if platform_posts:
            last = max(platform_posts, key=lambda x: x['timestamp'])
            return datetime.datetime.fromisoformat(last['timestamp'])
        return None
    
    def record_post(self, reel_id, platform, success, metadata=None):
        now = datetime.datetime.now()
        now_str = now.isoformat()
        today = now.strftime('%Y-%m-%d')
        
        post_record = {
            'reel_id': reel_id,
            'platform': platform,
            'timestamp': now_str,
            'success': success,
            'metadata': metadata or {}
        }
        
        if 'posts' not in self.posting_history:
            self.posting_history['posts'] = []
        self.posting_history['posts'].append(post_record)
        
        if 'daily_counts' not in self.posting_history:
            self.posting_history['daily_counts'] = {}
        if today not in self.posting_history['daily_counts']:
            self.posting_history['daily_counts'][today] = {}
        
        self.posting_history['daily_counts'][today][platform] = \
            self.posting_history['daily_counts'][today].get(platform, 0) + 1
        
        if len(self.posting_history['posts']) > 500:
            self.posting_history['posts'] = self.posting_history['posts'][-500:]
        
        self.save_posting_history()
    
    def get_platform_recommendations(self, reel_data):
        content_score = self.calculate_content_score(reel_data)
        recommendations = []
        
        duration = reel_data.get('duration', 0)
        is_long_form = duration > 60
        
        # Always return True for testing
        recommendations.append({
            'platform': 'instagram',
            'should_post': True,
            'reason': 'Testing mode - always post',
            'content_score': content_score,
            'optimal_time': True
        })
        
        youtube_platform = 'youtube_long' if is_long_form else 'youtube'
        recommendations.append({
            'platform': 'youtube',
            'content_type': 'long' if is_long_form else 'short',
            'should_post': True,
            'reason': 'Testing mode - always post',
            'content_score': content_score,
            'optimal_time': True
        })
        
        return recommendations, content_score

# ================= INSTAGRAM POSTER (PASSWORD ONLY) =================

class InstagramPoster:
    def __init__(self, username, password, session_file=INSTAGRAM_SESSION):
        self.username = username
        self.password = password
        self.session_file = session_file
        self.client = None
        self.login()
    
    def login(self):
        """Login to Instagram using only password (no session ID)"""
        try:
            self.client = Client()
            self.client.delay_range = [2, 5]  # Increased delay to avoid rate limiting
            
            # Set realistic device settings
            self.client.set_device({
                "app_version": "269.0.0.18.75",
                "android_version": 26,
                "android_release": "8.0.0",
                "dpi": "480dpi",
                "resolution": "1080x1920",
                "manufacturer": "OnePlus",
                "device": "devitron",
                "model": "6T Dev",
                "cpu": "qcom",
                "version_code": "314665256",
            })
            self.client.set_user_agent(
                "Instagram 269.0.0.18.75 Android (26/8.0.0; 480dpi; 1080x1920; OnePlus; ONEPLUS A6000; devitron; qcom; en_US)"
            )
            self.client.set_locale("en_US")
            self.client.set_country(1)
            
            # Try saved session first
            if os.path.exists(self.session_file):
                try:
                    with open(self.session_file, 'rb') as f:
                        session_data = pickle.load(f)
                        self.client.set_settings(session_data)
                    self.client.get_timeline_feed()
                    print("✅ Instagram: Logged in using saved session")
                    return
                except Exception as e:
                    print(f"⚠️ Instagram: Session expired, logging in with password...")
            
            # Password login only
            print("🔄 Logging in with password...")
            time.sleep(random.uniform(3, 6))
            self.client.login(self.username, self.password)
            
            # Save session for next time
            with open(self.session_file, 'wb') as f:
                pickle.dump(self.client.get_settings(), f)
            print("✅ Instagram: Logged in successfully")
            
        except ChallengeRequired:
            print("\n" + "="*60)
            print("⚠️ INSTAGRAM CHALLENGE REQUIRED")
            print("="*60)
            print("Instagram needs verification. Please:")
            print("1. Open Instagram app on your phone")
            print("2. Check for any security notifications")
            print("3. Complete the verification")
            print("4. Wait 2 minutes and restart this script")
            print("="*60 + "\n")
            self.client = None
            
        except TwoFactorRequired:
            print("\n" + "="*60)
            print("⚠️ TWO-FACTOR AUTHENTICATION REQUIRED")
            print("="*60)
            print("2FA is enabled. Please:")
            print("1. Get the code from your authenticator app")
            print("2. Enter it when prompted")
            print("="*60 + "\n")
            try:
                code = input("Enter 2FA code: ")
                self.client.login(self.username, self.password, verification_code=code)
                with open(self.session_file, 'wb') as f:
                    pickle.dump(self.client.get_settings(), f)
                print("✅ Instagram: 2FA successful")
            except Exception as e:
                print(f"❌ 2FA failed: {e}")
                self.client = None
                
        except Exception as e:
            print(f"❌ Instagram login failed: {e}")
            self.client = None

    def post_reel(self, video_path, caption=""):
        """Post a reel to Instagram with proper error handling"""
        if not self.client:
            return False, "Instagram not logged in"
        
        try:
            print(f"📁 Checking video: {os.path.basename(video_path)}")
            if not os.path.exists(video_path):
                return False, f"Video file not found"
            
            file_size = os.path.getsize(video_path) / (1024 * 1024)
            print(f"📊 Video size: {file_size:.1f}MB")
            
            if file_size > 100:
                return False, f"Video too large: {file_size:.1f}MB"
            
            print(f"📤 Uploading to Instagram...")
            print("⏳ This may take 1-2 minutes...")
            
            # Upload with retry logic
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    media = self.client.clip_upload(
                        video_path,
                        caption=caption[:2200],
                        thumbnail=None  # Let Instagram generate thumbnail
                    )
                    
                    if media and hasattr(media, 'pk'):
                        print(f"✅ Success! Media ID: {media.pk}")
                        return True, f"Posted! ID: {media.pk}"
                    else:
                        if attempt < max_retries - 1:
                            print(f"🔄 Retry {attempt + 2}/{max_retries}...")
                            time.sleep(5)
                            continue
                        return False, "Upload failed - no media returned"
                        
                except ChallengeRequired:
                    print("⚠️ Challenge required - complete in Instagram app")
                    return False, "Challenge required"
                except Exception as e:
                    if attempt < max_retries - 1:
                        print(f"🔄 Retry {attempt + 2}/{max_retries}...")
                        time.sleep(5)
                        continue
                    raise e
                    
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Instagram error: {error_msg[:100]}")
            return False, f"Error: {error_msg[:100]}"

# ================= YOUTUBE POSTER =================

class YouTubePoster:
    def __init__(self, client_secrets_file, token_file=YOUTUBE_TOKEN):
        self.client_secrets_file = client_secrets_file
        self.token_file = token_file
        self.service = None
        self.authenticate()
    
    def authenticate(self):
        SCOPES = ['https://www.googleapis.com/auth/youtube.upload']
        creds = None
        
        if os.path.exists(self.token_file):
            try:
                with open(self.token_file, 'rb') as token:
                    creds = pickle.load(token)
                print("✅ YouTube: Loaded saved token")
            except:
                print("⚠️ YouTube: Could not load saved token")
        
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                    print("✅ YouTube: Token refreshed")
                except:
                    creds = None
            
            if not creds:
                try:
                    print("\n🔄 YouTube authorization required...")
                    flow = InstalledAppFlow.from_client_secrets_file(
                        self.client_secrets_file, 
                        SCOPES,
                        redirect_uri='http://localhost:8080/'
                    )
                    creds = flow.run_local_server(port=8080, open_browser=True)
                    print("✅ YouTube: New authentication successful")
                except Exception as e:
                    print(f"❌ YouTube: Authentication failed: {e}")
                    return
            
            try:
                with open(self.token_file, 'wb') as token:
                    pickle.dump(creds, token)
                print("✅ YouTube: Token saved")
            except:
                pass
        
        try:
            self.service = build('youtube', 'v3', credentials=creds)
            print("✅ YouTube: Service built successfully")
        except Exception as e:
            print(f"❌ YouTube: Failed to build service: {e}")
    
    def post_video(self, video_path, title, description="", is_short=True):
        """Post video to YouTube"""
        if not self.service:
            return False, "YouTube not authenticated"
        
        if not os.path.exists(video_path):
            return False, f"Video file not found"
        
        try:
            file_size = os.path.getsize(video_path) / (1024 * 1024)
            print(f"📊 Video size: {file_size:.1f}MB")
            
            body = {
                'snippet': {
                    'title': title[:100],
                    'description': description[:5000],
                    'tags': ['reels', 'shorts', 'viral', 'trending'] if is_short else ['video', 'content'],
                    'categoryId': '22'
                },
                'status': {
                    'privacyStatus': 'public',
                    'selfDeclaredMadeForKids': False
                }
            }
            
            if is_short:
                body['snippet']['tags'].append('#Shorts')
            
            print(f"📤 Uploading to YouTube...")
            print("⏳ This may take a few minutes...")
            
            media = MediaFileUpload(video_path, mimetype='video/mp4', resumable=True)
            insert_request = self.service.videos().insert(
                part=','.join(body.keys()),
                body=body,
                media_body=media
            )
            
            response = None
            while response is None:
                status, response = insert_request.next_chunk()
                if status:
                    print(f"🎬 Progress: {int(status.progress() * 100)}%")
            
            if response and 'id' in response:
                video_url = f"https://youtu.be/{response['id']}"
                print(f"✅ Success! {video_url}")
                return True, f"Uploaded! {video_url}"
            else:
                return False, "Upload failed - no video ID returned"
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ YouTube error: {error_msg[:100]}")
            return False, f"Error: {error_msg[:100]}"

# ================= FLASK APP =================

app = Flask(__name__)
posting_strategy = PostingStrategy()

print("\n" + "=" * 60)
print("🚀 REEL SWIPER - Auto Posting Enabled (TESTING MODE)")
print("=" * 60)

print("\n📱 Initializing Instagram (password only)...")
instagram_poster = InstagramPoster(INSTAGRAM_USERNAME, INSTAGRAM_PASSWORD)

print("\n🎬 Initializing YouTube...")
youtube_poster = YouTubePoster(YOUTUBE_SECRETS)

print("\n📋 Posting Strategy:")
print(f"   • Instagram: {posting_strategy.daily_limits['instagram']} posts/day")
print(f"   • YouTube Shorts: {posting_strategy.daily_limits['youtube']} posts/day")
print(f"   • YouTube Long: {posting_strategy.daily_limits['youtube_long']} posts/day")
print("   ⚠️ TESTING MODE: Strategy disabled - will always post")
print("=" * 60)

# ================= UTILITIES =================

def load_metrics():
    files = sorted(glob.glob(METRICS_PATTERN), key=os.path.getmtime, reverse=True)
    if not files:
        return {}
    try:
        with open(files[0], 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            return {r.get("reelId", str(i)): r for i, r in enumerate(data)}
        return data
    except Exception as e:
        print(f"⚠️ Error loading metrics: {e}")
        return {}

def save_metrics(metrics_dict):
    files = sorted(glob.glob(METRICS_PATTERN), key=os.path.getmtime, reverse=True)
    if not files:
        return
    data = list(metrics_dict.values())
    try:
        with open(files[0], 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"⚠️ Error saving metrics: {e}")

def get_video_duration(path):
    try:
        result = subprocess.run([
            'ffprobe', '-v', 'error', '-show_entries',
            'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', path
        ], capture_output=True, text=True, timeout=5)
        return float(result.stdout.strip())
    except:
        return 0

def force_delete_file(file_path, max_attempts=3):
    """Force delete a file with multiple attempts"""
    for attempt in range(max_attempts):
        try:
            if os.path.exists(file_path):
                # Wait a bit for any handles to close
                time.sleep(2)
                os.remove(file_path)
                print(f"✅ Deleted: {os.path.basename(file_path)}")
                return True
        except Exception as e:
            if attempt < max_attempts - 1:
                print(f"⚠️ Delete attempt {attempt + 1} failed, retrying...")
                time.sleep(3)
            else:
                print(f"⚠️ Could not delete: {os.path.basename(file_path)}")
    return False

def get_video_list():
    mp4s = glob.glob(os.path.join(VIDEO_FOLDER, "*.mp4"))
    metrics = load_metrics()
    videos = []
    
    for path in mp4s:
        fname = os.path.basename(path)
        parts = fname.replace(".mp4", "").split("_")
        reel_id = parts[-1] if len(parts) >= 3 else fname.replace(".mp4", "")
        
        m = metrics.get(reel_id, {})
        
        if m.get("status") in ("posted", "review", "scheduled"):
            continue
        
        duration = get_video_duration(path)
        content_score = posting_strategy.calculate_content_score(m)
        
        videos.append({
            "reelId": reel_id,
            "filename": fname,
            "likes": m.get("likes", random.randint(1000, 50000)),
            "comments": m.get("comments", random.randint(50, 2000)),
            "time": m.get("time", ""),
            "duration": duration,
            "caption": m.get("caption", "")[:100],
            "music": m.get("music", ""),
            "content_score": content_score,
            "is_long_form": duration > 60
        })
    
    videos.sort(key=lambda v: v["content_score"], reverse=True)
    return videos

# ================= AUTO POSTING FUNCTIONS =================

def post_to_instagram(video_path, caption=""):
    with instagram_lock:
        print(f"\n{'='*40}")
        print(f"📸 INSTAGRAM POST")
        print(f"{'='*40}")
        
        if not instagram_poster or not instagram_poster.client:
            return False, "Instagram not logged in"
        
        full_caption = f"{caption}\n\n#reels #viral #trending #shorts"
        success, message = instagram_poster.post_reel(video_path, full_caption)
        
        if success:
            print(f"✅ Instagram: {message}")
        else:
            print(f"❌ Instagram: {message}")
        return success, message

def post_to_youtube(video_path, caption="", is_short=True):
    with youtube_lock:
        print(f"\n{'='*40}")
        print(f"🎬 YOUTUBE POST")
        print(f"{'='*40}")
        
        if not youtube_poster or not youtube_poster.service:
            return False, "YouTube not authenticated"
        
        filename = os.path.basename(video_path)
        title = f"Reel {filename[:30]} - {'Shorts' if is_short else 'Video'}"
        description = f"{caption}\n\n#reels #shorts #viral #trending" if is_short else caption
        
        success, message = youtube_poster.post_video(video_path, title, description, is_short)
        
        if success:
            print(f"✅ YouTube: {message}")
        else:
            print(f"❌ YouTube: {message}")
        return success, message

def post_to_all(video_path, reel_data):
    print(f"\n{'#'*60}")
    print(f"🚀 POSTING: {os.path.basename(video_path)}")
    print(f"{'#'*60}")
    
    results = {}
    caption = reel_data.get('caption', '')
    
    # Post to Instagram
    print("\n📱 Instagram...")
    success, msg = post_to_instagram(video_path, caption)
    results['Instagram'] = ('✅ ' + msg) if success else ('❌ ' + msg[:80])
    
    # Post to YouTube
    is_short = not reel_data.get('is_long_form', False)
    print("\n🎬 YouTube...")
    success, msg = post_to_youtube(video_path, caption, is_short)
    results[f"YouTube"] = ('✅ ' + msg) if success else ('❌ ' + msg[:80])
    
    # Record posts
    posting_strategy.record_post(reel_data.get('reelId', 'unknown'), 'instagram', '✅' in results['Instagram'], {'caption': caption[:50]})
    posting_strategy.record_post(reel_data.get('reelId', 'unknown'), 'youtube', '✅' in results['YouTube'], {'caption': caption[:50]})
    
    print(f"\n📊 Results: {results}")
    return results

# ================= ROUTES =================

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/videos')
def api_videos():
    try:
        videos = get_video_list()
        return jsonify(videos)
    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify([])

@app.route('/api/video/<filename>')
def api_video(filename):
    try:
        safe_filename = os.path.basename(filename)
        return send_from_directory(VIDEO_FOLDER, safe_filename)
    except Exception as e:
        return jsonify({"error": str(e)}), 404

@app.route('/api/swipe', methods=['POST'])
def api_swipe():
    try:
        data = request.json
        reel_id = data['reelId']
        filename = data['filename']
        direction = data['direction']
        
        print(f"\n{'='*60}")
        print(f"👉 SWIPE {direction.upper()}: {filename}")
        print(f"{'='*60}")
        
        metrics = load_metrics()
        video_path = os.path.join(VIDEO_FOLDER, filename)
        
        if not os.path.exists(video_path):
            return jsonify({"error": "Video not found"}), 404
        
        if direction == 'right':
            reel_data = metrics.get(reel_id, {})
            if not reel_data:
                reel_data = {
                    "reelId": reel_id,
                    "likes": random.randint(1000, 50000),
                    "comments": random.randint(50, 2000),
                    "time": datetime.datetime.now().isoformat(),
                    "duration": get_video_duration(video_path)
                }
            
            # Post to platforms
            results = post_to_all(video_path, reel_data)
            successful_posts = any('✅' in str(result) for result in results.values())
            
            # Update metrics
            if reel_id in metrics:
                metrics[reel_id]['status'] = 'posted' if successful_posts else 'failed'
                metrics[reel_id]['post_results'] = results
                metrics[reel_id]['posted_at'] = datetime.datetime.now().isoformat()
            else:
                metrics[reel_id] = {
                    "reelId": reel_id,
                    "status": 'posted' if successful_posts else 'failed',
                    "post_results": results,
                    "posted_at": datetime.datetime.now().isoformat()
                }
            
            save_metrics(metrics)
            
            # Try to delete if any post succeeded
            if successful_posts:
                force_delete_file(video_path)
            
            return jsonify({'status': 'success', 'results': results})
        
        elif direction == 'left':
            if reel_id in metrics:
                metrics[reel_id]['status'] = 'review'
                metrics[reel_id]['reviewed_at'] = datetime.datetime.now().isoformat()
            else:
                metrics[reel_id] = {
                    "reelId": reel_id,
                    "status": "review",
                    "reviewed_at": datetime.datetime.now().isoformat()
                }
            
            save_metrics(metrics)
            return jsonify({'status': 'review'})
        
        return jsonify({'error': 'Invalid direction'}), 400
    
    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/status')
def api_status():
    return jsonify({
        'instagram': {
            'connected': instagram_poster and instagram_poster.client is not None
        },
        'youtube': {
            'connected': youtube_poster and youtube_poster.service is not None
        }
    })

@app.route('/api/test/instagram')
def test_instagram():
    if instagram_poster and instagram_poster.client:
        return jsonify({'status': 'connected'})
    return jsonify({'status': 'disconnected'})

@app.route('/api/test/youtube')
def test_youtube():
    if youtube_poster and youtube_poster.service:
        return jsonify({'status': 'connected'})
    return jsonify({'status': 'disconnected'})

@app.route('/api/reauth/youtube')
def reauth_youtube():
    global youtube_poster
    if os.path.exists(YOUTUBE_TOKEN):
        os.remove(YOUTUBE_TOKEN)
    youtube_poster = YouTubePoster(YOUTUBE_SECRETS)
    return redirect(url_for('index'))

# ================= HTML TEMPLATE =================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Reel Swiper</title>
    <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/hammer.js/2.0.8/hammer.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            background: #0a0a0f; 
            color: white; 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            height: 100vh; 
            display: flex; 
            flex-direction: column; 
            overflow: hidden;
        }
        header { 
            padding: 12px 16px; 
            border-bottom: 1px solid #1e1e2e; 
            display: flex; 
            justify-content: space-between;
            background: rgba(10,10,15,0.95);
        }
        .logo { color: #b300ff; font-weight: bold; font-size: 18px; }
        .stats { display: flex; gap: 20px; font-size: 12px; color: #555570; }
        .stats span strong { color: white; font-size: 16px; }
        main { flex: 1; display: flex; align-items: center; justify-content: center; }
        #stack { 
            width: min(92vw, 360px); 
            height: min(70vh, 600px); 
            position: relative;
        }
        .card { 
            position: absolute; 
            width: 100%; 
            height: 100%; 
            background: #13131a; 
            border-radius: 24px; 
            border: 2px solid #1e1e2e; 
            overflow: hidden; 
            box-shadow: 0 20px 40px rgba(0,0,0,0.5);
            touch-action: none;
            transition: transform 0.3s ease;
            cursor: grab;
        }
        .card.dragging { transition: none; }
        video { 
            width: 100%; 
            height: 100%; 
            object-fit: cover; 
            pointer-events: none;
        }
        .overlay { 
            position: absolute; 
            inset: 0; 
            background: linear-gradient(to top, rgba(0,0,0,0.95), transparent);
        }
        .info { 
            position: absolute; 
            bottom: 0; 
            left: 0; 
            right: 0; 
            padding: 20px; 
        }
        .likes { 
            font-size: 32px; 
            font-weight: bold; 
            color: #ff3366;
        }
        .likes span { 
            font-size: 14px; 
            color: #8888a0; 
            margin-left: 8px; 
        }
        .actions { 
            display: flex; 
            justify-content: center; 
            gap: 20px; 
            padding: 16px; 
            background: rgba(10,10,15,0.95);
            border-top: 1px solid #1e1e2e;
        }
        button { 
            width: 64px; 
            height: 64px; 
            border-radius: 50%; 
            border: 2px solid; 
            font-size: 28px; 
            background: transparent; 
            cursor: pointer; 
        }
        .review { border-color: #ff3366; color: #ff3366; }
        .post { border-color: #00ff88; color: #00ff88; width: 72px; height: 72px; }
        .toast { 
            position: fixed; 
            top: 70px; 
            left: 50%; 
            transform: translateX(-50%) translateY(-100px); 
            background: #1a1a24; 
            border: 1px solid #2a2a35; 
            padding: 12px 24px; 
            border-radius: 40px; 
            transition: 0.3s; 
            z-index: 2000; 
        }
        .toast.show { transform: translateX(-50%) translateY(0); }
        .empty { display: none; text-align: center; color: #555570; padding: 40px; }
        .badge {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            display: inline-block;
        }
        .badge.connected { background: #00ff88; }
        .badge.disconnected { background: #ff3366; }
        .swipe-indicator {
            position: fixed;
            top: 50%;
            transform: translateY(-50%);
            padding: 12px 24px;
            border-radius: 40px;
            font-size: 24px;
            font-weight: bold;
            z-index: 1000;
            opacity: 0;
            pointer-events: none;
        }
        .swipe-indicator.right {
            right: 20px;
            background: rgba(0,255,136,0.2);
            color: #00ff88;
            border: 2px solid #00ff88;
        }
        .swipe-indicator.left {
            left: 20px;
            background: rgba(255,51,102,0.2);
            color: #ff3366;
            border: 2px solid #ff3366;
        }
        .swipe-indicator.show { opacity: 1; }
    </style>
</head>
<body>
    <header>
        <div class="logo">🎬 REEL SWIPER</div>
        <div class="stats">
            <span><strong id="queue">0</strong>QUEUE</span>
            <span><strong id="posted">0</strong>POSTED</span>
        </div>
        <div>
            <span class="badge" id="ig-status" title="Instagram"></span>
            <span class="badge" id="yt-status" title="YouTube"></span>
        </div>
    </header>
    
    <main>
        <div id="stack">
            <div class="loading" id="loading">Loading videos...</div>
            <div class="empty" id="empty">
                <h2>✨ ALL DONE</h2>
                <p>No more reels to process</p>
            </div>
        </div>
    </main>
    
    <div class="actions">
        <button class="review" onclick="manualSwipe('left')">↩</button>
        <button class="post" onclick="manualSwipe('right')">🚀</button>
    </div>
    
    <div class="toast" id="toast"></div>
    <div class="swipe-indicator" id="swipeIndicator"></div>

    <script>
        let videos = [];
        let isProcessing = false;
        let hammer = null;

        async function loadVideos() {
            try {
                const res = await fetch('/api/videos');
                videos = await res.json();
                document.getElementById('loading').style.display = 'none';
                updateQueue();
                if (videos.length > 0) {
                    renderCard(videos[0]);
                } else {
                    document.getElementById('empty').style.display = 'block';
                }
                checkStatus();
            } catch (e) {
                showToast('❌ Failed to load videos');
            }
        }

        function renderCard(video) {
            const stack = document.getElementById('stack');
            stack.innerHTML = '';
            
            const card = document.createElement('div');
            card.className = 'card';
            card.id = 'current-card';
            card.innerHTML = `
                <video src="/api/video/${encodeURIComponent(video.filename)}" loop muted autoplay playsinline></video>
                <div class="overlay"></div>
                <div class="info">
                    <div class="likes">${formatNumber(video.likes)}<span>likes</span></div>
                    <div style="color:#8888a0">${formatNumber(video.comments)} comments</div>
                    <div style="color:#b300ff; margin-top:8px">${video.caption || ''}</div>
                </div>
            `;
            
            stack.appendChild(card);
            setupSwipe(card, video);
            
            card.querySelector('video').play().catch(e => console.log('Autoplay failed:', e));
        }

        function setupSwipe(card, video) {
            if (hammer) hammer.destroy();
            
            hammer = new Hammer(card);
            hammer.get('pan').set({ direction: Hammer.DIRECTION_HORIZONTAL });
            
            const indicator = document.getElementById('swipeIndicator');
            
            hammer.on('pan', (e) => {
                if (isProcessing) return;
                
                card.style.transform = `translateX(${e.deltaX}px) rotate(${e.deltaX * 0.05}deg)`;
                
                if (e.deltaX > 50) {
                    indicator.textContent = '🚀 POST';
                    indicator.className = 'swipe-indicator right show';
                } else if (e.deltaX < -50) {
                    indicator.textContent = '↩ SKIP';
                    indicator.className = 'swipe-indicator left show';
                } else {
                    indicator.className = 'swipe-indicator';
                }
            });
            
            hammer.on('panend', (e) => {
                if (Math.abs(e.velocityX) > 0.5 || Math.abs(e.deltaX) > 80) {
                    completeSwipe(e.deltaX > 0 ? 'right' : 'left', video);
                } else {
                    card.style.transform = '';
                    indicator.className = 'swipe-indicator';
                }
            });
        }

        async function completeSwipe(direction, video) {
            if (isProcessing) return;
            isProcessing = true;
            
            const card = document.getElementById('current-card');
            card.style.transform = direction === 'right' 
                ? 'translateX(500px) rotate(20deg)' 
                : 'translateX(-500px) rotate(-20deg)';
            
            showToast(direction === 'right' ? '🚀 Posting...' : '💾 Saving...');
            
            try {
                const res = await fetch('/api/swipe', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        reelId: video.reelId,
                        filename: video.filename,
                        direction: direction
                    })
                });
                
                const data = await res.json();
                
                if (direction === 'right') {
                    const results = Object.values(data.results || {});
                    const success = results.some(r => r.includes('✅'));
                    showToast(success ? '✅ Posted!' : '❌ Failed');
                    if (success) {
                        document.getElementById('posted').textContent = 
                            parseInt(document.getElementById('posted').textContent || '0') + 1;
                    }
                } else {
                    showToast('📝 Saved');
                }
                
                videos = videos.filter(v => v.filename !== video.filename);
                updateQueue();
                
                setTimeout(() => {
                    if (videos.length > 0) {
                        renderCard(videos[0]);
                    } else {
                        document.getElementById('stack').innerHTML = '';
                        document.getElementById('empty').style.display = 'block';
                    }
                    isProcessing = false;
                }, 300);
                
            } catch (e) {
                showToast('❌ Error');
                isProcessing = false;
                card.style.transform = '';
            }
        }

        function manualSwipe(direction) {
            if (videos.length === 0 || isProcessing) return;
            completeSwipe(direction, videos[0]);
        }

        function formatNumber(num) {
            if (num >= 1000000) return (num/1000000).toFixed(1) + 'M';
            if (num >= 1000) return (num/1000).toFixed(1) + 'K';
            return num;
        }

        function showToast(msg) {
            const toast = document.getElementById('toast');
            toast.textContent = msg;
            toast.classList.add('show');
            setTimeout(() => toast.classList.remove('show'), 2000);
        }

        function updateQueue() {
            document.getElementById('queue').textContent = videos.length;
        }

        async function checkStatus() {
            try {
                const res = await fetch('/api/status');
                const status = await res.json();
                document.getElementById('ig-status').className = status.instagram.connected ? 'badge connected' : 'badge disconnected';
                document.getElementById('yt-status').className = status.youtube.connected ? 'badge connected' : 'badge disconnected';
            } catch (e) {}
        }

        document.addEventListener('keydown', (e) => {
            if (e.key === 'ArrowRight') manualSwipe('right');
            if (e.key === 'ArrowLeft') manualSwipe('left');
        });

        loadVideos();
        setInterval(checkStatus, 5000);
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    if not os.path.exists(VIDEO_FOLDER):
        os.makedirs(VIDEO_FOLDER, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("🚀 REEL SWIPER STARTING...")
    print("=" * 60)
    print(f"📁 Videos: {VIDEO_FOLDER}")
    print(f"📊 Metrics: {METRICS_PATTERN}")
    print("\n🌐 Open http://localhost:5000")
    print("\n💡 Swipe right → Post to Instagram & YouTube")
    print("   Swipe left  → Save for review")
    print("   Arrow keys  → Keyboard control")
    print("=" * 60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)