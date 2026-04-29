"""
poster.py — Instagram Reels + YouTube Shorts
---------------------------------------------
pip install requests google-api-python-client google-auth-oauthlib
Configure via config.json (run  python poster.py  to generate it)
"""

import os, json, time, requests
from pathlib import Path


def load_config():
    p = Path("config.json")
    return json.load(open(p)) if p.exists() else {}


# ── INSTAGRAM ─────────────────────────────────────────────────
# Setup: developers.facebook.com → Instagram Graph API
# Video MUST be at a public URL. Use ngrok: ngrok http 5000
# then set video_url_base = "https://xxx.ngrok.io/api/video"

def post_instagram(video_path, caption="", cfg={}):
    token  = cfg.get("access_token", "")
    uid    = cfg.get("user_id", "")
    base   = cfg.get("video_url_base", "")
    if not token or not uid:
        return False, "Missing access_token or user_id"
    if not base:
        return False, "Missing video_url_base (Instagram needs public URL — use ngrok)"

    url = base.rstrip("/") + "/" + Path(video_path).name
    print(f"    IG container: {url}")

    r = requests.post(
        f"https://graph.facebook.com/v19.0/{uid}/media",
        params={"media_type": "REELS", "video_url": url,
                "caption": caption, "access_token": token}
    ).json()
    if "error" in r:
        return False, r["error"]["message"]
    cid = r["id"]

    for i in range(15):
        time.sleep(4)
        s = requests.get(f"https://graph.facebook.com/v19.0/{cid}",
                         params={"fields": "status_code", "access_token": token}).json()
        code = s.get("status_code", "")
        print(f"    IG status: {code} ({i+1}/15)")
        if code == "FINISHED": break
        if code == "ERROR": return False, f"IG processing error: {s}"
    else:
        return False, "IG: timed out"

    pub = requests.post(
        f"https://graph.facebook.com/v19.0/{uid}/media_publish",
        params={"creation_id": cid, "access_token": token}
    ).json()
    if "id" in pub:
        return True, f"https://www.instagram.com/reel/{pub['id']}"
    return False, str(pub)


# ── YOUTUBE SHORTS ────────────────────────────────────────────
# Setup: console.cloud.google.com → Enable YouTube Data API v3
# → OAuth2 Desktop credentials → download as client_secrets.json

def post_youtube(video_path, caption="", cfg={}):
    secrets = cfg.get("client_secrets_file", "client_secrets.json")
    tokfile = cfg.get("token_file", "youtube_token.json")
    if not os.path.exists(secrets):
        return False, f"client_secrets.json not found at: {secrets}"
    try:
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaFileUpload

        SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
        creds  = None
        if os.path.exists(tokfile):
            creds = Credentials.from_authorized_user_file(tokfile, SCOPES)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow  = InstalledAppFlow.from_client_secrets_file(secrets, SCOPES)
                creds = flow.run_local_server(port=0)
            open(tokfile, "w").write(creds.to_json())

        yt   = build("youtube", "v3", credentials=creds)
        body = {
            "snippet": {"title": (caption[:100] or Path(video_path).stem),
                        "description": caption,
                        "tags": cfg.get("tags", ["shorts", "viral", "trending"]),
                        "categoryId": "22"},
            "status":  {"privacyStatus": cfg.get("privacy", "public"),
                        "selfDeclaredMadeForKids": False}
        }
        print(f"    YT: uploading {Path(video_path).name}")
        media = MediaFileUpload(video_path, mimetype="video/mp4",
                                resumable=True, chunksize=5*1024*1024)
        req = yt.videos().insert(part="snippet,status", body=body, media_body=media)
        resp = None
        while resp is None:
            st, resp = req.next_chunk()
            if st: print(f"    YT: {int(st.progress()*100)}%")
        return True, f"https://youtube.com/shorts/{resp['id']}"
    except ImportError:
        return False, "pip install google-api-python-client google-auth-oauthlib"
    except Exception as e:
        return False, str(e)


# ── DISPATCHER ────────────────────────────────────────────────

def post_to_all(video_path, caption=None):
    cfg     = load_config()
    caption = caption or cfg.get("caption", "")
    results = {}
    for name, fn in [("instagram", post_instagram), ("youtube", post_youtube)]:
        pcfg = cfg.get(name, {})
        if not pcfg.get("enabled", False):
            continue
        print(f"  → {name}...")
        try:
            ok, msg = fn(video_path, caption=caption, cfg=pcfg)
            results[name] = (ok, msg)
            print(f"    {'OK' if ok else 'FAIL'}: {msg}")
        except Exception as e:
            results[name] = (False, str(e))
    return results


# ── CONFIG GENERATOR ─────────────────────────────────────────

TEMPLATE = {
    "_readme": "Set enabled:true and fill in your credentials",
    "caption": "#viral #trending #reels #shorts",
    "instagram": {
        "enabled": False,
        "access_token": "YOUR_LONG_LIVED_INSTAGRAM_TOKEN",
        "user_id": "YOUR_INSTAGRAM_BUSINESS_USER_ID",
        "video_url_base": "https://YOUR_NGROK_URL/api/video"
    },
    "youtube": {
        "enabled": False,
        "client_secrets_file": "client_secrets.json",
        "token_file": "youtube_token.json",
        "privacy": "public",
        "tags": ["shorts", "viral", "trending", "reels"]
    }
}

if __name__ == "__main__":
    if not os.path.exists("config.json"):
        json.dump(TEMPLATE, open("config.json","w"), indent=2)
        print("✅ config.json created")
        print("   Instagram: developers.facebook.com → Instagram Graph API")
        print("   YouTube:   console.cloud.google.com → YouTube Data API v3")
    else:
        cfg = load_config()
        ig  = cfg.get("instagram", {})
        yt  = cfg.get("youtube", {})
        print(f"Instagram: {'✅ enabled' if ig.get('enabled') else '❌ disabled'}")
        print(f"YouTube:   {'✅ enabled' if yt.get('enabled') else '❌ disabled'}")