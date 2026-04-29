import cv2
import numpy as np
from PIL import Image, ImageStat, ImageDraw, ImageFont
from moviepy import VideoFileClip
import pytesseract
import json
import os
import math
import warnings
warnings.filterwarnings("ignore")

# Audio analysis
import librosa
import librosa.display

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

VIDEO_PATH = r"E:\reel_223000likes_AQPNXciBhndEMf3zQWBP.mp4"
OUTPUT_DIR = "analysis_output"
AUDIO_DIR  = os.path.join(OUTPUT_DIR, "audio")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR,  exist_ok=True)

try:
    ver = pytesseract.get_tesseract_version()
    print(f"✅ Tesseract {ver} ready")
except:
    print("❌ Tesseract not found — install from https://github.com/UB-Mannheim/tesseract/wiki")
    exit(1)

# ═══════════════════════════════════════════════════════════
# 1. MOVIEPY — container metadata
# ═══════════════════════════════════════════════════════════
def analyze_with_moviepy(path):
    clip  = VideoFileClip(path)
    audio = clip.audio
    data  = {
        "duration_sec": round(clip.duration, 2),
        "fps":          clip.fps,
        "resolution":   clip.size,
        "aspect_ratio": round(clip.size[0] / clip.size[1], 3),
        "total_frames": int(clip.duration * clip.fps),
        "has_audio":    audio is not None,
    }
    clip.close()
    return data

# ═══════════════════════════════════════════════════════════
# 2. LIBROSA — deep audio analysis
# ═══════════════════════════════════════════════════════════
def analyze_audio_deep(video_path):
    print("  🎵 Extracting audio track...")

    # Extract audio via moviepy
    clip       = VideoFileClip(video_path)
    audio_path = os.path.join(AUDIO_DIR, "temp_audio.wav")
    clip.audio.write_audiofile(audio_path, fps=22050, logger=None)
    clip.close()

    print("  🎵 Running librosa analysis...")
    y, sr = librosa.load(audio_path, sr=22050, mono=True)

    # ── Tempo & Beats ──────────────────────────────────────
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times         = librosa.frames_to_time(beat_frames, sr=sr).tolist()

    # ── Energy & RMS ──────────────────────────────────────
    rms         = librosa.feature.rms(y=y)[0]
    rms_mean    = float(np.mean(rms))
    rms_peak    = float(np.max(rms))
    rms_times   = librosa.frames_to_time(np.arange(len(rms)), sr=sr)

    # Energy over time (0.5s chunks)
    chunk_sec  = 0.5
    chunk_size = int(sr * chunk_sec)
    energy_over_time = []
    for i in range(0, len(y) - chunk_size, chunk_size):
        chunk  = y[i:i + chunk_size]
        energy = float(np.sqrt(np.mean(chunk**2)))
        energy_over_time.append({
            "sec":    round(i / sr, 2),
            "energy": round(energy, 4)
        })

    # ── Spectral Features ─────────────────────────────────
    spectral_centroid  = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spectral_rolloff   = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    zero_crossing      = librosa.feature.zero_crossing_rate(y)[0]

    # ── MFCCs (audio fingerprint) ──────────────────────────
    mfccs     = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = mfccs.mean(axis=1).tolist()

    # ── Chroma (harmonic content / key) ────────────────────
    chroma      = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1).tolist()
    note_names  = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    dominant_note = note_names[int(np.argmax(chroma_mean))]

    # ── Onset Detection (hard cuts / drops) ────────────────
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units="time")
    onset_times  = onset_frames.tolist()

    # ── Harmonic vs Percussive separation ──────────────────
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    harmonic_ratio   = float(np.mean(np.abs(y_harmonic)))
    percussive_ratio = float(np.mean(np.abs(y_percussive)))
    total            = harmonic_ratio + percussive_ratio + 1e-9
    harmony_pct      = round(harmonic_ratio / total * 100, 1)
    percussion_pct   = round(percussive_ratio / total * 100, 1)

    # ── Loudness dynamics ──────────────────────────────────
    # Find drops (energy dips) and builds (energy rises)
    energies = [e["energy"] for e in energy_over_time]
    mean_e   = np.mean(energies)
    drops     = [energy_over_time[i]["sec"] for i in range(1, len(energies)-1)
                 if energies[i] < mean_e * 0.5 and energies[i-1] > mean_e]
    builds    = [energy_over_time[i]["sec"] for i in range(1, len(energies)-1)
                 if energies[i] > mean_e * 1.5 and energies[i-1] < mean_e]

    # ── Audio Type Classification ───────────────────────────
    has_voice   = float(np.mean(spectral_centroid)) < 3000 and rms_mean > 0.02
    is_music    = len(beat_times) > 4 and harmony_pct > 40
    is_speech   = float(np.mean(zero_crossing)) > 0.1 and has_voice
    is_ambient  = rms_mean < 0.05 and len(beat_times) < 3

    if is_speech and is_music:   audio_type = "speech_over_music"
    elif is_speech:              audio_type = "voiceover"
    elif is_music:               audio_type = "music_only"
    elif is_ambient:             audio_type = "ambient"
    else:                        audio_type = "mixed"

    # ── Mood from audio ─────────────────────────────────────
    avg_tempo  = float(tempo)
    avg_energy = rms_mean
    if avg_tempo > 130 and avg_energy > 0.1:   audio_mood = "hype_energetic"
    elif avg_tempo > 110 and avg_energy > 0.06: audio_mood = "upbeat_positive"
    elif avg_tempo < 80 and avg_energy < 0.05:  audio_mood = "calm_melancholic"
    elif avg_tempo < 90:                         audio_mood = "chill_relaxed"
    elif harmony_pct > 60:                       audio_mood = "emotional_melodic"
    else:                                        audio_mood = "neutral_background"

    os.remove(audio_path)

    return {
        # Rhythm
        "tempo_bpm":           round(float(tempo), 1),
        "beat_times":          [round(t, 2) for t in beat_times[:20]],
        "beat_count":          len(beat_times),
        "onset_times":         [round(t, 2) for t in onset_times[:20]],
        "onset_count":         len(onset_times),
        # Energy
        "rms_mean":            round(rms_mean, 4),
        "rms_peak":            round(rms_peak, 4),
        "energy_over_time":    energy_over_time,
        "energy_drops":        [round(t, 2) for t in drops],
        "energy_builds":       [round(t, 2) for t in builds],
        # Spectral
        "spectral_centroid_mean":  round(float(np.mean(spectral_centroid)), 1),
        "spectral_rolloff_mean":   round(float(np.mean(spectral_rolloff)), 1),
        "spectral_bandwidth_mean": round(float(np.mean(spectral_bandwidth)), 1),
        "zero_crossing_mean":      round(float(np.mean(zero_crossing)), 4),
        # Harmonic
        "dominant_note":       dominant_note,
        "chroma_profile":      [round(x, 3) for x in chroma_mean],
        "mfcc_signature":      [round(x, 2) for x in mfcc_mean],
        "harmony_pct":         harmony_pct,
        "percussion_pct":      percussion_pct,
        # Classification
        "audio_type":          audio_type,
        "audio_mood":          audio_mood,
        "has_voice":           has_voice,
        "is_music":            is_music,
        "is_speech":           is_speech,
    }

# ═══════════════════════════════════════════════════════════
# 3. TESSERACT — OCR text extraction
# ═══════════════════════════════════════════════════════════
def extract_text_from_frame(frame):
    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_rgb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    inverted  = cv2.bitwise_not(thresh)
    try:
        t1 = pytesseract.image_to_string(img_rgb,                   config="--psm 11").strip()
        t2 = pytesseract.image_to_string(Image.fromarray(thresh),   config="--psm 11").strip()
        t3 = pytesseract.image_to_string(Image.fromarray(inverted), config="--psm 11").strip()
    except:
        return {"text": "", "word_count": 0, "words_with_position": [], "has_caption": False, "hook_text": ""}
    try:
        data  = pytesseract.image_to_data(img_rgb, output_type=pytesseract.Output.DICT, config="--psm 11")
        words = []
        for i, word in enumerate(data["text"]):
            if word.strip() and int(data["conf"][i]) > 40:
                cy = data["top"][i]  / frame.shape[0]
                cx = data["left"][i] / frame.shape[1]
                words.append({
                    "word":       word.strip(),
                    "confidence": int(data["conf"][i]),
                    "x":          data["left"][i],
                    "y":          data["top"][i],
                    "position_v": "top" if cy < 0.33 else "bottom" if cy > 0.66 else "middle",
                    "position_h": "left" if cx < 0.33 else "right"  if cx > 0.66 else "center",
                })
    except:
        words = []
    combined = " ".join(dict.fromkeys(f"{t1} {t2} {t3}".split()))
    return {
        "text":                combined,
        "word_count":          len(combined.split()),
        "words_with_position": words,
        "has_caption":         len(combined.strip()) > 3,
        "hook_text":           combined[:120],
    }

# ═══════════════════════════════════════════════════════════
# 4. OPENCV — per-frame visual analysis
# ═══════════════════════════════════════════════════════════
def analyze_with_opencv(path, sample_every_n_sec=1):
    cap      = cv2.VideoCapture(path)
    fps      = cap.get(cv2.CAP_PROP_FPS)
    total    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, int(fps * sample_every_n_sec))

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eye_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")

    frame_data, keyframes, prev_gray = [], [], None
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % interval == 0:
            sec  = round(frame_idx / fps, 2)
            print(f"  🖼️  Frame {frame_idx}/{total} @ {sec}s")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lab  = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)

            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
            eyes  = eye_cascade.detectMultiScale(gray, 1.1, 5)
            body  = body_cascade.detectMultiScale(gray, 1.1, 3)

            motion, camera_move = 0.0, "static"
            if prev_gray is not None:
                flow   = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                motion = float(np.mean(np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)))
                mx, my = float(np.mean(flow[..., 0])), float(np.mean(flow[..., 1]))
                if abs(mx) > abs(my):
                    camera_move = "pan_right" if mx > 1 else "pan_left" if mx < -1 else "static"
                else:
                    camera_move = "tilt_down" if my > 1 else "tilt_up" if my < -1 else "static"

            edges       = cv2.Canny(gray, 100, 200)
            sharpness   = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            hist        = cv2.calcHist([gray], [0], None, [256], [0, 256])
            b_mean      = float(frame[:, :, 0].mean())
            r_mean      = float(frame[:, :, 2].mean())
            color_temp  = "warm" if r_mean > b_mean * 1.1 else "cool" if b_mean > r_mean * 1.1 else "neutral"

            h, w = frame.shape[:2]
            thirds = {
                "top_left":   float(frame[:h//3,       :w//3].mean()),
                "top_center": float(frame[:h//3,       w//3:2*w//3].mean()),
                "top_right":  float(frame[:h//3,       2*w//3:].mean()),
                "mid_left":   float(frame[h//3:2*h//3, :w//3].mean()),
                "center":     float(frame[h//3:2*h//3, w//3:2*w//3].mean()),
                "mid_right":  float(frame[h//3:2*h//3, 2*w//3:].mean()),
                "bot_left":   float(frame[2*h//3:,     :w//3].mean()),
                "bot_center": float(frame[2*h//3:,     w//3:2*w//3].mean()),
                "bot_right":  float(frame[2*h//3:,     2*w//3:].mean()),
            }

            text_data = extract_text_from_frame(frame)
            frame_data.append({
                "frame": frame_idx, "sec": sec,
                "faces": len(faces), "eyes_detected": len(eyes), "body_detected": len(body),
                "face_positions": [{"cx": round((x+fw/2)/w,2),"cy": round((y+fh/2)/h,2)} for (x,y,fw,fh) in faces],
                "motion_score": round(motion, 3), "camera_move": camera_move,
                "sharpness": round(sharpness, 1),
                "edge_density": round(float(edges.mean()), 3),
                "hue": round(float(hsv[:,:,0].mean()),1),
                "saturation": round(float(hsv[:,:,1].mean()),1),
                "brightness": round(float(hsv[:,:,2].mean()),1),
                "perceptual_lightness": round(float(lab[:,:,0].mean()),1),
                "color_temp": color_temp,
                "hist_spread": round(float(np.std(hist)),1),
                "brightest_region": max(thirds, key=thirds.get),
                "ocr_text": text_data["text"],
                "ocr_word_count": text_data["word_count"],
                "ocr_words": text_data["words_with_position"],
                "has_text_overlay": text_data["has_caption"],
                "hook_text": text_data["hook_text"],
            })
            keyframes.append(frame.copy())
            prev_gray = gray
        frame_idx += 1

    cap.release()
    return frame_data, keyframes

# ═══════════════════════════════════════════════════════════
# 5. PIL — color palette + mood
# ═══════════════════════════════════════════════════════════
def analyze_with_pil(path, sample_every_n_sec=3):
    cap, fps  = cv2.VideoCapture(path), None
    fps       = cap.get(cv2.CAP_PROP_FPS)
    interval  = max(1, int(fps * sample_every_n_sec))
    frame_idx = 0
    pil_data  = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % interval == 0:
            img   = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            stat  = ImageStat.Stat(img)
            small = img.resize((50, 50))
            q     = small.quantize(colors=6).convert("RGB")
            pal   = sorted(q.getcolors(maxcolors=2500) or [], key=lambda x: -x[0])
            r, g, b = stat.mean[:3]
            if   r > 150 and g < 100 and b < 100: mood = "energetic_red"
            elif b > 150 and r < 120:              mood = "calm_cool"
            elif g > 140 and r < 130:              mood = "natural_green"
            elif r > 160 and g > 140 and b < 100: mood = "warm_golden"
            elif stat.mean[0] < 60:                mood = "dark_moody"
            elif stat.mean[0] > 200:               mood = "bright_clean"
            else:                                  mood = "neutral"
            pil_data.append({
                "sec": round(frame_idx/fps, 2),
                "mean_rgb": [round(x,1) for x in stat.mean[:3]],
                "stddev_rgb": [round(x,1) for x in stat.stddev[:3]],
                "dominant_colors": [list(p[1]) for p in pal[:5]],
                "perceived_brightness": round(0.299*r + 0.587*g + 0.114*b, 1),
                "color_mood": mood,
                "contrast_score": round(float(np.mean(stat.stddev[:3])), 1),
            })
        frame_idx += 1
    cap.release()
    return pil_data

# ═══════════════════════════════════════════════════════════
# 6. CONTENT CATEGORIZER
# ═══════════════════════════════════════════════════════════
def categorize_content(visual_summary, audio_data, ocr_text):

    face_ratio  = visual_summary["face_presence_ratio"]
    text_ratio  = visual_summary["text_overlay_ratio"]
    avg_motion  = visual_summary["avg_motion"]
    pacing      = visual_summary["pacing"]
    audio_mood  = audio_data["audio_mood"]
    audio_type  = audio_data["audio_type"]
    tempo       = audio_data["tempo_bpm"]
    has_voice   = audio_data["has_voice"]
    ocr_lower   = ocr_text.lower()

    # ── Hook Format Detection ────────────────────────────────
    hook_formats = {
        "relatable_situation": any(w in ocr_lower for w in ["feels like","when you","that moment","nobody","everyone"]),
        "list_format":         any(w in ocr_lower for w in ["#1","#2","step","tip","reason","things","ways"]),
        "question_hook":       any(w in ocr_lower for w in ["why","how","what","did you","have you","?"]),
        "transformation":      any(w in ocr_lower for w in ["before","after","change","from","to"]),
        "controversy":         any(w in ocr_lower for w in ["unpopular","nobody talks","secret","truth","actually"]),
        "educational":         any(w in ocr_lower for w in ["learn","did you know","fact","study","research"]),
        "storytelling":        any(w in ocr_lower for w in ["story","happened","one day","so i","then"]),
    }
    detected_hooks = [k for k, v in hook_formats.items() if v]

    # ── Niche Detection ──────────────────────────────────────
    niche_keywords = {
        "humor_relatable":  ["driving","feels","nobody","lol","literally","when you","mood"],
        "finance":          ["money","invest","rich","income","bank","finance","crypto","stock","wealth"],
        "fitness":          ["gym","workout","fitness","muscle","diet","protein","exercise","body"],
        "food":             ["recipe","cook","food","eat","restaurant","taste","meal","kitchen"],
        "tech_ai":          ["ai","tech","tool","app","software","code","digital","automation"],
        "motivation":       ["success","mindset","hustle","grind","goal","dream","achieve","believe"],
        "beauty_fashion":   ["makeup","outfit","style","fashion","beauty","skincare","look","wear"],
        "travel":           ["travel","trip","country","city","hotel","explore","adventure","beach"],
        "education":        ["learn","study","school","fact","science","history","book","knowledge"],
        "relationship":     ["love","relationship","partner","dating","couple","heart","feelings"],
    }
    niche_scores = {}
    for niche, keywords in niche_keywords.items():
        score = sum(1 for kw in keywords if kw in ocr_lower)
        if score > 0:
            niche_scores[niche] = score
    detected_niche = max(niche_scores, key=niche_scores.get) if niche_scores else "general"

    # ── Content Format ───────────────────────────────────────
    if face_ratio > 0.6 and has_voice:         content_format = "talking_head"
    elif face_ratio > 0.6 and not has_voice:   content_format = "silent_presenter"
    elif text_ratio > 0.7 and not face_ratio:  content_format = "text_only"
    elif text_ratio > 0.5 and face_ratio > 0.3:content_format = "text_plus_face"
    elif avg_motion > 8:                        content_format = "action_broll"
    elif avg_motion < 2:                        content_format = "static_broll"
    else:                                       content_format = "mixed_broll"

    # ── Viral Pattern Score (0-100) ──────────────────────────
    score = 0
    # Fast pacing = good
    if pacing == "fast":   score += 20
    elif pacing == "medium": score += 10
    # Matching audio mood
    if audio_mood in ["hype_energetic", "upbeat_positive"]: score += 15
    # Has clear hook format
    if detected_hooks:  score += 15
    # Text overlay (always boosts retention)
    if text_ratio > 0.5: score += 10
    # Tempo sweet spot (100-140 BPM performs best)
    if 100 <= tempo <= 140: score += 10
    # Relatable content
    if "relatable_situation" in detected_hooks: score += 10
    # Has voice (boosts watch time)
    if has_voice: score += 10
    # Strong niche signal
    if detected_niche != "general": score += 10

    # ── Platform Strategy ────────────────────────────────────
    duration = visual_summary.get("duration_sec", 15)
    if duration < 8:     duration_tag = "ultra_short"   # hook-only, max reach
    elif duration < 15:  duration_tag = "short"         # ideal for reels
    elif duration < 30:  duration_tag = "medium"        # good for storytelling
    elif duration < 60:  duration_tag = "long"          # educational
    else:                duration_tag = "extended"

    # ── Posting Recommendations ──────────────────────────────
    recommendations = []
    if pacing == "slow":
        recommendations.append("⚡ Increase pacing — cut every 2-3 seconds")
    if not detected_hooks:
        recommendations.append("🪝 Add a stronger hook in first 2 seconds")
    if text_ratio < 0.3:
        recommendations.append("📝 Add text overlays to boost silent viewership")
    if tempo < 100:
        recommendations.append("🎵 Use faster trending audio (100-130 BPM)")
    if face_ratio == 0 and content_format != "text_only":
        recommendations.append("👤 Consider adding face — increases trust + saves")
    if not recommendations:
        recommendations.append("✅ Strong format — replicate this structure")

    return {
        "detected_niche":    detected_niche,
        "content_format":    content_format,
        "hook_formats":      detected_hooks,
        "audio_mood":        audio_mood,
        "audio_type":        audio_type,
        "duration_category": duration_tag,
        "viral_score":       min(score, 100),
        "recommendations":   recommendations,
        "niche_scores":      niche_scores,
        "replication_template": {
            "format":        content_format,
            "hook_style":    detected_hooks[0] if detected_hooks else "unknown",
            "audio_mood":    audio_mood,
            "pacing":        pacing,
            "use_face":      face_ratio > 0.3,
            "use_text":      text_ratio > 0.3,
            "target_bpm":    round(tempo),
            "niche":         detected_niche,
        }
    }

# ═══════════════════════════════════════════════════════════
# 7. AGGREGATE visual summary
# ═══════════════════════════════════════════════════════════
def aggregate(frame_data, pil_data, moviepy_data):
    if not frame_data:
        return {}
    def avg(key):
        vals = [f[key] for f in frame_data if isinstance(f.get(key), (int,float))]
        return round(float(np.mean(vals)), 3) if vals else 0.0

    face_ratio = round(sum(1 for f in frame_data if f["faces"] > 0) / len(frame_data), 3)
    text_ratio = round(sum(1 for f in frame_data if f["has_text_overlay"]) / len(frame_data), 3)
    cuts       = sum(1 for f in frame_data if f["motion_score"] > 12)
    all_text   = " ".join(f["ocr_text"] for f in frame_data if f["ocr_text"]).strip()
    moods      = [p["color_mood"] for p in pil_data]
    dom_mood   = max(set(moods), key=moods.count) if moods else "unknown"
    cam_moves  = [f["camera_move"] for f in frame_data]
    dom_cam    = max(set(cam_moves), key=cam_moves.count)

    return {
        "duration_sec":         moviepy_data["duration_sec"],
        "avg_brightness":       avg("brightness"),
        "avg_saturation":       avg("saturation"),
        "avg_sharpness":        avg("sharpness"),
        "avg_motion":           avg("motion_score"),
        "avg_edge_density":     avg("edge_density"),
        "face_presence_ratio":  face_ratio,
        "text_overlay_ratio":   text_ratio,
        "estimated_cuts":       cuts,
        "pacing":               "fast" if cuts > 8 else "medium" if cuts > 3 else "slow",
        "dominant_camera_move": dom_cam,
        "dominant_color_mood":  dom_mood,
        "all_ocr_text":         all_text,
        "style": (
            "talking_head"      if face_ratio > 0.6        else
            "text_educational"  if text_ratio > 0.5        else
            "broll_cinematic"   if avg("motion_score") < 2 else
            "action_dynamic"    if avg("motion_score") > 8 else
            "mixed"
        ),
    }

# ═══════════════════════════════════════════════════════════
# 8. STORYBOARD generator
# ═══════════════════════════════════════════════════════════
def generate_storyboard(keyframes, frame_data, audio_data, category, output_path, cols=4):
    if not keyframes:
        return
    thumb_w, thumb_h = 360, 640
    label_h = 100
    rows    = math.ceil(len(keyframes) / cols)
    board   = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + label_h) + 80), (10, 10, 10))
    draw    = ImageDraw.Draw(board)

    try:
        font       = ImageFont.truetype("arial.ttf", 14)
        font_small = ImageFont.truetype("arial.ttf", 11)
        font_bold  = ImageFont.truetype("arialbd.ttf", 15)
        font_title = ImageFont.truetype("arialbd.ttf", 18)
    except:
        font = font_small = font_bold = font_title = ImageFont.load_default()

    # Header bar
    draw.rectangle([0, 0, cols * thumb_w, 75], fill=(25, 25, 40))
    niche  = category["detected_niche"].upper().replace("_", " ")
    fmt    = category["content_format"].replace("_", " ")
    score  = category["viral_score"]
    bpm    = audio_data["tempo_bpm"]
    amood  = audio_data["audio_mood"].replace("_", " ")
    draw.text((10, 8),  f"🎯 NICHE: {niche}  |  FORMAT: {fmt}  |  VIRAL SCORE: {score}/100", font=font_title, fill=(255, 220, 50))
    draw.text((10, 35), f"🎵 BPM: {bpm}  |  MOOD: {amood}  |  AUDIO: {audio_data['audio_type'].replace('_',' ')}", font=font_bold, fill=(160, 210, 255))
    draw.text((10, 55), f"🪝 HOOKS: {', '.join(category['hook_formats']) or 'none detected'}", font=font, fill=(180, 255, 180))

    for i, (frame, fd) in enumerate(zip(keyframes, frame_data)):
        col = i % cols
        row = i // cols
        x   = col * thumb_w
        y   = row * (thumb_h + label_h) + 80

        img      = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((thumb_w, thumb_h))
        img_draw = ImageDraw.Draw(img)

        for fp in fd.get("face_positions", []):
            fx, fy = int(fp["cx"] * thumb_w), int(fp["cy"] * thumb_h)
            img_draw.ellipse([fx-12, fy-12, fx+12, fy+12], outline=(0, 255, 80), width=3)

        for word in fd.get("ocr_words", [])[:5]:
            wx = int(word["x"] * thumb_w / frame.shape[1])
            wy = int(word["y"] * thumb_h / frame.shape[0])
            img_draw.rectangle([wx, wy, wx+60, wy+16], outline=(255, 220, 0), width=1)

        # Beat markers — red line if beat within 0.5s of this frame
        beat_near = any(abs(bt - fd["sec"]) < 0.5 for bt in audio_data["beat_times"])
        if beat_near:
            img_draw.rectangle([0, thumb_h-6, thumb_w, thumb_h], fill=(255, 60, 60))
            img_draw.text((4, thumb_h-18), "♪ BEAT", font=font_small, fill=(255,255,255))

        board.paste(img, (x, y))

        ly = y + thumb_h
        draw.rectangle([x, ly, x + thumb_w, ly + label_h], fill=(18, 18, 28))
        draw.text((x+5, ly+4),  f"⏱ {fd['sec']}s  📷 {fd['camera_move']}  🌡 {fd['color_temp']}", font=font,       fill=(160, 210, 255))
        draw.text((x+5, ly+22), f"💨 motion:{fd['motion_score']:.1f}  👤 {fd['faces']}  💡 {fd['brightness']:.0f}", font=font, fill=(200,200,200))
        draw.text((x+5, ly+40), f"📝 {'✅' if fd['has_text_overlay'] else '❌'}  {fd['hook_text'][:36]}{'…' if len(fd['hook_text'])>36 else ''}", font=font_small, fill=(255,220,80))
        draw.text((x+5, ly+58), f"🎨 {fd['brightest_region']}", font=font_small, fill=(140,140,160))
        draw.text((x+5, ly+76), f"♪ beat:{'yes' if beat_near else 'no'}", font=font_small, fill=(255,100,100) if beat_near else (100,100,100))

        draw.rectangle([x, y, x+42, y+22], fill=(220, 50, 50))
        draw.text((x+5, y+4), f"#{i+1}", font=font_bold, fill="white")

    board.save(output_path, quality=95)
    print(f"🎬 Storyboard saved: {output_path}")

# ═══════════════════════════════════════════════════════════
# RUN ALL
# ═══════════════════════════════════════════════════════════
print(f"\n🎬 Analyzing: {VIDEO_PATH}\n")

print("📦 Step 1/5 — MoviePy (container)...")
moviepy_data = analyze_with_moviepy(VIDEO_PATH)

print("\n🎵 Step 2/5 — Librosa (deep audio)...")
audio_data = analyze_audio_deep(VIDEO_PATH)

print("\n🖼️  Step 3/5 — OpenCV + Tesseract (frames)...")
opencv_frames, kframes = analyze_with_opencv(VIDEO_PATH)

print("\n🎨 Step 4/5 — PIL (color palette)...")
pil_frames = analyze_with_pil(VIDEO_PATH)

print("\n📊 Step 5/5 — Aggregating + Categorizing...")
summary  = aggregate(opencv_frames, pil_frames, moviepy_data)
category = categorize_content(summary, audio_data, summary["all_ocr_text"])

storyboard_path = os.path.join(OUTPUT_DIR, "storyboard.jpg")
generate_storyboard(kframes, opencv_frames, audio_data, category, storyboard_path)

report = {
    "file":               VIDEO_PATH,
    "container":          moviepy_data,
    "audio_analysis":     audio_data,
    "visual_summary":     summary,
    "content_category":   category,
    "frame_analysis":     opencv_frames,
    "color_analysis":     pil_frames,
}
report_path = os.path.join(OUTPUT_DIR, "video_report.json")
with open(report_path, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

# ── Print summary ────────────────────────────────────────
SEP = "=" * 60
print(f"\n{SEP}\n📦 CONTAINER\n{SEP}")
print(json.dumps(moviepy_data, indent=2))

print(f"\n{SEP}\n🎵 AUDIO ANALYSIS\n{SEP}")
audio_print = {k: v for k, v in audio_data.items() if k not in ["energy_over_time","chroma_profile","mfcc_signature"]}
print(json.dumps(audio_print, indent=2))

print(f"\n{SEP}\n🧠 VISUAL SUMMARY\n{SEP}")
print(json.dumps({k: v for k, v in summary.items() if k != "all_ocr_text"}, indent=2))

print(f"\n{SEP}\n🎯 CONTENT CATEGORY\n{SEP}")
print(json.dumps(category, indent=2))

print(f"\n{SEP}\n📝 OCR TEXT\n{SEP}")
print(summary.get("all_ocr_text") or "(none)")

print(f"\n✅ Storyboard  → {storyboard_path}")
print(f"✅ Full report → {report_path}")