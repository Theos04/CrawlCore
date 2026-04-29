"""
PATCH NOTES — What changed from 4.py:
  ① EasyOCR: ["en"] → ["en", "hi"] so Hindi/Devanagari text is detected
  ② Whisper: forced language="hi" so transcript comes out in Devanagari, not Urdu script
  ③ New: detect_scene_background() — indoor/outdoor, wall color, room type, sky, greenery
  ④ New: count_bodies_hog() — HOG+SVM person detector counts full bodies (not just faces)
  ⑤ analyze_face_pose() — now returns per-person data for all faces (was only face[0])
  ⑥ frame_data now stores: scene_type, background_desc, body_count, people_positions
  ⑦ aggregate_visual() — adds avg_body_count, dominant_scene, indoor_outdoor
  ⑧ HTML report — new "Scene & People" card
"""

import cv2
import numpy as np
from PIL import Image, ImageStat, ImageDraw, ImageFont
from moviepy import VideoFileClip
import librosa
import json
import os
import math
import csv
import argparse
import warnings
import hashlib
import pickle
import time
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
warnings.filterwarnings("ignore")

# ── Optional imports ────────────────────────────────────────
try:
    import whisper
    WHISPER_AVAILABLE = True
    print("✅ Whisper (speech-to-text) ready")
except ImportError:
    WHISPER_AVAILABLE = False
    print("⚠️  Whisper not installed — pip install openai-whisper")

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
    mp_face_mesh      = mp.solutions.face_mesh
    mp_face_detection = mp.solutions.face_detection
    mp_pose           = mp.solutions.pose
    print("✅ MediaPipe (face/pose) ready")
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("⚠️  MediaPipe not installed — pip install mediapipe")

try:
    import easyocr
    EASYOCR_AVAILABLE = True
    EASYOCR_READER    = None
    print("✅ EasyOCR ready")
except ImportError:
    EASYOCR_AVAILABLE = False
    print("⚠️  EasyOCR not installed — pip install easyocr")

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
    print("✅ Anthropic SDK ready")
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("⚠️  Anthropic SDK not installed — pip install anthropic")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


# ═══════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════
class Config:
    OUTPUT_DIR        = "analysis_output"
    AUDIO_DIR         = os.path.join(OUTPUT_DIR, "audio")
    REPORTS_DIR       = os.path.join(OUTPUT_DIR, "reports")
    CACHE_DIR         = os.path.join(OUTPUT_DIR, ".cache")
    CSV_LOG           = os.path.join(OUTPUT_DIR, "all_reels_log.csv")
    SAMPLE_EVERY_SEC  = 1
    PIL_SAMPLE_SEC    = 3
    STORYBOARD_COLS   = 4
    WHISPER_MODEL     = "base"
    # ① PATCHED: added "hi" for Hindi Devanagari detection
    EASYOCR_LANGS     = ["en", "hi"]
    ANTHROPIC_MODEL   = "claude-opus-4-6"
    ENABLE_AI_BRIEF   = True
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    # ② Whisper language override — set None for auto-detect
    WHISPER_LANG      = "hi"   # Force Hindi so output is Devanagari, not Urdu/Arabic script

for d in [Config.OUTPUT_DIR, Config.AUDIO_DIR, Config.REPORTS_DIR, Config.CACHE_DIR]:
    os.makedirs(d, exist_ok=True)


# ═══════════════════════════════════════════════════════════
# CACHE
# ═══════════════════════════════════════════════════════════
def get_video_hash(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read(65536))
    return h.hexdigest()[:12]

def load_cache(path: str, key: str):
    cache_file = os.path.join(Config.CACHE_DIR, f"{get_video_hash(path)}_{key}.pkl")
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    return None

def save_cache(path: str, key: str, data):
    cache_file = os.path.join(Config.CACHE_DIR, f"{get_video_hash(path)}_{key}.pkl")
    with open(cache_file, "wb") as f:
        pickle.dump(data, f)


# ═══════════════════════════════════════════════════════════
# MODULE 1 — Container metadata
# ═══════════════════════════════════════════════════════════
def analyze_container(path: str) -> dict:
    clip = VideoFileClip(path)
    data = {
        "duration_sec": round(clip.duration, 2),
        "fps":          clip.fps,
        "resolution":   list(clip.size),
        "aspect_ratio": round(clip.size[0] / clip.size[1], 3),
        "total_frames": int(clip.duration * clip.fps),
        "has_audio":    clip.audio is not None,
        "is_vertical":  clip.size[1] > clip.size[0],
        "format_hint":  "reel/story" if clip.size[1] > clip.size[0] else "landscape",
    }
    clip.close()
    return data


# ═══════════════════════════════════════════════════════════
# MODULE 2 — Whisper STT
# ② PATCHED: language forced to Config.WHISPER_LANG (default "hi")
#    This ensures Hindi audio outputs Devanagari script, not Urdu/Arabic
# ═══════════════════════════════════════════════════════════
def transcribe_speech(video_path: str) -> dict:
    if not WHISPER_AVAILABLE:
        return {"transcript": "", "language": "unknown", "segments": [], "word_timestamps": []}

    cached = load_cache(video_path, "whisper")
    if cached:
        print("  ♻️  Whisper — using cache")
        return cached

    print("  🎙️  Whisper — extracting audio...")
    clip       = VideoFileClip(video_path)
    audio_path = os.path.join(Config.AUDIO_DIR, "whisper_temp.wav")
    clip.audio.write_audiofile(audio_path, fps=16000, logger=None)
    clip.close()

    print(f"  🎙️  Whisper ({Config.WHISPER_MODEL}) — transcribing (lang={Config.WHISPER_LANG or 'auto'})...")
    model = whisper.load_model(Config.WHISPER_MODEL)

    # ② KEY FIX: pass language= to force Devanagari Hindi output
    transcribe_kwargs = dict(word_timestamps=True, verbose=False)
    if Config.WHISPER_LANG:
        transcribe_kwargs["language"] = Config.WHISPER_LANG

    result = model.transcribe(audio_path, **transcribe_kwargs)

    segments = []
    for seg in result.get("segments", []):
        segments.append({
            "start": round(seg["start"], 2),
            "end":   round(seg["end"], 2),
            "text":  seg["text"].strip(),
        })

    word_ts = []
    for seg in result.get("segments", []):
        for w in seg.get("words", []):
            word_ts.append({
                "word":  w["word"].strip(),
                "start": round(w["start"], 2),
                "end":   round(w["end"], 2),
            })

    os.remove(audio_path)

    data = {
        "transcript":        result["text"].strip(),
        "language":          result.get("language", Config.WHISPER_LANG or "unknown"),
        "segments":          segments,
        "word_timestamps":   word_ts,
        "word_count":        len(result["text"].split()),
        "speaking_rate_wpm": round(
            len(result["text"].split()) / max(result["segments"][-1]["end"] / 60, 0.01)
            if result.get("segments") else 0, 1
        ),
    }
    save_cache(video_path, "whisper", data)
    return data


# ═══════════════════════════════════════════════════════════
# MODULE 3 — Librosa audio analysis (unchanged)
# ═══════════════════════════════════════════════════════════
def analyze_audio(video_path: str) -> dict:
    cached = load_cache(video_path, "audio")
    if cached:
        print("  ♻️  Audio — using cache")
        return cached

    print("  🎵 Extracting audio track...")
    clip       = VideoFileClip(video_path)
    audio_path = os.path.join(Config.AUDIO_DIR, "temp_audio.wav")
    clip.audio.write_audiofile(audio_path, fps=22050, logger=None)
    clip.close()

    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times         = librosa.frames_to_time(beat_frames, sr=sr).tolist()
    rms      = librosa.feature.rms(y=y)[0]
    rms_mean = float(np.mean(rms))
    rms_peak = float(np.max(rms))

    chunk_sec  = 0.5
    chunk_size = int(sr * chunk_sec)
    energy_over_time = []
    for i in range(0, len(y) - chunk_size, chunk_size):
        chunk = y[i:i + chunk_size]
        energy_over_time.append({
            "sec":    round(i / sr, 2),
            "energy": round(float(np.sqrt(np.mean(chunk**2))), 4)
        })

    spectral_centroid  = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spectral_rolloff   = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    zero_crossing      = librosa.feature.zero_crossing_rate(y)[0]
    mfccs              = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean          = mfccs.mean(axis=1).tolist()
    chroma             = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean        = chroma.mean(axis=1).tolist()
    note_names         = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    dominant_note      = note_names[int(np.argmax(chroma_mean))]

    chroma_arr     = np.array(chroma_mean)
    major_template = np.array([1,0,1,0,1,1,0,1,0,1,0,1], dtype=float)
    minor_template = np.array([1,0,1,1,0,1,0,1,1,0,1,0], dtype=float)
    major_scores   = [np.dot(np.roll(chroma_arr, -i), major_template) for i in range(12)]
    minor_scores   = [np.dot(np.roll(chroma_arr, -i), minor_template) for i in range(12)]
    best_major     = int(np.argmax(major_scores))
    best_minor     = int(np.argmax(minor_scores))
    if max(major_scores) >= max(minor_scores):
        detected_key = f"{note_names[best_major]} major"
    else:
        detected_key = f"{note_names[best_minor]} minor"

    onset_times    = librosa.onset.onset_detect(y=y, sr=sr, units="time").tolist()
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    harmonic_ratio   = float(np.mean(np.abs(y_harmonic)))
    percussive_ratio = float(np.mean(np.abs(y_percussive)))
    total            = harmonic_ratio + percussive_ratio + 1e-9
    harmony_pct      = round(harmonic_ratio / total * 100, 1)
    percussion_pct   = round(percussive_ratio / total * 100, 1)

    energies = [e["energy"] for e in energy_over_time]
    mean_e   = np.mean(energies) if energies else 0
    drops    = [energy_over_time[i]["sec"] for i in range(1, len(energies)-1)
                if energies[i] < mean_e * 0.5 and energies[i-1] > mean_e]
    builds   = [energy_over_time[i]["sec"] for i in range(1, len(energies)-1)
                if energies[i] > mean_e * 1.5 and energies[i-1] < mean_e]

    silence_threshold = 0.005
    silent_chunks  = [e for e in energy_over_time if e["energy"] < silence_threshold]
    silence_ratio  = round(len(silent_chunks) / max(len(energy_over_time), 1), 3)

    has_voice  = float(np.mean(spectral_centroid)) < 3500 and rms_mean > 0.015
    is_music   = len(beat_times) > 4 and harmony_pct > 35
    is_speech  = float(np.mean(zero_crossing)) > 0.08 and has_voice
    is_ambient = rms_mean < 0.05 and len(beat_times) < 3

    if is_speech and is_music:   audio_type = "speech_over_music"
    elif is_speech:              audio_type = "voiceover"
    elif is_music:               audio_type = "music_only"
    elif is_ambient:             audio_type = "ambient"
    else:                        audio_type = "mixed"

    avg_tempo = float(tempo)
    if avg_tempo > 130 and rms_mean > 0.1:    audio_mood = "hype_energetic"
    elif avg_tempo > 110 and rms_mean > 0.06: audio_mood = "upbeat_positive"
    elif avg_tempo < 80 and rms_mean < 0.05:  audio_mood = "calm_melancholic"
    elif avg_tempo < 90:                       audio_mood = "chill_relaxed"
    elif harmony_pct > 60:                     audio_mood = "emotional_melodic"
    else:                                      audio_mood = "neutral_background"

    os.remove(audio_path)

    data = {
        "tempo_bpm":           round(float(tempo), 1),
        "detected_key":        detected_key,
        "dominant_note":       dominant_note,
        "beat_times":          [round(t, 2) for t in beat_times[:20]],
        "beat_count":          len(beat_times),
        "onset_times":         [round(t, 2) for t in onset_times[:20]],
        "onset_count":         len(onset_times),
        "rms_mean":            round(rms_mean, 4),
        "rms_peak":            round(rms_peak, 4),
        "silence_ratio":       silence_ratio,
        "energy_over_time":    energy_over_time,
        "energy_drops":        [round(t, 2) for t in drops],
        "energy_builds":       [round(t, 2) for t in builds],
        "spectral_centroid_mean":  round(float(np.mean(spectral_centroid)), 1),
        "spectral_rolloff_mean":   round(float(np.mean(spectral_rolloff)), 1),
        "spectral_bandwidth_mean": round(float(np.mean(spectral_bandwidth)), 1),
        "zero_crossing_mean":      round(float(np.mean(zero_crossing)), 4),
        "chroma_profile":          [round(x, 3) for x in chroma_mean],
        "mfcc_signature":          [round(x, 2) for x in mfcc_mean],
        "harmony_pct":             harmony_pct,
        "percussion_pct":          percussion_pct,
        "audio_type":              audio_type,
        "audio_mood":              audio_mood,
        "has_voice":               has_voice,
        "is_music":                is_music,
        "is_speech":               is_speech,
    }
    save_cache(video_path, "audio", data)
    return data


# ═══════════════════════════════════════════════════════════
# MODULE 4A — EasyOCR (now includes Hindi)
# ① PATCHED: EASYOCR_LANGS now ["en","hi"] — detects Devanagari text overlays
# ═══════════════════════════════════════════════════════════
def get_easyocr_reader():
    global EASYOCR_READER
    if EASYOCR_READER is None:
        print("  📝 Loading EasyOCR model (first time only)...")
        # ① KEY FIX: include "hi" in reader languages
        EASYOCR_READER = easyocr.Reader(Config.EASYOCR_LANGS, gpu=False)
    return EASYOCR_READER

def extract_text_easyocr(frame) -> dict:
    if not EASYOCR_AVAILABLE:
        return {"text": "", "word_count": 0, "words": [], "has_text": False,
                "readability_score": 0, "text_zones": [], "hook_text": ""}

    reader  = get_easyocr_reader()
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = reader.readtext(img_rgb, detail=1, paragraph=False)

    h, w = frame.shape[:2]
    words, zones = [], []
    full_text = ""

    for (bbox, text, conf) in results:
        if conf < 0.3 or len(text.strip()) < 1:
            continue
        x1 = int(bbox[0][0]); y1 = int(bbox[0][1])
        x2 = int(bbox[2][0]); y2 = int(bbox[2][1])
        cx, cy      = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
        font_size_est = abs(y2 - y1)

        # Detect script type
        script = "latin"
        for ch in text:
            cp = ord(ch)
            if 0x0900 <= cp <= 0x097F:
                script = "devanagari"
                break
            elif 0x0600 <= cp <= 0x06FF:
                script = "arabic_urdu"
                break

        words.append({
            "text":       text.strip(),
            "confidence": round(conf, 2),
            "bbox":       [x1, y1, x2, y2],
            "pos_v":      "top" if cy < 0.33 else "bottom" if cy > 0.66 else "middle",
            "pos_h":      "left" if cx < 0.33 else "right" if cx > 0.66 else "center",
            "font_size":  font_size_est,
            "is_large":   font_size_est > 40,
            "script":     script,
        })
        zones.append({"x1":x1,"y1":y1,"x2":x2,"y2":y2,"text":text.strip(),"script":script})
        full_text += " " + text.strip()

    full_text   = full_text.strip()
    large_words = [w for w in words if w["is_large"]]
    readability = round(len(large_words) / max(len(words), 1) * 100, 1)

    return {
        "text":              full_text,
        "word_count":        len(full_text.split()),
        "words":             words,
        "has_text":          len(full_text) > 1,
        "readability_score": readability,
        "text_zones":        zones,
        "hook_text":         full_text[:150],
        "scripts_detected":  list({w["script"] for w in words}),
    }


# ═══════════════════════════════════════════════════════════
# MODULE 4B — Background / Scene Detection (NEW ③)
# Classifies: indoor/outdoor, setting type, wall color, lighting
# Uses HSV color analysis + region heuristics — no extra models needed
# ═══════════════════════════════════════════════════════════
def detect_scene_background(frame, face_boxes: list) -> dict:
    """
    Analyzes the background region (excluding face areas) to determine:
    - indoor vs outdoor
    - setting type (bedroom, studio, street, nature, etc.)
    - background dominant color / texture
    - lighting quality
    """
    h, w = frame.shape[:2]

    # Build a mask that excludes detected face regions
    mask = np.ones((h, w), dtype=np.uint8) * 255
    for fb in face_boxes:
        cx, cy = int(fb["cx"] * w), int(fb["cy"] * h)
        sz     = int(math.sqrt(max(fb.get("size_ratio", 0.05), 0.01)) * w * 1.5)
        x1 = max(0, cx - sz); y1 = max(0, cy - sz)
        x2 = min(w, cx + sz); y2 = min(h, cy + sz)
        mask[y1:y2, x1:x2] = 0

    # Background pixels only
    bg_pixels = frame[mask > 0]
    if len(bg_pixels) < 100:
        bg_pixels = frame.reshape(-1, 3)

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # --- Sky / outdoor detection ---
    # Sample top 25% of frame
    top_zone    = hsv_frame[:h//4, :, :]
    sky_blue    = np.sum((top_zone[:,:,0] > 95) & (top_zone[:,:,0] < 135) &
                         (top_zone[:,:,1] > 40) & (top_zone[:,:,2] > 80))
    sky_ratio   = sky_blue / max(top_zone.shape[0]*top_zone.shape[1], 1)

    # Greenery detection (full frame)
    green_px  = np.sum((hsv_frame[:,:,0] > 35) & (hsv_frame[:,:,0] < 85) &
                       (hsv_frame[:,:,1] > 40) & (hsv_frame[:,:,2] > 40))
    green_ratio = green_px / (h * w)

    # --- Wall / flat background detection ---
    # Low texture variation in background = flat wall or studio
    bg_gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bg_masked  = cv2.bitwise_and(bg_gray, bg_gray, mask=mask)
    nonzero    = bg_masked[mask > 0]
    texture_std = float(np.std(nonzero)) if len(nonzero) > 0 else 0

    # --- Dominant background color ---
    if len(bg_pixels) > 0:
        mean_bgr  = bg_pixels.mean(axis=0)
        r, g, b   = float(mean_bgr[2]), float(mean_bgr[1]), float(mean_bgr[0])
    else:
        r, g, b = 128.0, 128.0, 128.0

    # Named color
    if r > 200 and g > 200 and b > 200:
        wall_color = "white"
    elif r < 60 and g < 60 and b < 60:
        wall_color = "black"
    elif r > g + 30 and r > b + 30:
        wall_color = "red_warm"
    elif b > r + 30 and b > g + 20:
        wall_color = "blue_cool"
    elif g > r + 20 and g > b + 20:
        wall_color = "green"
    elif r > 160 and g > 130 and b < 100:
        wall_color = "golden_warm"
    elif r > 180 and g > 180:
        wall_color = "bright_yellow"
    else:
        val = int((r + g + b) / 3)
        wall_color = f"grey_{val//50*50}"

    # --- Classify setting ---
    is_outdoor      = sky_ratio > 0.08 or green_ratio > 0.15
    is_nature       = green_ratio > 0.25
    is_studio_flat  = texture_std < 18 and not is_outdoor
    is_busy_bg      = texture_std > 45

    if is_nature and is_outdoor:
        setting = "outdoor_nature"
    elif is_outdoor and sky_ratio > 0.08:
        setting = "outdoor_street_or_open"
    elif is_outdoor:
        setting = "outdoor_other"
    elif is_studio_flat:
        setting = "studio_or_plain_wall"
    elif is_busy_bg:
        setting = "indoor_busy_background"
    else:
        setting = "indoor_room"

    # Detailed room type heuristics (indoor only)
    room_type = None
    if not is_outdoor:
        brightness = float(np.mean(bg_masked[mask > 0])) if len(nonzero) > 0 else 128
        if brightness > 200 and texture_std < 20:
            room_type = "studio_white"
        elif wall_color in ["red_warm", "golden_warm", "bright_yellow"]:
            room_type = "warm_decorated_room"
        elif wall_color == "blue_cool":
            room_type = "cool_toned_room"
        elif texture_std > 40:
            room_type = "cluttered_or_bookshelf"
        else:
            room_type = "generic_indoor"

    # Lighting quality
    bgr_std = float(np.std(bg_pixels)) if len(bg_pixels) > 0 else 0
    bg_bright = float(np.mean(bg_pixels)) if len(bg_pixels) > 0 else 128
    if bg_bright > 170:
        lighting = "bright_well_lit"
    elif bg_bright > 100:
        lighting = "normal"
    elif bg_bright > 50:
        lighting = "dim"
    else:
        lighting = "dark_low_light"

    return {
        "indoor_outdoor":   "outdoor" if is_outdoor else "indoor",
        "setting_type":     setting,
        "room_type":        room_type,
        "wall_color":       wall_color,
        "bg_texture_std":   round(texture_std, 1),
        "is_flat_bg":       is_studio_flat,
        "is_busy_bg":       is_busy_bg,
        "sky_ratio":        round(sky_ratio, 3),
        "green_ratio":      round(green_ratio, 3),
        "lighting":         lighting,
        "bg_mean_rgb":      [round(r,1), round(g,1), round(b,1)],
    }


# ═══════════════════════════════════════════════════════════
# MODULE 4C — Body / People counter (NEW ④)
# Uses OpenCV HOG person detector (no extra install needed)
# Also uses MediaPipe face count as secondary source
# ═══════════════════════════════════════════════════════════

# Initialize HOG person detector once
_HOG = cv2.HOGDescriptor()
_HOG.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

def count_bodies_hog(frame) -> dict:
    """
    Uses HOG+SVM (built into OpenCV) to detect full human bodies.
    Returns count and rough positions.
    Works best when full or half body is visible.
    """
    h, w    = frame.shape[:2]
    # Resize for speed (HOG works on smaller images)
    scale   = 480 / h if h > 480 else 1.0
    small   = cv2.resize(frame, (int(w * scale), int(h * scale)))

    rects, weights = _HOG.detectMultiScale(
        small,
        winStride=(8, 8),
        padding=(4, 4),
        scale=1.05,
        hitThreshold=0,
    )

    people = []
    for (bx, by, bw, bh) in rects:
        cx = round((bx + bw/2) / small.shape[1], 3)
        cy = round((by + bh/2) / small.shape[0], 3)
        people.append({
            "cx": cx, "cy": cy,
            "pos_v": "top" if cy < 0.33 else "bottom" if cy > 0.66 else "center",
            "pos_h": "left" if cx < 0.33 else "right" if cx > 0.66 else "center",
        })

    return {
        "body_count_hog": len(people),
        "body_positions":  people,
    }


# ═══════════════════════════════════════════════════════════
# MODULE 4D — MediaPipe face + pose (updated for multi-face)
# ═══════════════════════════════════════════════════════════
def analyze_face_pose(frame) -> dict:
    if not MEDIAPIPE_AVAILABLE:
        return {
            "face_count": 0, "faces": [], "body_detected": False,
            "gaze_direction": "unknown", "upper_body_visible": False,
        }

    h, w    = frame.shape[:2]
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result  = {"face_count": 0, "faces": [], "body_detected": False,
               "gaze_direction": "unknown", "upper_body_visible": False}

    with mp_face_detection.FaceDetection(min_detection_confidence=0.4) as fd:
        detection = fd.process(img_rgb)
        if detection.detections:
            result["face_count"] = len(detection.detections)
            for det in detection.detections:
                bbox   = det.location_data.relative_bounding_box
                cx     = round(bbox.xmin + bbox.width / 2, 3)
                cy     = round(bbox.ymin + bbox.height / 2, 3)
                size   = round(bbox.width * bbox.height, 4)
                score  = round(det.score[0], 2)
                result["faces"].append({
                    "cx": cx, "cy": cy,
                    "size_ratio": size,
                    "confidence": score,
                    "is_close_up": size > 0.15,
                    "position": "top" if cy < 0.33 else "bottom" if cy > 0.66 else "center",
                })

    # Gaze only for primary (largest) face
    with mp_face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1,
        refine_landmarks=True, min_detection_confidence=0.4
    ) as fm:
        mesh_result = fm.process(img_rgb)
        if mesh_result.multi_face_landmarks:
            lm           = mesh_result.multi_face_landmarks[0].landmark
            nose_x       = lm[1].x
            left_eye_x   = lm[33].x
            right_eye_x  = lm[263].x
            eye_center_x = (left_eye_x + right_eye_x) / 2
            diff = nose_x - eye_center_x
            if diff > 0.03:    result["gaze_direction"] = "looking_right"
            elif diff < -0.03: result["gaze_direction"] = "looking_left"
            else:              result["gaze_direction"] = "looking_at_camera"
            result["eye_contact"] = result["gaze_direction"] == "looking_at_camera"

    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.4) as pose:
        pose_result = pose.process(img_rgb)
        if pose_result.pose_landmarks:
            result["body_detected"]      = True
            result["upper_body_visible"] = True
            lms = pose_result.pose_landmarks.landmark
            left_wrist  = lms[mp_pose.PoseLandmark.LEFT_WRIST]
            right_wrist = lms[mp_pose.PoseLandmark.RIGHT_WRIST]
            result["hands_raised"] = (left_wrist.y < 0.5 or right_wrist.y < 0.5)

    return result


# ═══════════════════════════════════════════════════════════
# MODULE 4E — Per-frame OpenCV visual analysis (updated)
# Now includes background scene + body count per frame
# ═══════════════════════════════════════════════════════════
def analyze_frames(path: str) -> tuple:
    cached = load_cache(path, "frames_v2")  # new cache key so old cache is ignored
    if cached:
        print("  ♻️  Frames — using cache")
        # Keyframes can't be cached, regenerate them
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        interval = max(1, int(fps * Config.SAMPLE_EVERY_SEC))
        keyframes = []
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            if frame_idx % interval == 0:
                keyframes.append(frame.copy())
            frame_idx += 1
        cap.release()
        return cached, keyframes

    cap      = cv2.VideoCapture(path)
    fps      = cap.get(cv2.CAP_PROP_FPS)
    total    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, int(fps * Config.SAMPLE_EVERY_SEC))

    frame_data, keyframes = [], []
    prev_gray = None
    frame_idx = 0

    with tqdm(total=total, desc="  🖼️  Frames", unit="fr") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % interval == 0:
                sec  = round(frame_idx / fps, 2)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                lab  = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)

                # Motion
                motion, camera_move = 0.0, "static"
                if prev_gray is not None:
                    flow   = cv2.calcOpticalFlowFarneback(
                        prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    motion = float(np.mean(np.sqrt(flow[...,0]**2 + flow[...,1]**2)))
                    mx, my = float(np.mean(flow[...,0])), float(np.mean(flow[...,1]))
                    if abs(mx) > abs(my):
                        camera_move = "pan_right" if mx > 1 else "pan_left" if mx < -1 else "static"
                    else:
                        camera_move = "tilt_down" if my > 1 else "tilt_up" if my < -1 else "static"

                sharpness   = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                edges       = cv2.Canny(gray, 100, 200)
                edge_density = float(edges.mean())

                h_frame, w_frame = frame.shape[:2]
                b_mean = float(frame[:,:,0].mean())
                r_mean = float(frame[:,:,2].mean())
                color_temp = "warm" if r_mean > b_mean*1.1 else "cool" if b_mean > r_mean*1.1 else "neutral"

                h3, w3 = h_frame//3, w_frame//3
                thirds = {
                    "top_left":   float(frame[:h3,     :w3].mean()),
                    "top_center": float(frame[:h3,     w3:2*w3].mean()),
                    "top_right":  float(frame[:h3,     2*w3:].mean()),
                    "mid_left":   float(frame[h3:2*h3, :w3].mean()),
                    "center":     float(frame[h3:2*h3, w3:2*w3].mean()),
                    "mid_right":  float(frame[h3:2*h3, 2*w3:].mean()),
                    "bot_left":   float(frame[2*h3:,   :w3].mean()),
                    "bot_center": float(frame[2*h3:,   w3:2*w3].mean()),
                    "bot_right":  float(frame[2*h3:,   2*w3:].mean()),
                }
                brightest_region = max(thirds, key=thirds.get)
                scene_change = motion > 20

                # OCR — EasyOCR with Hindi
                if EASYOCR_AVAILABLE:
                    text_data = extract_text_easyocr(frame)
                else:
                    text_data = {"text":"","word_count":0,"words":[],"has_text":False,
                                 "readability_score":0,"text_zones":[],"hook_text":"",
                                 "scripts_detected":[]}

                # Face + pose
                face_data = analyze_face_pose(frame)

                # ③ Background / scene detection (NEW)
                scene_data = detect_scene_background(frame, face_data.get("faces", []))

                # ④ Body count via HOG (NEW)
                body_data = count_bodies_hog(frame)

                # Use max of face count and HOG body count as people_in_frame
                people_count = max(face_data["face_count"], body_data["body_count_hog"])

                frame_data.append({
                    "frame":           frame_idx,
                    "sec":             sec,
                    # Face
                    "face_count":      face_data["face_count"],
                    "face_positions":  face_data["faces"],
                    "gaze_direction":  face_data.get("gaze_direction", "unknown"),
                    "eye_contact":     face_data.get("eye_contact", False),
                    "body_detected":   face_data["body_detected"],
                    "hands_raised":    face_data.get("hands_raised", False),
                    # ④ People / bodies (NEW)
                    "people_count":    people_count,
                    "body_count_hog":  body_data["body_count_hog"],
                    "body_positions":  body_data["body_positions"],
                    # ③ Scene / background (NEW)
                    "indoor_outdoor":  scene_data["indoor_outdoor"],
                    "setting_type":    scene_data["setting_type"],
                    "room_type":       scene_data["room_type"],
                    "wall_color":      scene_data["wall_color"],
                    "is_flat_bg":      scene_data["is_flat_bg"],
                    "bg_texture_std":  scene_data["bg_texture_std"],
                    "sky_ratio":       scene_data["sky_ratio"],
                    "green_ratio":     scene_data["green_ratio"],
                    "lighting":        scene_data["lighting"],
                    "bg_mean_rgb":     scene_data["bg_mean_rgb"],
                    # Motion
                    "motion_score":    round(motion, 3),
                    "camera_move":     camera_move,
                    "scene_change":    scene_change,
                    # Visual quality
                    "sharpness":       round(sharpness, 1),
                    "edge_density":    round(edge_density, 3),
                    "hue":             round(float(hsv[:,:,0].mean()), 1),
                    "saturation":      round(float(hsv[:,:,1].mean()), 1),
                    "brightness":      round(float(hsv[:,:,2].mean()), 1),
                    "perceptual_lightness": round(float(lab[:,:,0].mean()), 1),
                    "color_temp":      color_temp,
                    "brightest_region": brightest_region,
                    # Text
                    "ocr_text":        text_data["text"],
                    "ocr_word_count":  text_data["word_count"],
                    "ocr_words":       text_data["words"],
                    "has_text_overlay": text_data["has_text"],
                    "text_readability": text_data["readability_score"],
                    "hook_text":       text_data["hook_text"],
                    "scripts_detected": text_data.get("scripts_detected", []),
                })
                keyframes.append(frame.copy())
                prev_gray = gray
            frame_idx += 1
            pbar.update(1)

    cap.release()
    save_cache(path, "frames_v2", frame_data)
    return frame_data, keyframes


# ═══════════════════════════════════════════════════════════
# MODULE 5 — PIL color palette (unchanged)
# ═══════════════════════════════════════════════════════════
def analyze_color_palette(path: str) -> list:
    cached = load_cache(path, "pil")
    if cached:
        return cached

    cap      = cv2.VideoCapture(path)
    fps      = cap.get(cv2.CAP_PROP_FPS)
    interval = max(1, int(fps * Config.PIL_SAMPLE_SEC))
    frame_idx = 0
    pil_data  = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        if frame_idx % interval == 0:
            img  = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            stat = ImageStat.Stat(img)
            small = img.resize((50, 50))
            q    = small.quantize(colors=8).convert("RGB")
            pal  = sorted(q.getcolors(maxcolors=2500) or [], key=lambda x: -x[0])
            r, g, b = stat.mean[:3]

            if   r > 150 and g < 100 and b < 100: mood = "energetic_red"
            elif b > 150 and r < 120:              mood = "calm_cool"
            elif g > 140 and r < 130:              mood = "natural_green"
            elif r > 160 and g > 140 and b < 100: mood = "warm_golden"
            elif stat.mean[0] < 60:                mood = "dark_moody"
            elif stat.mean[0] > 200:               mood = "bright_clean"
            else:                                  mood = "neutral"

            pil_data.append({
                "sec":                 round(frame_idx / fps, 2),
                "mean_rgb":            [round(x, 1) for x in [r, g, b]],
                "stddev_rgb":          [round(x, 1) for x in stat.stddev[:3]],
                "dominant_colors":     [list(p[1]) for p in pal[:6]],
                "perceived_brightness": round(0.299*r + 0.587*g + 0.114*b, 1),
                "color_mood":          mood,
                "contrast_score":      round(float(np.mean(stat.stddev[:3])), 1),
                "color_variance":      round(float(np.std([r, g, b])), 1),
            })
        frame_idx += 1

    cap.release()
    save_cache(path, "pil", pil_data)
    return pil_data


# ═══════════════════════════════════════════════════════════
# MODULE 6 — Aggregation (updated with scene + people stats)
# ═══════════════════════════════════════════════════════════
def aggregate_visual(frame_data: list, pil_data: list, moviepy_data: dict) -> dict:
    if not frame_data:
        return {}

    def avg(key):
        vals = [f[key] for f in frame_data if isinstance(f.get(key), (int, float))]
        return round(float(np.mean(vals)), 3) if vals else 0.0

    face_ratio  = round(sum(1 for f in frame_data if f["face_count"] > 0) / len(frame_data), 3)
    text_ratio  = round(sum(1 for f in frame_data if f["has_text_overlay"]) / len(frame_data), 3)
    eye_ratio   = round(sum(1 for f in frame_data if f.get("eye_contact")) / len(frame_data), 3)
    cuts        = sum(1 for f in frame_data if f["scene_change"])
    all_text    = " ".join(dict.fromkeys(f["ocr_text"] for f in frame_data if f["ocr_text"]))
    all_scripts = list({s for f in frame_data for s in f.get("scripts_detected", [])})

    moods     = [p["color_mood"] for p in pil_data]
    dom_mood  = max(set(moods), key=moods.count) if moods else "unknown"
    cam_moves = [f["camera_move"] for f in frame_data]
    dom_cam   = max(set(cam_moves), key=cam_moves.count)
    gaze_modes = [f.get("gaze_direction","unknown") for f in frame_data]
    dom_gaze  = max(set(gaze_modes), key=gaze_modes.count)

    # ③ Scene aggregation (NEW)
    settings  = [f["setting_type"] for f in frame_data if f.get("setting_type")]
    dom_setting = max(set(settings), key=settings.count) if settings else "unknown"
    io_tags   = [f["indoor_outdoor"] for f in frame_data if f.get("indoor_outdoor")]
    dom_io    = max(set(io_tags), key=io_tags.count) if io_tags else "unknown"
    lightings = [f["lighting"] for f in frame_data if f.get("lighting")]
    dom_light = max(set(lightings), key=lightings.count) if lightings else "unknown"
    wall_cols = [f["wall_color"] for f in frame_data if f.get("wall_color")]
    dom_wall  = max(set(wall_cols), key=wall_cols.count) if wall_cols else "unknown"

    # ④ People aggregation (NEW)
    avg_people   = avg("people_count")
    max_people   = max((f["people_count"] for f in frame_data), default=0)
    solo_frames  = sum(1 for f in frame_data if f["people_count"] == 1)
    multi_frames = sum(1 for f in frame_data if f["people_count"] > 1)

    scene_changes = [f["sec"] for f in frame_data if f["scene_change"]]
    avg_motion    = avg("motion_score")
    pacing        = "fast" if cuts > 8 else "medium" if cuts > 3 else "slow"

    return {
        "duration_sec":          moviepy_data["duration_sec"],
        "avg_brightness":        avg("brightness"),
        "avg_saturation":        avg("saturation"),
        "avg_sharpness":         avg("sharpness"),
        "avg_motion":            avg_motion,
        "avg_edge_density":      avg("edge_density"),
        "face_presence_ratio":   face_ratio,
        "eye_contact_ratio":     eye_ratio,
        "text_overlay_ratio":    text_ratio,
        "estimated_cuts":        cuts,
        "scene_changes":         scene_changes,
        "pacing":                pacing,
        "dominant_camera_move":  dom_cam,
        "dominant_gaze":         dom_gaze,
        "dominant_color_mood":   dom_mood,
        "all_ocr_text":          all_text,
        "scripts_detected":      all_scripts,
        "avg_text_readability":  avg("text_readability"),
        # ③ Scene fields (NEW)
        "dominant_setting":      dom_setting,
        "indoor_outdoor":        dom_io,
        "dominant_lighting":     dom_light,
        "dominant_wall_color":   dom_wall,
        # ④ People fields (NEW)
        "avg_people_in_frame":   round(avg_people, 1),
        "max_people_in_frame":   max_people,
        "solo_presenter_frames": solo_frames,
        "multi_person_frames":   multi_frames,
        "style": (
            "talking_head"     if face_ratio > 0.6            else
            "text_educational" if text_ratio > 0.5            else
            "broll_cinematic"  if avg_motion < 2              else
            "action_dynamic"   if avg_motion > 8              else
            "mixed"
        ),
    }


# ═══════════════════════════════════════════════════════════
# MODULE 7 — Content categorizer (unchanged logic, new fields passed through)
# ═══════════════════════════════════════════════════════════
def categorize_content(visual_summary: dict, audio_data: dict,
                       transcript_data: dict) -> dict:

    face_ratio = visual_summary["face_presence_ratio"]
    text_ratio = visual_summary["text_overlay_ratio"]
    eye_ratio  = visual_summary["eye_contact_ratio"]
    avg_motion = visual_summary["avg_motion"]
    pacing     = visual_summary["pacing"]
    duration   = visual_summary["duration_sec"]
    audio_mood = audio_data["audio_mood"]
    audio_type = audio_data["audio_type"]
    tempo      = audio_data["tempo_bpm"]
    has_voice  = audio_data["has_voice"]
    transcript = transcript_data.get("transcript", "")
    combined_text = (visual_summary.get("all_ocr_text","") + " " + transcript).lower().strip()

    hook_patterns = {
        "relatable_situation": ["feels like","when you","that moment","nobody","everyone","we all"],
        "list_format":         ["#1","#2","step","tip","reason","things","ways","here are","number"],
        "question_hook":       ["why","how","what if","did you","have you","?","can you"],
        "transformation":      ["before","after","change","went from","turned","glow up"],
        "controversy":         ["unpopular","nobody talks","secret","truth","actually","they don't"],
        "educational":         ["did you know","fact","study","research","science","learn","explains"],
        "storytelling":        ["story","happened","one day","so i","then i","this is","true story"],
        "call_to_action":      ["follow","subscribe","save this","share","comment","link in bio"],
        "tutorial":            ["how to","step by step","tutorial","guide","show you","do this"],
        "pov":                 ["pov","point of view","imagine","picture this"],
        # Hindi/Urdu horoscope + social logic patterns (NEW)
        "horoscope_astrology": ["राशि","राशिफल","भाग्य","ग्रह","zodiac","lucky","unlucky","star","fate",
                                "kundali","rashi","bhagya","kismat","किस्मत","भाग्यशाली"],
        "spiritual_wisdom":    ["karma","dharma","soul","God","भगवान","जीवन","सच","सत्य","आत्मा"],
    }
    detected_hooks = [k for k, phrases in hook_patterns.items()
                      if any(p in combined_text for p in phrases)]

    niche_keywords = {
        "humor_relatable":  ["driving","feels","nobody","lol","literally","when you","mood","hilarious"],
        "finance":          ["money","invest","rich","income","bank","finance","crypto","stock","wealth","profit"],
        "fitness":          ["gym","workout","fitness","muscle","diet","protein","exercise","body","calories"],
        "food":             ["recipe","cook","food","eat","restaurant","taste","meal","kitchen","ingredient"],
        "tech_ai":          ["ai","tech","tool","app","software","code","digital","automation","prompt"],
        "motivation":       ["success","mindset","hustle","grind","goal","dream","achieve","believe","win"],
        "beauty_fashion":   ["makeup","outfit","style","fashion","beauty","skincare","look","wear","trend"],
        "travel":           ["travel","trip","country","city","hotel","explore","adventure","beach","flight"],
        "education":        ["learn","study","school","fact","science","history","book","knowledge","explain"],
        "relationship":     ["love","relationship","partner","dating","couple","heart","feelings","breakup"],
        "entrepreneurship": ["business","entrepreneur","startup","brand","client","revenue","founder","launch"],
        "mental_health":    ["anxiety","stress","therapy","mental","mindfulness","healing","trauma","growth"],
        # Horoscope / astrology (NEW)
        "astrology_horoscope": ["राशि","भाग्य","ग्रह","zodiac","lucky","unlucky","horoscope","kundali",
                                "किस्मत","fate","rashi","astro","planet","dasha","nakshatra","साप","सर्प"],
    }
    niche_scores = {}
    for niche, keywords in niche_keywords.items():
        score = sum(1 for kw in keywords if kw in combined_text)
        if score > 0:
            niche_scores[niche] = score
    detected_niche = max(niche_scores, key=niche_scores.get) if niche_scores else "general"

    if face_ratio > 0.6 and has_voice and eye_ratio > 0.4:  content_format = "direct_to_camera"
    elif face_ratio > 0.6 and has_voice:                     content_format = "talking_head"
    elif face_ratio > 0.6 and not has_voice:                 content_format = "silent_presenter"
    elif text_ratio > 0.7:                                   content_format = "text_only"
    elif text_ratio > 0.5 and face_ratio > 0.3:              content_format = "text_plus_face"
    elif avg_motion > 8:                                     content_format = "action_broll"
    elif avg_motion < 2:                                     content_format = "static_broll"
    else:                                                    content_format = "mixed_broll"

    wpm = transcript_data.get("speaking_rate_wpm", 0)
    if wpm > 160:   speaking_style = "fast_energetic"
    elif wpm > 120: speaking_style = "conversational"
    elif wpm > 80:  speaking_style = "deliberate"
    elif wpm > 0:   speaking_style = "slow_dramatic"
    else:           speaking_style = "no_speech"

    score = 0
    if pacing == "fast":             score += 20
    elif pacing == "medium":         score += 10
    if audio_mood in ["hype_energetic","upbeat_positive"]: score += 12
    if detected_hooks:               score += 15
    if text_ratio > 0.5:            score += 8
    if 100 <= tempo <= 140:         score += 8
    if "relatable_situation" in detected_hooks: score += 10
    if "call_to_action" in detected_hooks:      score += 5
    if has_voice and transcript:    score += 10
    if detected_niche != "general": score += 8
    if eye_ratio > 0.4:             score += 4
    if audio_data.get("silence_ratio", 0) > 0.2: score -= 5
    if pacing == "slow":             score -= 5
    if not detected_hooks:           score -= 5
    score = max(0, min(score, 100))

    if duration < 8:    duration_tag = "ultra_short"
    elif duration < 15: duration_tag = "short"
    elif duration < 30: duration_tag = "medium"
    elif duration < 60: duration_tag = "long"
    else:               duration_tag = "extended"

    recommendations = []
    if pacing == "slow":
        recommendations.append("⚡ Speed up pacing — aim for a cut every 2-3 seconds")
    if not detected_hooks:
        recommendations.append("🪝 Add a stronger hook in the first 1-2 seconds")
    if text_ratio < 0.3:
        recommendations.append("📝 Add text overlays — 85% of reels watched on mute")
    if tempo < 100:
        recommendations.append("🎵 Use faster trending audio (100-130 BPM performs best)")
    if face_ratio == 0:
        recommendations.append("👤 Add a face — videos with faces get 38% more engagement")
    if eye_ratio < 0.3 and face_ratio > 0.3:
        recommendations.append("👁️ Improve eye contact with camera for trust signal")
    if audio_data.get("silence_ratio", 0) > 0.15:
        recommendations.append("🔇 Reduce silence gaps — viewers drop off in quiet sections")
    if "call_to_action" not in detected_hooks:
        recommendations.append("📣 Add a CTA (save, follow, comment) before the end")
    if not recommendations:
        recommendations.append("✅ Strong format — replicate this structure exactly")

    return {
        "detected_niche":    detected_niche,
        "niche_scores":      niche_scores,
        "content_format":    content_format,
        "hook_formats":      detected_hooks,
        "speaking_style":    speaking_style,
        "audio_mood":        audio_mood,
        "audio_type":        audio_type,
        "duration_category": duration_tag,
        "viral_score":       score,
        "recommendations":   recommendations,
        "replication_template": {
            "format":         content_format,
            "hook_style":     detected_hooks[0] if detected_hooks else "unknown",
            "audio_mood":     audio_mood,
            "pacing":         pacing,
            "use_face":       face_ratio > 0.3,
            "use_eye_contact": eye_ratio > 0.3,
            "use_text":       text_ratio > 0.3,
            "target_bpm":     round(tempo),
            "niche":          detected_niche,
            "speaking_style": speaking_style,
        }
    }


# ═══════════════════════════════════════════════════════════
# MODULE 8 — AI brief (unchanged)
# ═══════════════════════════════════════════════════════════
def generate_ai_brief(full_report: dict) -> dict:
    if not ANTHROPIC_AVAILABLE or not Config.ANTHROPIC_API_KEY or not Config.ENABLE_AI_BRIEF:
        return {
            "replication_script": "(AI brief disabled — set ANTHROPIC_API_KEY env var)",
            "hook_variations": [],
            "caption_pack": [],
            "posting_strategy": "",
        }

    client  = anthropic.Anthropic(api_key=Config.ANTHROPIC_API_KEY)
    cat     = full_report["content_category"]
    audio   = full_report["audio_analysis"]
    visual  = full_report["visual_summary"]
    transcript = full_report.get("speech_transcript", {}).get("transcript", "")
    ocr_text   = visual.get("all_ocr_text", "")
    container  = full_report["container"]

    compressed = {
        "duration_sec":       container["duration_sec"],
        "niche":              cat["detected_niche"],
        "content_format":     cat["content_format"],
        "hook_formats":       cat["hook_formats"],
        "speaking_style":     cat["speaking_style"],
        "viral_score":        cat["viral_score"],
        "pacing":             visual["pacing"],
        "cuts":               visual["estimated_cuts"],
        "face_ratio":         visual["face_presence_ratio"],
        "text_overlay_ratio": visual["text_overlay_ratio"],
        "audio_mood":         audio["audio_mood"],
        "audio_type":         audio["audio_type"],
        "tempo_bpm":          audio["tempo_bpm"],
        "key":                audio["detected_key"],
        "transcript_sample":  transcript[:500] if transcript else "(none)",
        "ocr_text_sample":    ocr_text[:300]   if ocr_text   else "(none)",
        "recommendations":    cat["recommendations"],
        "indoor_outdoor":     visual.get("indoor_outdoor","unknown"),
        "setting_type":       visual.get("dominant_setting","unknown"),
        "avg_people_in_frame": visual.get("avg_people_in_frame", 1),
    }

    prompt = f"""You are a viral short-form video strategist specializing in Instagram Reels and TikTok.

Here is a technical analysis of a reel with {compressed['viral_score']}/100 viral score:
{json.dumps(compressed, indent=2)}

Please produce the following (in JSON format only, no markdown):

1. "replication_script": Shot-by-shot script to recreate this reel (6-10 shots).
   Format: {{ "shot": 1, "time": "0-2s", "visual": "...", "audio": "...", "text_overlay": "...", "note": "..." }}

2. "hook_variations": 5 opening hook lines in the same niche and style.
   Format: {{ "hook": "...", "type": "question/statement/relatable/controversy/educational" }}

3. "caption_pack": 3 complete Instagram captions with hashtags.
   Format: {{ "caption": "...", "hashtags": [...], "cta": "..." }}

4. "posting_strategy": One paragraph on when and how to post for max reach.

5. "content_improvement": 3 specific, actionable improvements.

Return ONLY valid JSON."""

    try:
        response = client.messages.create(
            model=Config.ANTHROPIC_MODEL,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = "\n".join(raw.split("\n")[1:-1])
        return json.loads(raw)
    except Exception as e:
        return {
            "error":              str(e),
            "replication_script": [],
            "hook_variations":    [],
            "caption_pack":       [],
            "posting_strategy":   "API call failed — check ANTHROPIC_API_KEY",
        }


# ═══════════════════════════════════════════════════════════
# MODULE 9 — HTML report (adds Scene & People card)
# ═══════════════════════════════════════════════════════════
def generate_html_report(full_report: dict, ai_brief: dict, output_path: str):
    cat      = full_report["content_category"]
    audio    = full_report["audio_analysis"]
    visual   = full_report["visual_summary"]
    container= full_report["container"]
    speech   = full_report.get("speech_transcript", {})
    score    = cat["viral_score"]
    score_color = "#22c55e" if score >= 70 else "#f59e0b" if score >= 40 else "#ef4444"

    shots_html = ""
    for s in ai_brief.get("replication_script", []):
        if isinstance(s, dict):
            shots_html += f"""
            <div class="shot-card">
              <div class="shot-num">Shot {s.get('shot','?')} <span>{s.get('time','')}</span></div>
              <div class="shot-row"><b>🎬 Visual:</b> {s.get('visual','')}</div>
              <div class="shot-row"><b>🎵 Audio:</b> {s.get('audio','')}</div>
              <div class="shot-row"><b>📝 Text:</b> {s.get('text_overlay','')}</div>
              <div class="shot-note">{s.get('note','')}</div>
            </div>"""

    hooks_html = ""
    for h in ai_brief.get("hook_variations", []):
        if isinstance(h, dict):
            hooks_html += f'<div class="hook-pill"><span class="hook-type">{h.get("type","")}</span>{h.get("hook","")}</div>'

    captions_html = ""
    for c in ai_brief.get("caption_pack", []):
        if isinstance(c, dict):
            tags = " ".join(f"#{t.lstrip('#')}" for t in c.get("hashtags",[]))
            captions_html += f"""
            <div class="caption-card">
              <p>{c.get('caption','')}</p>
              <div class="hashtags">{tags}</div>
              <div class="cta">📣 {c.get('cta','')}</div>
            </div>"""

    improvements_html = ""
    for imp in ai_brief.get("content_improvement", []):
        improvements_html += f"<li>{imp}</li>"

    rec_html = "".join(f"<li>{r}</li>" for r in cat.get("recommendations",[]))
    transcript_text = speech.get("transcript","") or "(no spoken audio detected)"

    energy_data  = json.dumps([e["energy"] for e in audio.get("energy_over_time",[])][:60])
    energy_times = json.dumps([e["sec"]    for e in audio.get("energy_over_time",[])][:60])

    scripts_str = ", ".join(visual.get("scripts_detected", [])) or "latin"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Reel Analysis Report</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
<style>
  *{{ box-sizing:border-box; margin:0; padding:0; }}
  body{{ font-family:'Segoe UI',system-ui,sans-serif; background:#0f0f17; color:#e2e8f0; }}
  .header{{ background:linear-gradient(135deg,#1a1a2e,#16213e,#0f3460); padding:40px; text-align:center; }}
  .header h1{{ font-size:2.4em; font-weight:800; color:#fff; }}
  .header p{{ color:#94a3b8; margin-top:8px; font-size:1.1em; }}
  .score-badge{{ display:inline-block; background:{score_color}22; border:3px solid {score_color};
    color:{score_color}; font-size:3em; font-weight:900; padding:16px 32px;
    border-radius:16px; margin:20px 0; }}
  .grid{{ display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); gap:20px;
    padding:30px; max-width:1400px; margin:0 auto; }}
  .card{{ background:#1e1e2e; border-radius:16px; padding:24px; border:1px solid #2d2d44; }}
  .card h2{{ font-size:1em; color:#94a3b8; text-transform:uppercase; letter-spacing:.08em; margin-bottom:16px; }}
  .stat{{ display:flex; justify-content:space-between; padding:8px 0; border-bottom:1px solid #2d2d44; }}
  .stat:last-child{{ border-bottom:none; }}
  .stat-label{{ color:#94a3b8; font-size:.9em; }}
  .stat-value{{ font-weight:700; color:#e2e8f0; }}
  .tag{{ display:inline-block; background:#3b1f5e; color:#c084fc; padding:4px 12px;
    border-radius:20px; font-size:.8em; margin:3px; font-weight:600; }}
  .section{{ max-width:1400px; margin:0 auto; padding:0 30px 30px; }}
  .section-title{{ font-size:1.4em; font-weight:700; margin:30px 0 16px;
    color:#fff; border-left:4px solid #7c3aed; padding-left:12px; }}
  .shot-card{{ background:#1e1e2e; border-radius:12px; padding:20px;
    border-left:4px solid #7c3aed; margin-bottom:12px; }}
  .shot-num{{ font-size:1.1em; font-weight:800; color:#a78bfa; margin-bottom:10px; }}
  .shot-num span{{ font-size:.8em; color:#64748b; font-weight:400; margin-left:8px; }}
  .shot-row{{ font-size:.9em; margin:4px 0; color:#cbd5e1; }}
  .shot-note{{ margin-top:8px; font-size:.85em; color:#7c3aed; font-style:italic; }}
  .hook-pill{{ background:#1e1e2e; border:1px solid #2d2d44; border-radius:12px;
    padding:14px 18px; margin-bottom:10px; font-size:.95em; }}
  .hook-type{{ display:inline-block; background:#7c3aed22; color:#a78bfa;
    font-size:.75em; padding:2px 8px; border-radius:10px; margin-right:8px; font-weight:700; }}
  .caption-card{{ background:#1e1e2e; border-radius:12px; padding:20px;
    border:1px solid #2d2d44; margin-bottom:12px; }}
  .caption-card p{{ color:#e2e8f0; line-height:1.6; }}
  .hashtags{{ color:#7c3aed; font-size:.85em; margin-top:10px; }}
  .cta{{ color:#22c55e; font-size:.85em; margin-top:6px; }}
  .rec-list{{ list-style:none; }}
  .rec-list li{{ padding:10px 14px; background:#1e1e2e; border-radius:8px;
    margin-bottom:8px; border-left:3px solid #f59e0b; font-size:.95em; }}
  .transcript-box{{ background:#1e1e2e; border-radius:12px; padding:20px;
    border:1px solid #2d2d44; line-height:1.7; color:#cbd5e1; font-size:.95em;
    font-family: 'Noto Sans Devanagari', 'Segoe UI', sans-serif; }}
  canvas{{ max-height:200px; }}
  .strategy-box{{ background:#1e1e2e; border-radius:12px; padding:20px;
    border-left:4px solid #22c55e; color:#cbd5e1; line-height:1.7; }}
  .improve-list{{ list-style:none; }}
  .improve-list li{{ padding:10px 14px; background:#1e1e2e; border-radius:8px;
    margin-bottom:8px; border-left:3px solid #22c55e; font-size:.95em; }}
  .footer{{ text-align:center; padding:30px; color:#475569; font-size:.85em; }}
</style>
</head>
<body>

<div class="header">
  <h1>🎬 Reel Intelligence Report</h1>
  <p>Generated {datetime.now().strftime("%B %d, %Y at %H:%M")} · Reel Analyzer v2.1</p>
  <div class="score-badge">{score}/100 Viral Score</div>
</div>

<div class="grid">

  <div class="card">
    <h2>📦 Container</h2>
    <div class="stat"><span class="stat-label">Duration</span><span class="stat-value">{container['duration_sec']}s</span></div>
    <div class="stat"><span class="stat-label">Resolution</span><span class="stat-value">{container['resolution'][0]}×{container['resolution'][1]}</span></div>
    <div class="stat"><span class="stat-label">FPS</span><span class="stat-value">{container['fps']:.1f}</span></div>
    <div class="stat"><span class="stat-label">Format</span><span class="stat-value">{container['format_hint']}</span></div>
    <div class="stat"><span class="stat-label">Has Audio</span><span class="stat-value">{'✅' if container['has_audio'] else '❌'}</span></div>
  </div>

  <div class="card">
    <h2>🎯 Content Identity</h2>
    <div class="stat"><span class="stat-label">Niche</span><span class="stat-value">{cat['detected_niche'].replace('_',' ').title()}</span></div>
    <div class="stat"><span class="stat-label">Format</span><span class="stat-value">{cat['content_format'].replace('_',' ').title()}</span></div>
    <div class="stat"><span class="stat-label">Duration Type</span><span class="stat-value">{cat['duration_category'].replace('_',' ').title()}</span></div>
    <div class="stat"><span class="stat-label">Speaking Style</span><span class="stat-value">{cat['speaking_style'].replace('_',' ').title()}</span></div>
    <div style="margin-top:12px;">
      {''.join(f'<span class="tag">{h.replace("_"," ")}</span>' for h in cat.get('hook_formats',[]))}
    </div>
  </div>

  <div class="card">
    <h2>🎵 Audio Profile</h2>
    <div class="stat"><span class="stat-label">Tempo</span><span class="stat-value">{audio['tempo_bpm']} BPM</span></div>
    <div class="stat"><span class="stat-label">Key</span><span class="stat-value">{audio['detected_key']}</span></div>
    <div class="stat"><span class="stat-label">Mood</span><span class="stat-value">{audio['audio_mood'].replace('_',' ').title()}</span></div>
    <div class="stat"><span class="stat-label">Type</span><span class="stat-value">{audio['audio_type'].replace('_',' ').title()}</span></div>
    <div class="stat"><span class="stat-label">Harmony</span><span class="stat-value">{audio['harmony_pct']}%</span></div>
    <div class="stat"><span class="stat-label">Percussion</span><span class="stat-value">{audio['percussion_pct']}%</span></div>
  </div>

  <div class="card">
    <h2>🖼️ Visual Metrics</h2>
    <div class="stat"><span class="stat-label">Avg Brightness</span><span class="stat-value">{visual['avg_brightness']:.1f}</span></div>
    <div class="stat"><span class="stat-label">Avg Saturation</span><span class="stat-value">{visual['avg_saturation']:.1f}</span></div>
    <div class="stat"><span class="stat-label">Face Presence</span><span class="stat-value">{int(visual['face_presence_ratio']*100)}%</span></div>
    <div class="stat"><span class="stat-label">Eye Contact</span><span class="stat-value">{int(visual['eye_contact_ratio']*100)}%</span></div>
    <div class="stat"><span class="stat-label">Text Overlay</span><span class="stat-value">{int(visual['text_overlay_ratio']*100)}%</span></div>
    <div class="stat"><span class="stat-label">Cuts Detected</span><span class="stat-value">{visual['estimated_cuts']} ({visual['pacing']} pacing)</span></div>
    <div class="stat"><span class="stat-label">Scripts in Text</span><span class="stat-value">{scripts_str}</span></div>
  </div>

  <!-- ③ NEW: Scene & Background card -->
  <div class="card">
    <h2>🏠 Scene & Background</h2>
    <div class="stat"><span class="stat-label">Indoor / Outdoor</span><span class="stat-value">{visual.get('indoor_outdoor','?').title()}</span></div>
    <div class="stat"><span class="stat-label">Setting</span><span class="stat-value">{visual.get('dominant_setting','?').replace('_',' ').title()}</span></div>
    <div class="stat"><span class="stat-label">Lighting</span><span class="stat-value">{visual.get('dominant_lighting','?').replace('_',' ').title()}</span></div>
    <div class="stat"><span class="stat-label">Wall / BG Color</span><span class="stat-value">{visual.get('dominant_wall_color','?').replace('_',' ').title()}</span></div>
  </div>

  <!-- ④ NEW: People card -->
  <div class="card">
    <h2>👥 People in Frame</h2>
    <div class="stat"><span class="stat-label">Avg People / Frame</span><span class="stat-value">{visual.get('avg_people_in_frame', 0)}</span></div>
    <div class="stat"><span class="stat-label">Max People (any frame)</span><span class="stat-value">{visual.get('max_people_in_frame', 0)}</span></div>
    <div class="stat"><span class="stat-label">Solo Presenter Frames</span><span class="stat-value">{visual.get('solo_presenter_frames', 0)}</span></div>
    <div class="stat"><span class="stat-label">Multi-Person Frames</span><span class="stat-value">{visual.get('multi_person_frames', 0)}</span></div>
  </div>

</div>

<div class="section">
  <div class="section-title">📈 Audio Energy Over Time</div>
  <div class="card">
    <canvas id="energyChart"></canvas>
  </div>

  <div class="section-title">🚀 Recommendations</div>
  <ul class="rec-list">{rec_html}</ul>

  <div class="section-title">🎙️ Speech Transcript (Whisper · Hindi)</div>
  <div class="transcript-box">{transcript_text}</div>

  <div class="section-title">🎬 Replication Shot List</div>
  {shots_html or '<div class="transcript-box">Enable AI brief to generate shot list.</div>'}

  <div class="section-title">🪝 Hook Variations</div>
  {hooks_html or '<div class="transcript-box">Enable AI brief to generate hook variations.</div>'}

  <div class="section-title">📣 Caption Pack</div>
  {captions_html or '<div class="transcript-box">Enable AI brief to generate captions.</div>'}

  <div class="section-title">📆 Posting Strategy</div>
  <div class="strategy-box">{ai_brief.get('posting_strategy','Enable AI brief to generate strategy.')}</div>

  <div class="section-title">⚡ Content Improvements</div>
  <ul class="improve-list">{improvements_html or '<li>Enable AI brief to generate improvements.</li>'}</ul>
</div>

<div class="footer">Reel Analyzer v2.1 · Whisper (Hindi) · MediaPipe · EasyOCR (en+hi) · HOG People Detector</div>

<script>
const ctx = document.getElementById('energyChart').getContext('2d');
new Chart(ctx, {{
  type: 'line',
  data: {{
    labels: {energy_times},
    datasets: [{{
      label: 'Audio Energy',
      data: {energy_data},
      borderColor: '#7c3aed',
      backgroundColor: '#7c3aed22',
      fill: true,
      tension: 0.4,
      pointRadius: 0,
    }}]
  }},
  options: {{
    responsive: true,
    plugins: {{ legend: {{ labels: {{ color:'#94a3b8' }} }} }},
    scales: {{
      x: {{ ticks: {{ color:'#64748b' }}, grid: {{ color:'#1e1e2e' }} }},
      y: {{ ticks: {{ color:'#64748b' }}, grid: {{ color:'#2d2d44' }} }},
    }}
  }}
}});
</script>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"📄 HTML report saved: {output_path}")


# ═══════════════════════════════════════════════════════════
# MODULE 10 — Storyboard (updated labels)
# ═══════════════════════════════════════════════════════════
def generate_storyboard(keyframes: list, frame_data: list,
                        audio_data: dict, category: dict,
                        output_path: str, cols: int = 4):
    if not keyframes:
        return

    thumb_w, thumb_h = 360, 640
    label_h  = 140   # slightly taller to fit new fields
    rows     = math.ceil(len(keyframes) / cols)
    board    = Image.new("RGB", (cols*thumb_w, rows*(thumb_h+label_h)+90), (10,10,14))
    draw     = ImageDraw.Draw(board)

    try:
        font       = ImageFont.truetype("arial.ttf", 14)
        font_small = ImageFont.truetype("arial.ttf", 11)
        font_bold  = ImageFont.truetype("arialbd.ttf", 15)
        font_title = ImageFont.truetype("arialbd.ttf", 18)
    except:
        font = font_small = font_bold = font_title = ImageFont.load_default()

    draw.rectangle([0,0,cols*thumb_w,85], fill=(20,10,40))
    niche  = category["detected_niche"].upper().replace("_"," ")
    fmt    = category["content_format"].replace("_"," ")
    score  = category["viral_score"]
    bpm    = audio_data["tempo_bpm"]
    key    = audio_data["detected_key"]
    amood  = audio_data["audio_mood"].replace("_"," ")
    draw.text((10, 8),  f"🎯 {niche}  |  {fmt}  |  VIRAL: {score}/100", font=font_title, fill=(255,220,50))
    draw.text((10,35),  f"🎵 {bpm} BPM  ·  {key}  ·  {amood}", font=font_bold, fill=(160,210,255))
    draw.text((10,58),  f"🪝 Hooks: {', '.join(category['hook_formats']) or 'none detected'}", font=font, fill=(180,255,180))

    for i, (frame, fd) in enumerate(zip(keyframes, frame_data)):
        col = i % cols
        row = i // cols
        x   = col * thumb_w
        y   = row * (thumb_h + label_h) + 90

        img      = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((thumb_w, thumb_h))
        img_draw = ImageDraw.Draw(img)

        # Face boxes (green = eye contact, orange = no eye contact)
        for fp in fd.get("face_positions", []):
            fx = int(fp["cx"] * thumb_w)
            fy = int(fp["cy"] * thumb_h)
            sz = int(math.sqrt(max(fp.get("size_ratio",0.05), 0.01)) * thumb_w)
            col_c = (0,255,80) if fd.get("eye_contact") else (255,160,0)
            img_draw.ellipse([fx-sz//2, fy-sz//2, fx+sz//2, fy+sz//2], outline=col_c, width=3)

        # HOG body boxes (blue)
        for bp in fd.get("body_positions", []):
            bx = int(bp["cx"] * thumb_w)
            by = int(bp["cy"] * thumb_h)
            img_draw.rectangle([bx-30, by-60, bx+30, by+60], outline=(80,160,255), width=2)

        # OCR word boxes (yellow)
        for wrd in fd.get("ocr_words",[])[:8]:
            if "bbox" in wrd:
                bx1 = int(wrd["bbox"][0] * thumb_w / frame.shape[1])
                by1 = int(wrd["bbox"][1] * thumb_h / frame.shape[0])
                bx2 = int(wrd["bbox"][2] * thumb_w / frame.shape[1])
                by2 = int(wrd["bbox"][3] * thumb_h / frame.shape[0])
                c   = (255,100,100) if wrd.get("script") == "devanagari" else (255,220,0)
                img_draw.rectangle([bx1,by1,bx2,by2], outline=c, width=1)

        # Beat marker
        beat_near = any(abs(bt - fd["sec"]) < 0.5 for bt in audio_data["beat_times"])
        if beat_near:
            img_draw.rectangle([0, thumb_h-8, thumb_w, thumb_h], fill=(200,40,40))
            img_draw.text((4, thumb_h-20), "♪ BEAT", font=font_small, fill=(255,255,255))

        if fd.get("scene_change"):
            img_draw.rectangle([0,0,thumb_w,6], fill=(255,80,0))

        # Indoor/outdoor badge
        io_col  = (80,200,120) if fd.get("indoor_outdoor") == "outdoor" else (100,140,255)
        io_text = fd.get("indoor_outdoor","?")[:3].upper()
        img_draw.rectangle([thumb_w-45, 4, thumb_w-4, 22], fill=io_col)
        img_draw.text((thumb_w-42, 6), io_text, font=font_small, fill=(0,0,0))

        board.paste(img, (x, y))

        # Label panel
        ly = y + thumb_h
        draw.rectangle([x, ly, x+thumb_w, ly+label_h], fill=(14,14,22))
        ec = "👁️" if fd.get("eye_contact") else ""
        draw.text((x+5, ly+4),   f"⏱ {fd['sec']}s  📷 {fd['camera_move']}  🌡 {fd['color_temp']} {ec}", font=font, fill=(160,210,255))
        draw.text((x+5, ly+22),  f"💨 {fd['motion_score']:.1f}  👤 {fd['face_count']} faces  👥 {fd.get('people_count',0)} people", font=font, fill=(200,200,200))
        draw.text((x+5, ly+40),  f"🏠 {fd.get('setting_type','?')[:22]}  💡 {fd.get('lighting','?')[:12]}", font=font_small, fill=(255,200,100))
        draw.text((x+5, ly+56),  f"🎨 bg:{fd.get('wall_color','?')[:14]}  📝 {'✅' if fd['has_text_overlay'] else '❌'}", font=font_small, fill=(255,220,80))
        draw.text((x+5, ly+72),  f"{fd['hook_text'][:42]}{'…' if len(fd['hook_text'])>42 else ''}", font=font_small, fill=(200,200,200))
        draw.text((x+5, ly+88),  f"♪ {'BEAT' if beat_near else '—'}  gaze:{fd.get('gaze_direction','?')[:14]}", font=font_small, fill=(255,100,100) if beat_near else (80,80,100))
        draw.text((x+5, ly+104), f"scripts: {', '.join(fd.get('scripts_detected',[]) or ['?'])[:25]}", font=font_small, fill=(140,220,140))
        draw.text((x+5, ly+120), f"📖 readability: {fd.get('text_readability',0):.0f}%  ✂️ {'CUT' if fd.get('scene_change') else ''}", font=font_small, fill=(100,180,100))

        draw.rectangle([x, y, x+42, y+22], fill=(120,30,200))
        draw.text((x+5, y+4), f"#{i+1}", font=font_bold, fill="white")

    board.save(output_path, quality=95)
    print(f"🎬 Storyboard saved: {output_path}")


# ═══════════════════════════════════════════════════════════
# MODULE 11 — CSV aggregator (adds new fields)
# ═══════════════════════════════════════════════════════════
def append_to_csv(full_report: dict, video_path: str):
    cat    = full_report["content_category"]
    audio  = full_report["audio_analysis"]
    visual = full_report["visual_summary"]
    speech = full_report.get("speech_transcript",{})
    row = {
        "filename":           os.path.basename(video_path),
        "analyzed_at":        datetime.now().isoformat(),
        "duration_sec":       full_report["container"]["duration_sec"],
        "viral_score":        cat["viral_score"],
        "niche":              cat["detected_niche"],
        "content_format":     cat["content_format"],
        "pacing":             visual["pacing"],
        "cuts":               visual["estimated_cuts"],
        "hook_formats":       "|".join(cat.get("hook_formats",[])),
        "tempo_bpm":          audio["tempo_bpm"],
        "detected_key":       audio["detected_key"],
        "audio_mood":         audio["audio_mood"],
        "audio_type":         audio["audio_type"],
        "face_ratio":         visual["face_presence_ratio"],
        "eye_contact_ratio":  visual["eye_contact_ratio"],
        "text_ratio":         visual["text_overlay_ratio"],
        "avg_brightness":     visual["avg_brightness"],
        "avg_saturation":     visual["avg_saturation"],
        "avg_motion":         visual["avg_motion"],
        "harmony_pct":        audio["harmony_pct"],
        "speaking_style":     cat.get("speaking_style",""),
        "transcript_words":   speech.get("word_count",0),
        "speaking_rate_wpm":  speech.get("speaking_rate_wpm",0),
        "silence_ratio":      audio.get("silence_ratio",0),
        # NEW fields
        "indoor_outdoor":     visual.get("indoor_outdoor",""),
        "setting_type":       visual.get("dominant_setting",""),
        "wall_color":         visual.get("dominant_wall_color",""),
        "lighting":           visual.get("dominant_lighting",""),
        "avg_people":         visual.get("avg_people_in_frame",0),
        "max_people":         visual.get("max_people_in_frame",0),
        "scripts_detected":   "|".join(visual.get("scripts_detected",[])),
    }
    file_exists = os.path.exists(Config.CSV_LOG)
    with open(Config.CSV_LOG, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"📊 CSV log updated: {Config.CSV_LOG}")


# ═══════════════════════════════════════════════════════════
# MODULE 12 — Category folder organizer
# ═══════════════════════════════════════════════════════════
def organize_by_category(report_path: str, storyboard_path: str,
                         niche: str, viral_score: int):
    import shutil
    tier   = "top_tier" if viral_score >= 70 else "mid_tier" if viral_score >= 40 else "low_tier"
    folder = os.path.join(Config.OUTPUT_DIR, "by_category", niche, tier)
    os.makedirs(folder, exist_ok=True)
    if os.path.exists(report_path):
        shutil.copy(report_path, os.path.join(folder, os.path.basename(report_path)))
    if os.path.exists(storyboard_path):
        shutil.copy(storyboard_path, os.path.join(folder, os.path.basename(storyboard_path)))
    print(f"📁 Filed under: by_category/{niche}/{tier}/")


# ═══════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════
def analyze_video(video_path: str) -> dict:
    print(f"\n{'═'*62}")
    print(f"  🎬 Analyzing: {os.path.basename(video_path)}")
    print(f"{'═'*62}\n")

    slug = Path(video_path).stem[:30].replace(" ","_")
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("📦 [1/7] Container metadata...")
    moviepy_data = analyze_container(video_path)
    print(f"    Duration: {moviepy_data['duration_sec']}s  FPS: {moviepy_data['fps']}  "
          f"Res: {moviepy_data['resolution']}  Format: {moviepy_data['format_hint']}")

    print("\n🎙️  [2/7] Speech-to-text (Whisper, lang=hi)...")
    speech_data = transcribe_speech(video_path)
    if speech_data["transcript"]:
        print(f"    Language: {speech_data['language']}  Words: {speech_data['word_count']}  "
              f"WPM: {speech_data['speaking_rate_wpm']}")
        print(f"    Sample: «{speech_data['transcript'][:120]}»")
    else:
        print("    (no speech detected)")

    print("\n🎵 [3/7] Deep audio analysis (Librosa)...")
    audio_data = analyze_audio(video_path)
    print(f"    {audio_data['tempo_bpm']} BPM  Key: {audio_data['detected_key']}  "
          f"Mood: {audio_data['audio_mood']}  Type: {audio_data['audio_type']}")

    print("\n🖼️  [4/7] Frame analysis (OpenCV + MediaPipe + EasyOCR en+hi + HOG)...")
    frame_data, keyframes = analyze_frames(video_path)

    print("\n🎨 [5/7] Color palette (PIL)...")
    pil_data = analyze_color_palette(video_path)

    print("\n📊 [6/7] Aggregating & categorizing...")
    visual_summary = aggregate_visual(frame_data, pil_data, moviepy_data)
    category       = categorize_content(visual_summary, audio_data, speech_data)
    print(f"    Niche: {category['detected_niche']}  Format: {category['content_format']}  "
          f"Viral Score: {category['viral_score']}/100")
    print(f"    Scene: {visual_summary.get('indoor_outdoor','?')} / {visual_summary.get('dominant_setting','?')}")
    print(f"    People avg: {visual_summary.get('avg_people_in_frame',0)}  "
          f"max: {visual_summary.get('max_people_in_frame',0)}")
    print(f"    Scripts in text: {visual_summary.get('scripts_detected', [])}")

    full_report = {
        "file":              video_path,
        "analyzed_at":       datetime.now().isoformat(),
        "container":         moviepy_data,
        "speech_transcript": speech_data,
        "audio_analysis":    audio_data,
        "visual_summary":    visual_summary,
        "content_category":  category,
        "frame_analysis":    frame_data,
        "color_analysis":    pil_data,
    }

    print("\n🤖 [7/7] Generating outputs...")
    ai_brief = {}
    if Config.ENABLE_AI_BRIEF and Config.ANTHROPIC_API_KEY:
        print("  🧠 Requesting AI replication brief (Claude)...")
        ai_brief = generate_ai_brief(full_report)
        full_report["ai_brief"] = ai_brief
    else:
        print("  ⏭️  AI brief skipped (set ANTHROPIC_API_KEY to enable)")

    json_path = os.path.join(Config.REPORTS_DIR, f"{slug}_{ts}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)
    print(f"  ✅ JSON report: {json_path}")

    sb_path   = os.path.join(Config.REPORTS_DIR, f"{slug}_{ts}_storyboard.jpg")
    html_path = os.path.join(Config.REPORTS_DIR, f"{slug}_{ts}_report.html")
    generate_storyboard(keyframes, frame_data, audio_data, category, sb_path)
    generate_html_report(full_report, ai_brief, html_path)
    append_to_csv(full_report, video_path)
    organize_by_category(html_path, sb_path, category["detected_niche"], category["viral_score"])

    print(f"\n{'═'*62}")
    print(f"  ✅ ANALYSIS COMPLETE")
    print(f"{'─'*62}")
    print(f"  🎯 Niche:        {category['detected_niche'].replace('_',' ').title()}")
    print(f"  📋 Format:       {category['content_format'].replace('_',' ').title()}")
    print(f"  🏆 Viral Score:  {category['viral_score']}/100")
    print(f"  🎵 BPM / Key:   {audio_data['tempo_bpm']} / {audio_data['detected_key']}")
    print(f"  ✂️  Pacing:       {visual_summary['pacing']} ({visual_summary['estimated_cuts']} cuts)")
    print(f"  👤 Face:         {int(visual_summary['face_presence_ratio']*100)}%  "
          f"Eye contact: {int(visual_summary['eye_contact_ratio']*100)}%")
    print(f"  👥 People avg:   {visual_summary.get('avg_people_in_frame',0)}  "
          f"max: {visual_summary.get('max_people_in_frame',0)}")
    print(f"  🏠 Scene:        {visual_summary.get('indoor_outdoor','?')} / "
          f"{visual_summary.get('dominant_setting','?')}")
    print(f"  💡 Lighting:     {visual_summary.get('dominant_lighting','?')}")
    print(f"  🎨 BG Color:     {visual_summary.get('dominant_wall_color','?')}")
    print(f"  📝 Text overlay: {int(visual_summary['text_overlay_ratio']*100)}%  "
          f"Scripts: {visual_summary.get('scripts_detected',[])}")
    print(f"  🪝 Hooks:        {', '.join(category['hook_formats']) or 'none'}")
    print(f"  🎙️  Speech:       {speech_data.get('word_count',0)} words @ "
          f"{speech_data.get('speaking_rate_wpm',0)} WPM")
    print(f"\n  📄 HTML Report → {html_path}")
    print(f"  🎬 Storyboard  → {sb_path}")
    print(f"  📊 CSV Log     → {Config.CSV_LOG}")
    print(f"{'═'*62}\n")
    return full_report


def batch_analyze(folder: str):
    exts   = {".mp4",".mov",".avi",".mkv",".webm",".m4v"}
    videos = [str(p) for p in Path(folder).rglob("*") if p.suffix.lower() in exts]
    print(f"\n📂 Found {len(videos)} video(s) in {folder}\n")
    for i, vp in enumerate(videos, 1):
        print(f"[{i}/{len(videos)}] {os.path.basename(vp)}")
        try:
            analyze_video(vp)
        except Exception as e:
            print(f"  ❌ Failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reel Analyzer v2.1")
    parser.add_argument("path",   nargs="?", help="Video file or folder")
    parser.add_argument("--video",  type=str)
    parser.add_argument("--folder", type=str)
    parser.add_argument("--no-ai",  action="store_true")
    parser.add_argument("--whisper-model", default="base",
                        choices=["tiny","base","small","medium","large"])
    parser.add_argument("--lang", default="hi",
                        help="Whisper language code (default: hi for Hindi). "
                             "Use 'auto' for auto-detect.")
    args = parser.parse_args()

    if args.no_ai:
        Config.ENABLE_AI_BRIEF = False
    Config.WHISPER_MODEL = args.whisper_model
    Config.WHISPER_LANG  = None if args.lang == "auto" else args.lang

    target = args.video or args.path
    if args.folder:
        batch_analyze(args.folder)
    elif target:
        if os.path.isdir(target):
            batch_analyze(target)
        else:
            analyze_video(target)
    else:
        VIDEO_PATH = r"E:\reel_19900likes_AQOWUlLtvNSdtIwKYntR.mp4"
        analyze_video(VIDEO_PATH)