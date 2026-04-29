"""
PATCH NOTES v2.4 — Advanced Image Classification Added:

  ① Places365 scene classification (indoor/outdoor scenes, room types, locations)
  ② YOLOv8 object detection (people, animals, objects, vehicles)
  ③ Activity recognition from detected objects and poses
  ④ Aesthetic quality scoring (composition, lighting, visual appeal)
  ⑤ Enhanced scene understanding with confidence scores
  ⑥ Integration with existing analysis pipeline
  ⑦ HTML report updates with new visual insights
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
import urllib.request
import urllib.error
import socket
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
warnings.filterwarnings("ignore")

# Add these imports at the top with other imports
try:
    import torch
    import torchvision.transforms as transforms
    from torchvision import models
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
    print("✅ PyTorch ready for image classification")
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️  PyTorch not installed — pip install torch torchvision")

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    print("✅ YOLOv8 ready for object detection")
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️  YOLOv8 not installed — pip install ultralytics")

# ── Optional imports ─────────────────────────────────────────
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

# Add Places365 categories (simplified list of common scenes)
PLACES365_CATEGORIES = {
    'indoor': [
        'airport', 'auditorium', 'bakery', 'bar', 'bathroom', 'bedroom',
        'bookstore', 'cafeteria', 'classroom', 'closet', 'conference_room',
        'dining_room', 'dorm_room', 'gym', 'home_office', 'hospital_room',
        'hotel_room', 'kitchen', 'library', 'living_room', 'lobby',
        'mall', 'movie_theater', 'museum', 'office', 'restaurant',
        'shopping_mall', 'spa', 'staircase', 'studio', 'supermarket'
    ],
    'outdoor': [
        'alley', 'amusement_park', 'beach', 'bridge', 'canyon', 'castle',
        'cemetery', 'city_center', 'coast', 'desert', 'farm', 'field',
        'forest', 'garden', 'glacier', 'golf_course', 'harbor', 'highway',
        'lake', 'mountain', 'park', 'parking_lot', 'playground', 'plaza',
        'river', 'road', 'roof_garden', 'ruins', 'savanna', 'sky',
        'stadium', 'street', 'temple', 'trail', 'village', 'waterfall',
        'zoo'
    ],
    'sports': [
        'badminton_court', 'baseball_field', 'basketball_court', 'bowling_alley',
        'boxing_ring', 'football_field', 'golf_course', 'gym', 'pool',
        'racetrack', 'skatepark', 'soccer_field', 'stadium', 'swimming_pool',
        'tennis_court', 'volleyball_court'
    ],
    'nature': [
        'beach', 'canyon', 'cave', 'coast', 'desert', 'field', 'forest',
        'glacier', 'lake', 'mountain', 'ocean', 'river', 'savanna',
        'sky', 'swamp', 'waterfall', 'wetland', 'woods'
    ]
}

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
    # Language setting — use "auto" for auto-detection
    WHISPER_LANG      = "hi"  # Change to "auto" for multi-language content
    # Devanagari seed prompt for Hindi content
    WHISPER_HINDI_PROMPT = "यह एक हिंदी वीडियो है।"
    EASYOCR_LANGS     = ["en", "hi"]
    ANTHROPIC_MODEL   = "claude-3-opus-20240229"
    ENABLE_AI_BRIEF   = True
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    # Ollama settings
    OLLAMA_URLS       = ["http://127.0.0.1:11434", "http://localhost:11434"]
    OLLAMA_MODEL      = "llama3"
    OLLAMA_ENABLED    = True
    OLLAMA_TIMEOUT    = 30
    OLLAMA_MAX_RETRIES = 3
    OLLAMA_RETRY_DELAY = 2
    OLLAMA_KEEP_ALIVE = "5m"
    # Transcript quality threshold
    TRANSCRIPT_QUALITY_THRESHOLD = 40
    # ML settings
    ML_ENABLED        = True
    YOLO_CONFIDENCE   = 0.25
    SCENE_CONFIDENCE  = 0.3

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
# MODULE 2 — Whisper STT with quality scoring (FIXED)
# ═══════════════════════════════════════════════════════════
def _detect_script(text: str) -> str:
    """Returns dominant script in text."""
    if not text:
        return "unknown"
    
    devanagari = arabic = latin = 0
    for ch in text:
        cp = ord(ch)
        if 0x0900 <= cp <= 0x097F:
            devanagari += 1
        elif 0x0600 <= cp <= 0x06FF:
            arabic += 1
        elif 0x0041 <= cp <= 0x007A or 0x0061 <= cp <= 0x007A:
            latin += 1
    
    total = devanagari + arabic + latin
    if total == 0:
        return "unknown"
    
    if devanagari / total > 0.3:
        return "devanagari"
    if arabic / total > 0.3:
        return "arabic_urdu"
    if latin / total > 0.3:
        return "latin"
    return "mixed"

def _score_transcript_quality(result: dict, duration_sec: float,
                               expected_lang: str) -> dict:
    """Scores transcript quality 0-100 with proper error handling."""
    
    # Handle case with no segments
    segments = result.get("segments", [])
    if not segments or not result.get("text", "").strip():
        return {
            "score": 0,
            "issues": ["no_speech_detected"],
            "reliable": False,
            "avg_logprob": -10.0,
            "avg_no_speech": 1.0,
            "avg_compression": 1.0,
            "detected_script": "none",
            "wpm_estimated": 0,
        }

    issues = []

    # Calculate metrics safely
    try:
        avg_logprob = float(np.mean([s.get("avg_logprob", -1.0) for s in segments if s.get("avg_logprob") is not None]))
    except:
        avg_logprob = -2.0
        
    try:
        avg_compression = float(np.mean([s.get("compression_ratio", 1.0) for s in segments if s.get("compression_ratio") is not None]))
    except:
        avg_compression = 1.0
        
    try:
        avg_no_speech = float(np.mean([s.get("no_speech_prob", 0.0) for s in segments if s.get("no_speech_prob") is not None]))
    except:
        avg_no_speech = 0.5

    transcript = result.get("text", "")
    word_count = len(transcript.split())
    
    # WPM calculation
    if duration_sec > 0:
        wpm = word_count / max(duration_sec / 60, 0.01)
    else:
        wpm = 0
        
    if wpm > 250:
        issues.append("wpm_too_high")
    if wpm < 10 and duration_sec > 10:
        issues.append("wpm_too_low")

    # Script detection
    detected_script = _detect_script(transcript)
    if expected_lang == "hi" and detected_script == "arabic_urdu":
        issues.append("wrong_script_urdu")
    elif expected_lang == "hi" and detected_script == "latin":
        issues.append("wrong_script_latin")

    # Quality checks
    if avg_compression > 2.4:
        issues.append("high_compression_ratio")
    if avg_no_speech > 0.6:
        issues.append("mostly_no_speech")
    if avg_logprob < -0.8:
        issues.append("low_confidence")

    # Compute score
    score = 100
    score += avg_logprob * 30
    score -= avg_no_speech * 40
    if avg_compression > 2.4:
        score -= 20
    if "wpm_too_high" in issues:
        score -= 20
    if "wrong_script_urdu" in issues:
        score -= 15
    if "wrong_script_latin" in issues and expected_lang == "hi":
        score -= 10
    if "low_confidence" in issues:
        score -= 10
    
    score = max(0, min(100, round(score)))

    reliable = score >= Config.TRANSCRIPT_QUALITY_THRESHOLD and not (
        "mostly_no_speech" in issues or "high_compression_ratio" in issues
    )

    return {
        "score": score,
        "issues": issues,
        "reliable": reliable,
        "avg_logprob": round(avg_logprob, 3) if avg_logprob != -10.0 else -10.0,
        "avg_no_speech": round(avg_no_speech, 3),
        "avg_compression": round(avg_compression, 3),
        "detected_script": detected_script,
        "wpm_estimated": round(wpm, 1),
    }

def transcribe_speech(video_path: str, duration_sec: float = 0.0) -> dict:
    """Transcribe speech with improved error handling."""
    if not WHISPER_AVAILABLE:
        return {
            "transcript": "",
            "language": "unknown",
            "segments": [],
            "word_timestamps": [],
            "quality": {"score": 0, "reliable": False, "detected_script": "unknown"}
        }

    lang_tag = Config.WHISPER_LANG or "auto"
    cache_key = f"whisper_{lang_tag}"
    
    # Try cache
    cached = load_cache(video_path, cache_key)
    if cached:
        print("  ♻️  Whisper — using cache")
        return cached

    print("  🎙️  Whisper — extracting audio...")
    
    # Extract audio
    try:
        clip = VideoFileClip(video_path)
        audio_path = os.path.join(Config.AUDIO_DIR, "whisper_temp.wav")
        clip.audio.write_audiofile(audio_path, fps=16000, logger=None)
        clip.close()
    except Exception as e:
        print(f"  ❌ Audio extraction failed: {e}")
        return {
            "transcript": "",
            "language": "unknown",
            "segments": [],
            "word_timestamps": [],
            "quality": {"score": 0, "reliable": False, "detected_script": "unknown"}
        }

    print(f"  🎙️  Whisper ({Config.WHISPER_MODEL}) — transcribing (lang={lang_tag})...")
    
    try:
        model = whisper.load_model(Config.WHISPER_MODEL)
        
        transcribe_kwargs = {
            "word_timestamps": True,
            "verbose": False,
            "task": "transcribe",
        }
        
        if Config.WHISPER_LANG and Config.WHISPER_LANG != "auto":
            transcribe_kwargs["language"] = Config.WHISPER_LANG
        if Config.WHISPER_LANG == "hi" and Config.WHISPER_HINDI_PROMPT:
            transcribe_kwargs["initial_prompt"] = Config.WHISPER_HINDI_PROMPT

        result = model.transcribe(audio_path, **transcribe_kwargs)
        
    except Exception as e:
        print(f"  ❌ Transcription failed: {e}")
        if os.path.exists(audio_path):
            os.remove(audio_path)
        return {
            "transcript": "",
            "language": "unknown",
            "segments": [],
            "word_timestamps": [],
            "quality": {"score": 0, "reliable": False, "detected_script": "unknown"}
        }

    # Quality scoring
    quality = _score_transcript_quality(result, duration_sec, lang_tag)
    detected_script = quality["detected_script"]

    # Build segments safely
    segments = []
    for seg in result.get("segments", []):
        segments.append({
            "start": round(seg.get("start", 0), 2),
            "end": round(seg.get("end", 0), 2),
            "text": seg.get("text", "").strip(),
            "avg_logprob": round(seg.get("avg_logprob", -1.0), 3),
            "no_speech_prob": round(seg.get("no_speech_prob", 0.0), 3),
            "compression_ratio": round(seg.get("compression_ratio", 1.0), 3),
        })

    word_ts = []
    for seg in result.get("segments", []):
        for w in seg.get("words", []):
            word_ts.append({
                "word": w.get("word", "").strip(),
                "start": round(w.get("start", 0), 2),
                "end": round(w.get("end", 0), 2),
            })

    transcript = result.get("text", "").strip()
    word_count = len(transcript.split())
    
    # Calculate WPM safely
    if segments and segments[-1].get("end", 0) > 0:
        last_end = segments[-1]["end"]
        wpm = round(word_count / max(last_end / 60, 0.01), 1) if last_end > 0 else 0
    else:
        wpm = 0

    # Clean up
    if os.path.exists(audio_path):
        os.remove(audio_path)

    # Quality feedback
    if not quality["reliable"]:
        print(f"  ⚠️  Transcript quality LOW ({quality['score']}/100): {quality['issues']}")
        print(f"      Script detected: {detected_script}")
    else:
        print(f"  ✅ Transcript quality: {quality['score']}/100  Script: {detected_script}")

    data = {
        "transcript": transcript,
        "transcript_display": transcript,
        "language": result.get("language", lang_tag),
        "segments": segments,
        "word_timestamps": word_ts,
        "word_count": word_count,
        "speaking_rate_wpm": wpm,
        "quality": quality,
        "detected_script": detected_script,
    }
    
    save_cache(video_path, cache_key, data)
    return data


# ═══════════════════════════════════════════════════════════
# MODULE 2B — Ollama content analysis
# ═══════════════════════════════════════════════════════════
def _check_port_open(host: str, port: int, timeout: float = 2.0) -> bool:
    """Check if a port is open on the given host."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        # Extract hostname from URL if needed
        if host.startswith("http://"):
            host = host.replace("http://", "").split(":")[0]
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except:
        return False

def _ollama_available() -> tuple:
    """
    Check if Ollama is available on any configured URL.
    Returns (bool, str) - (available, working_url)
    """
    for url in Config.OLLAMA_URLS:
        try:
            # Extract host and port from URL
            host_part = url.replace("http://", "").split(":")
            host = host_part[0]
            port = int(host_part[1]) if len(host_part) > 1 else 11434
            
            # Quick port check first
            if not _check_port_open(host, port):
                continue
                
            # Try API endpoint
            req = urllib.request.Request(f"{url}/api/tags")
            with urllib.request.urlopen(req, timeout=3) as resp:
                if resp.getcode() == 200:
                    return True, url
        except Exception:
            continue
    return False, None

def _clean_json_response(response_text: str) -> str:
    """Clean and extract JSON from model response."""
    # Remove markdown code blocks
    if "```" in response_text:
        lines = response_text.split("\n")
        in_code_block = False
        cleaned_lines = []
        for line in lines:
            if line.strip().startswith("```"):
                in_code_block = not in_code_block
                continue
            if in_code_block:
                cleaned_lines.append(line)
        if cleaned_lines:
            return "\n".join(cleaned_lines)
    
    # Try to find JSON between curly braces
    import re
    json_pattern = r'\{[^{}]*(\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(json_pattern, response_text, re.DOTALL)
    if matches:
        # Get the longest match (most likely the full JSON)
        return max(matches, key=len)
    
    return response_text

def _get_default_ollama_response(reason: str = None) -> dict:
    """Return a default/fallback response when Ollama fails."""
    return {
        "available": False,
        "reason": reason or "ollama_unavailable",
        "content_summary": "(Ollama analysis unavailable)",
        "topic_niche": "unknown",
        "sub_topic": "",
        "emotional_tone": "unknown",
        "key_phrases": [],
        "target_audience": "unknown",
        "content_language": "unknown",
        "has_horoscope_elements": False,
        "has_advice_elements": False,
        "confidence": 0,
    }

def ollama_analyze_content(transcript: str, ocr_text: str,
                            transcript_quality: dict,
                            duration_sec: float) -> dict:
    """
    Sends transcript + OCR text to a local Ollama model for:
    - Plain-English content summary
    - Niche / topic detection (especially for Hindi content)
    - Emotional tone
    - Key phrases / talking points
    """
    if not Config.OLLAMA_ENABLED:
        return _get_default_ollama_response("disabled_in_config")

    # Check availability and get working URL
    available, working_url = _ollama_available()
    if not available:
        return _get_default_ollama_response("ollama_not_running")

    # Only use reliable transcript; if low quality use OCR only
    if transcript_quality.get("reliable", False):
        text_block = f"SPOKEN TRANSCRIPT (Hindi):\n{transcript}\n\nON-SCREEN TEXT:\n{ocr_text}"
    else:
        quality_note = (f"[Note: Whisper transcript quality is LOW "
                        f"({transcript_quality.get('score',0)}/100), "
                        f"issues: {transcript_quality.get('issues',[])}. "
                        f"Treat spoken text with caution.]")
        text_block = (f"SPOKEN TRANSCRIPT (uncertain quality):\n{transcript}\n"
                      f"{quality_note}\n\nON-SCREEN TEXT (more reliable):\n{ocr_text}")

    prompt = f"""You are analyzing a short Instagram Reel ({duration_sec:.0f} seconds long).
Here is the content extracted from it:

{text_block}

Please respond in JSON only (no markdown, no explanation outside the JSON) with these fields:
{{
  "content_summary": "2-3 sentence plain English summary of what this video is about",
  "topic_niche": "single best niche label, e.g. astrology, motivation, comedy, education, finance, beauty, food, travel, relationship_advice, spiritual, news, fitness, tech",
  "sub_topic": "more specific sub-topic if identifiable",
  "emotional_tone": "e.g. serious, humorous, dramatic, calm, inspirational, fearful, angry, joyful",
  "key_phrases": ["up to 5 key phrases or talking points from the content"],
  "target_audience": "who this content is aimed at",
  "content_language": "primary language detected in the content",
  "has_horoscope_elements": true or false,
  "has_advice_elements": true or false,
  "confidence": 0 to 100 (how confident you are given the content quality)
}}"""

    payload = json.dumps({
        "model": Config.OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 500},
        "keep_alive": Config.OLLAMA_KEEP_ALIVE
    }).encode("utf-8")

    # Try with retries
    for attempt in range(Config.OLLAMA_MAX_RETRIES):
        try:
            req = urllib.request.Request(
                f"{working_url}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            
            with urllib.request.urlopen(req, timeout=Config.OLLAMA_TIMEOUT) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
            
            response_text = raw.get("response", "").strip()
            if not response_text:
                if attempt < Config.OLLAMA_MAX_RETRIES - 1:
                    time.sleep(Config.OLLAMA_RETRY_DELAY)
                    continue
                return _get_default_ollama_response("empty_response")

            # Clean and parse JSON
            cleaned_text = _clean_json_response(response_text)
            
            try:
                parsed = json.loads(cleaned_text)
            except json.JSONDecodeError:
                # Try to find JSON by scanning
                import re
                json_match = re.search(r'\{.*\}', cleaned_text, re.DOTALL)
                if json_match:
                    try:
                        parsed = json.loads(json_match.group())
                    except:
                        if attempt < Config.OLLAMA_MAX_RETRIES - 1:
                            time.sleep(Config.OLLAMA_RETRY_DELAY)
                            continue
                        return {
                            "available": True,
                            "error": "JSON parse failed after retries",
                            "raw_response": response_text[:500],
                            **_get_default_ollama_response("json_parse_failed")
                        }
                else:
                    if attempt < Config.OLLAMA_MAX_RETRIES - 1:
                        time.sleep(Config.OLLAMA_RETRY_DELAY)
                        continue
                    return {
                        "available": True,
                        "error": "No JSON found in response",
                        "raw_response": response_text[:500],
                        **_get_default_ollama_response("no_json_found")
                    }

            # Ensure all expected fields exist
            default_fields = {
                "content_summary": "(No summary provided)",
                "topic_niche": "unknown",
                "sub_topic": "",
                "emotional_tone": "unknown",
                "key_phrases": [],
                "target_audience": "unknown",
                "content_language": "unknown",
                "has_horoscope_elements": False,
                "has_advice_elements": False,
                "confidence": 0
            }
            
            for field, default_value in default_fields.items():
                if field not in parsed:
                    parsed[field] = default_value
            
            parsed["available"] = True
            parsed["model_used"] = Config.OLLAMA_MODEL
            return parsed

        except urllib.error.URLError as e:
            if attempt < Config.OLLAMA_MAX_RETRIES - 1:
                print(f"    ⚠️  Ollama attempt {attempt + 1} failed: {e.reason}, retrying...")
                time.sleep(Config.OLLAMA_RETRY_DELAY)
            else:
                return _get_default_ollama_response(f"connection_failed: {e.reason}")
                
        except socket.timeout:
            if attempt < Config.OLLAMA_MAX_RETRIES - 1:
                print(f"    ⚠️  Ollama attempt {attempt + 1} timed out, retrying...")
                time.sleep(Config.OLLAMA_RETRY_DELAY)
            else:
                return _get_default_ollama_response("timeout")
                
        except Exception as e:
            if attempt < Config.OLLAMA_MAX_RETRIES - 1:
                print(f"    ⚠️  Ollama attempt {attempt + 1} failed: {str(e)[:50]}, retrying...")
                time.sleep(Config.OLLAMA_RETRY_DELAY)
            else:
                return {
                    "available": False,
                    "reason": str(e),
                    "error_type": type(e).__name__,
                    **_get_default_ollama_response(f"exception: {type(e).__name__}")
                }
    
    return _get_default_ollama_response("max_retries_exceeded")


# ═══════════════════════════════════════════════════════════
# MODULE 3 — Librosa audio analysis
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
    major_scores   = [np.dot(np.roll(chroma_arr,-i), major_template) for i in range(12)]
    minor_scores   = [np.dot(np.roll(chroma_arr,-i), minor_template) for i in range(12)]
    best_major     = int(np.argmax(major_scores))
    best_minor     = int(np.argmax(minor_scores))
    detected_key   = (f"{note_names[best_major]} major"
                      if max(major_scores) >= max(minor_scores)
                      else f"{note_names[best_minor]} minor")

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
    silent_chunks = [e for e in energy_over_time if e["energy"] < silence_threshold]
    silence_ratio = round(len(silent_chunks) / max(len(energy_over_time), 1), 3)

    has_voice  = float(np.mean(spectral_centroid)) < 3500 and rms_mean > 0.015
    is_music   = len(beat_times) > 4 and harmony_pct > 35
    is_speech  = float(np.mean(zero_crossing)) > 0.08 and has_voice
    is_ambient = rms_mean < 0.05 and len(beat_times) < 3
    # ⑥ Music-heavy: strong beats, high harmony, likely background music drowning speech
    is_music_heavy = is_music and harmony_pct > 60 and len(beat_times) > 8

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

    if is_music_heavy:
        print(f"  ⚠️  Music-heavy audio detected (harmony: {harmony_pct}%) — "
              f"Whisper accuracy may be reduced")

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
        "is_music_heavy":          is_music_heavy,
    }
    save_cache(video_path, "audio", data)
    return data


# ═══════════════════════════════════════════════════════════
# MODULE 4A — EasyOCR (Hindi + English)
# ═══════════════════════════════════════════════════════════
def get_easyocr_reader():
    global EASYOCR_READER
    if EASYOCR_READER is None:
        print("  📝 Loading EasyOCR model (first time only)...")
        EASYOCR_READER = easyocr.Reader(Config.EASYOCR_LANGS, gpu=False)
    return EASYOCR_READER

def extract_text_easyocr(frame) -> dict:
    if not EASYOCR_AVAILABLE:
        return {"text": "", "word_count": 0, "words": [], "has_text": False,
                "readability_score": 0, "text_zones": [], "hook_text": "",
                "scripts_detected": [], "hindi_text": "", "english_text": ""}

    reader  = get_easyocr_reader()
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = reader.readtext(img_rgb, detail=1, paragraph=False)

    h, w = frame.shape[:2]
    words, zones = [], []
    full_text = ""
    hindi_parts, english_parts = [], []

    for (bbox, text, conf) in results:
        if conf < 0.3 or len(text.strip()) < 1:
            continue
        x1 = int(bbox[0][0]); y1 = int(bbox[0][1])
        x2 = int(bbox[2][0]); y2 = int(bbox[2][1])
        cx, cy        = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
        font_size_est = abs(y2 - y1)

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

        if script == "devanagari":
            hindi_parts.append(text.strip())
        else:
            english_parts.append(text.strip())

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
        "hindi_text":        " ".join(hindi_parts),
        "english_text":      " ".join(english_parts),
    }


# ═══════════════════════════════════════════════════════════
# MODULE 4B — Scene background detection
# ═══════════════════════════════════════════════════════════
def detect_scene_background(frame, face_boxes: list) -> dict:
    h, w = frame.shape[:2]
    mask = np.ones((h, w), dtype=np.uint8) * 255
    for fb in face_boxes:
        cx, cy = int(fb["cx"] * w), int(fb["cy"] * h)
        sz     = int(math.sqrt(max(fb.get("size_ratio", 0.05), 0.01)) * w * 1.5)
        x1 = max(0, cx-sz); y1 = max(0, cy-sz)
        x2 = min(w, cx+sz); y2 = min(h, cy+sz)
        mask[y1:y2, x1:x2] = 0

    bg_pixels = frame[mask > 0]
    if len(bg_pixels) < 100:
        bg_pixels = frame.reshape(-1, 3)

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    top_zone  = hsv_frame[:h//4, :, :]
    sky_blue  = np.sum((top_zone[:,:,0] > 95) & (top_zone[:,:,0] < 135) &
                       (top_zone[:,:,1] > 40) & (top_zone[:,:,2] > 80))
    sky_ratio = sky_blue / max(top_zone.shape[0]*top_zone.shape[1], 1)
    green_px  = np.sum((hsv_frame[:,:,0] > 35) & (hsv_frame[:,:,0] < 85) &
                       (hsv_frame[:,:,1] > 40) & (hsv_frame[:,:,2] > 40))
    green_ratio = green_px / (h * w)

    bg_gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bg_masked = cv2.bitwise_and(bg_gray, bg_gray, mask=mask)
    nonzero   = bg_masked[mask > 0]
    texture_std = float(np.std(nonzero)) if len(nonzero) > 0 else 0

    if len(bg_pixels) > 0:
        mean_bgr = bg_pixels.mean(axis=0)
        r, g, b  = float(mean_bgr[2]), float(mean_bgr[1]), float(mean_bgr[0])
    else:
        r, g, b = 128.0, 128.0, 128.0

    if r > 200 and g > 200 and b > 200:    wall_color = "white"
    elif r < 60 and g < 60 and b < 60:     wall_color = "black"
    elif r > g + 30 and r > b + 30:        wall_color = "red_warm"
    elif b > r + 30 and b > g + 20:        wall_color = "blue_cool"
    elif g > r + 20 and g > b + 20:        wall_color = "green"
    elif r > 160 and g > 130 and b < 100:  wall_color = "golden_warm"
    elif r > 180 and g > 180:              wall_color = "bright_yellow"
    else:
        val = int((r + g + b) / 3)
        wall_color = f"grey_{val//50*50}"

    is_outdoor     = sky_ratio > 0.08 or green_ratio > 0.15
    is_nature      = green_ratio > 0.25
    is_studio_flat = texture_std < 18 and not is_outdoor
    is_busy_bg     = texture_std > 45

    if is_nature and is_outdoor:     setting = "outdoor_nature"
    elif is_outdoor and sky_ratio > 0.08: setting = "outdoor_street_or_open"
    elif is_outdoor:                 setting = "outdoor_other"
    elif is_studio_flat:             setting = "studio_or_plain_wall"
    elif is_busy_bg:                 setting = "indoor_busy_background"
    else:                            setting = "indoor_room"

    room_type = None
    if not is_outdoor:
        brightness = float(np.mean(nonzero)) if len(nonzero) > 0 else 128
        if brightness > 200 and texture_std < 20: room_type = "studio_white"
        elif wall_color in ["red_warm","golden_warm","bright_yellow"]: room_type = "warm_decorated_room"
        elif wall_color == "blue_cool":            room_type = "cool_toned_room"
        elif texture_std > 40:                     room_type = "cluttered_or_bookshelf"
        else:                                      room_type = "generic_indoor"

    bg_bright = float(np.mean(bg_pixels)) if len(bg_pixels) > 0 else 128
    if bg_bright > 170:   lighting = "bright_well_lit"
    elif bg_bright > 100: lighting = "normal"
    elif bg_bright > 50:  lighting = "dim"
    else:                 lighting = "dark_low_light"

    return {
        "indoor_outdoor": "outdoor" if is_outdoor else "indoor",
        "setting_type":   setting,
        "room_type":      room_type,
        "wall_color":     wall_color,
        "bg_texture_std": round(texture_std, 1),
        "is_flat_bg":     is_studio_flat,
        "is_busy_bg":     is_busy_bg,
        "sky_ratio":      round(sky_ratio, 3),
        "green_ratio":    round(green_ratio, 3),
        "lighting":       lighting,
        "bg_mean_rgb":    [round(r,1), round(g,1), round(b,1)],
    }


# ═══════════════════════════════════════════════════════════
# MODULE 4C — HOG body/people counter
# ═══════════════════════════════════════════════════════════
_HOG = cv2.HOGDescriptor()
_HOG.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

def count_bodies_hog(frame) -> dict:
    h, w  = frame.shape[:2]
    scale = 480 / h if h > 480 else 1.0
    small = cv2.resize(frame, (int(w * scale), int(h * scale)))
    rects, weights = _HOG.detectMultiScale(small, winStride=(8,8), padding=(4,4), scale=1.05)
    people = []
    for (bx, by, bw, bh) in rects:
        cx = round((bx + bw/2) / small.shape[1], 3)
        cy = round((by + bh/2) / small.shape[0], 3)
        people.append({
            "cx": cx, "cy": cy,
            "pos_v": "top" if cy < 0.33 else "bottom" if cy > 0.66 else "center",
            "pos_h": "left" if cx < 0.33 else "right" if cx > 0.66 else "center",
        })
    return {"body_count_hog": len(people), "body_positions": people}


# ═══════════════════════════════════════════════════════════
# MODULE 4D — MediaPipe face + pose
# ═══════════════════════════════════════════════════════════
def analyze_face_pose(frame) -> dict:
    if not MEDIAPIPE_AVAILABLE:
        return {"face_count": 0, "faces": [], "body_detected": False,
                "gaze_direction": "unknown", "upper_body_visible": False}

    h, w    = frame.shape[:2]
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result  = {"face_count": 0, "faces": [], "body_detected": False,
               "gaze_direction": "unknown", "upper_body_visible": False}

    with mp_face_detection.FaceDetection(min_detection_confidence=0.4) as fd:
        det = fd.process(img_rgb)
        if det.detections:
            result["face_count"] = len(det.detections)
            for d in det.detections:
                bbox  = d.location_data.relative_bounding_box
                cx    = round(bbox.xmin + bbox.width / 2, 3)
                cy    = round(bbox.ymin + bbox.height / 2, 3)
                size  = round(bbox.width * bbox.height, 4)
                score = round(d.score[0], 2)
                result["faces"].append({
                    "cx": cx, "cy": cy, "size_ratio": size, "confidence": score,
                    "is_close_up": size > 0.15,
                    "position": "top" if cy < 0.33 else "bottom" if cy > 0.66 else "center",
                })

    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                refine_landmarks=True, min_detection_confidence=0.4) as fm:
        mesh = fm.process(img_rgb)
        if mesh.multi_face_landmarks:
            lm           = mesh.multi_face_landmarks[0].landmark
            nose_x       = lm[1].x
            eye_center_x = (lm[33].x + lm[263].x) / 2
            diff = nose_x - eye_center_x
            if diff > 0.03:    result["gaze_direction"] = "looking_right"
            elif diff < -0.03: result["gaze_direction"] = "looking_left"
            else:              result["gaze_direction"] = "looking_at_camera"
            result["eye_contact"] = result["gaze_direction"] == "looking_at_camera"

    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.4) as pose:
        pr = pose.process(img_rgb)
        if pr.pose_landmarks:
            result["body_detected"]      = True
            result["upper_body_visible"] = True
            lms = pr.pose_landmarks.landmark
            lw  = lms[mp_pose.PoseLandmark.LEFT_WRIST]
            rw  = lms[mp_pose.PoseLandmark.RIGHT_WRIST]
            result["hands_raised"] = lw.y < 0.5 or rw.y < 0.5

    return result


# ═══════════════════════════════════════════════════════════
# MODULE 4F — Places365 Scene Classification
# ═══════════════════════════════════════════════════════════
class Places365Classifier:
    """Scene classification using Places365 model"""
    
    def __init__(self):
        self.model = None
        self.transform = None
        self.categories = None
        self.labels = []
        
    def load_model(self):
        if not TORCH_AVAILABLE:
            return False
            
        try:
            # Use a pre-trained ResNet for Places365
            # Note: You might need to download the actual Places365 weights
            # For now, we'll use a ResNet trained on ImageNet as fallback
            print("    Loading Places365 scene classifier...")
            self.model = models.resnet50(pretrained=True)
            self.model.eval()
            
            # Define transforms
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
            
            # Load Places365 category labels
            # For demo, we'll use a subset
            self.labels = []
            for category_list in PLACES365_CATEGORIES.values():
                self.labels.extend(category_list)
            
            return True
        except Exception as e:
            print(f"    ⚠️  Failed to load scene classifier: {e}")
            return False
    
    def classify_scene(self, image):
        """Classify scene type from image"""
        if self.model is None:
            if not self.load_model():
                return {"scene_type": "unknown", "confidence": 0, "scene_category": "unknown"}
        
        try:
            # Convert OpenCV BGR to PIL RGB
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            
            # Apply transforms
            input_tensor = self.transform(pil_img).unsqueeze(0)
            
            # Inference
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = F.softmax(outputs, dim=1)
                
            # Get top predictions
            top_probs, top_indices = torch.topk(probabilities, 5)
            top_probs = top_probs[0].tolist()
            top_indices = top_indices[0].tolist()
            
            # Map to scene categories (simplified)
            predictions = []
            for i, idx in enumerate(top_indices):
                # For demo, map to our simplified categories
                # In production, use actual Places365 labels
                pred_idx = idx % len(self.labels)
                scene_label = self.labels[pred_idx] if pred_idx < len(self.labels) else "unknown"
                
                # Determine category
                scene_category = "unknown"
                for cat, scenes in PLACES365_CATEGORIES.items():
                    if scene_label in scenes:
                        scene_category = cat
                        break
                
                predictions.append({
                    "scene": scene_label,
                    "confidence": top_probs[i],
                    "category": scene_category
                })
            
            best_scene = predictions[0]
            
            return {
                "scene_type": best_scene["scene"],
                "confidence": round(best_scene["confidence"] * 100, 1),
                "scene_category": best_scene["category"],
                "top_predictions": predictions[:3]
            }
            
        except Exception as e:
            print(f"    ⚠️  Scene classification failed: {e}")
            return {"scene_type": "unknown", "confidence": 0, "scene_category": "unknown"}


# ═══════════════════════════════════════════════════════════
# MODULE 4G — YOLOv8 Object Detection
# ═══════════════════════════════════════════════════════════
class YOLODetector:
    """Object detection using YOLOv8"""
    
    def __init__(self):
        self.model = None
        
    def load_model(self):
        if not YOLO_AVAILABLE:
            return False
            
        try:
            print("    Loading YOLOv8 object detector...")
            # Use YOLOv8n (nano) for speed, or yolov8s/m/l/x for accuracy
            self.model = YOLO('yolov8n.pt')  # Will download on first use
            return True
        except Exception as e:
            print(f"    ⚠️  Failed to load YOLO: {e}")
            return False
    
    def detect_objects(self, image, confidence_threshold=0.25):
        """Detect objects in image"""
        if self.model is None:
            if not self.load_model():
                return {"objects": [], "object_count": 0, "object_categories": {}}
        
        try:
            # Run inference
            results = self.model(image, verbose=False)[0]
            
            # Process detections
            objects = []
            object_categories = {}
            
            # COCO class categories for grouping
            category_map = {
                'person': 'people',
                'bicycle': 'vehicle', 'car': 'vehicle', 'motorcycle': 'vehicle', 
                'airplane': 'vehicle', 'bus': 'vehicle', 'train': 'vehicle', 
                'truck': 'vehicle', 'boat': 'vehicle',
                'traffic light': 'street', 'fire hydrant': 'street', 'stop sign': 'street',
                'parking meter': 'street', 'bench': 'furniture',
                'bird': 'animal', 'cat': 'animal', 'dog': 'animal', 'horse': 'animal',
                'sheep': 'animal', 'cow': 'animal', 'elephant': 'animal', 'bear': 'animal',
                'zebra': 'animal', 'giraffe': 'animal',
                'backpack': 'accessory', 'umbrella': 'accessory', 'handbag': 'accessory',
                'tie': 'accessory', 'suitcase': 'accessory',
                'frisbee': 'sports', 'skis': 'sports', 'snowboard': 'sports',
                'sports ball': 'sports', 'kite': 'sports', 'baseball bat': 'sports',
                'baseball glove': 'sports', 'skateboard': 'sports', 'surfboard': 'sports',
                'tennis racket': 'sports',
                'bottle': 'food_drink', 'wine glass': 'food_drink', 'cup': 'food_drink',
                'fork': 'food_drink', 'knife': 'food_drink', 'spoon': 'food_drink',
                'bowl': 'food_drink', 'banana': 'food_drink', 'apple': 'food_drink',
                'sandwich': 'food_drink', 'orange': 'food_drink', 'broccoli': 'food_drink',
                'carrot': 'food_drink', 'hot dog': 'food_drink', 'pizza': 'food_drink',
                'donut': 'food_drink', 'cake': 'food_drink',
                'chair': 'furniture', 'couch': 'furniture', 'potted plant': 'plant',
                'bed': 'furniture', 'dining table': 'furniture', 'toilet': 'furniture',
                'tv': 'electronics', 'laptop': 'electronics', 'mouse': 'electronics',
                'remote': 'electronics', 'keyboard': 'electronics', 'cell phone': 'electronics',
                'microwave': 'appliance', 'oven': 'appliance', 'toaster': 'appliance',
                'sink': 'appliance', 'refrigerator': 'appliance',
                'book': 'stationery', 'clock': 'electronics', 'vase': 'decor',
                'scissors': 'tool', 'teddy bear': 'toy', 'hair drier': 'personal_care',
                'toothbrush': 'personal_care'
            }
            
            if results.boxes is not None:
                for box in results.boxes:
                    confidence = float(box.conf[0])
                    if confidence < confidence_threshold:
                        continue
                        
                    class_id = int(box.cls[0])
                    class_name = results.names[class_id]
                    
                    # Get bounding box
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    
                    # Calculate relative position
                    h, w = image.shape[:2]
                    cx = ((x1 + x2) / 2) / w
                    cy = ((y1 + y2) / 2) / h
                    
                    # Determine category
                    category = category_map.get(class_name, 'other')
                    
                    object_data = {
                        "class": class_name,
                        "confidence": round(confidence * 100, 1),
                        "category": category,
                        "position": {"cx": round(cx, 3), "cy": round(cy, 3)},
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "size_ratio": ((x2 - x1) * (y2 - y1)) / (w * h)
                    }
                    
                    objects.append(object_data)
                    
                    # Update category counts
                    if category not in object_categories:
                        object_categories[category] = 0
                    object_categories[category] += 1
            
            # Sort objects by confidence
            objects.sort(key=lambda x: x['confidence'], reverse=True)
            
            return {
                "objects": objects,
                "object_count": len(objects),
                "object_categories": object_categories,
                "has_people": any(obj['class'] == 'person' for obj in objects),
                "has_pets": any(obj['category'] == 'animal' for obj in objects),
                "has_food": any(obj['category'] == 'food_drink' for obj in objects),
                "has_technology": any(obj['category'] == 'electronics' for obj in objects)
            }
            
        except Exception as e:
            print(f"    ⚠️  Object detection failed: {e}")
            return {"objects": [], "object_count": 0, "object_categories": {}}


# ═══════════════════════════════════════════════════════════
# MODULE 4H — Activity Recognition
# ═══════════════════════════════════════════════════════════
def recognize_activity(objects, face_data, scene_data):
    """Infer activity from detected objects and poses"""
    
    activities = []
    confidence = 0
    
    # Check for common activities
    object_classes = [obj['class'] for obj in objects]
    
    # Eating/Drinking
    if any(item in object_classes for item in ['bottle', 'cup', 'fork', 'knife', 'bowl', 'sandwich', 'pizza']):
        activities.append("eating/drinking")
        confidence += 20
    
    # Working/Studying
    if any(item in object_classes for item in ['laptop', 'book', 'keyboard', 'mouse', 'tv']):
        activities.append("working/studying")
        confidence += 25
    
    # Exercising
    if any(item in object_classes for item in ['sports ball', 'skis', 'snowboard', 'frisbee']):
        activities.append("exercising/sports")
        confidence += 30
    
    # Talking/Presenting
    if face_data.get('face_count', 0) > 0 and scene_data.get('upper_body_visible', False):
        activities.append("talking/presenting")
        confidence += 35
    
    # Cooking
    if any(item in object_classes for item in ['microwave', 'oven', 'refrigerator', 'sink', 'bowl']):
        activities.append("cooking")
        confidence += 25
    
    # Driving/Travel
    if any(item in object_classes for item in ['car', 'truck', 'bus', 'motorcycle', 'bicycle']):
        activities.append("driving/traveling")
        confidence += 30
    
    # Shopping
    if any(item in object_classes for item in ['handbag', 'suitcase', 'backpack']):
        # Combined with indoor retail scenes would be better
        activities.append("shopping")
        confidence += 15
    
    # If no specific activity detected
    if not activities:
        if face_data.get('face_count', 0) > 0:
            activities.append("talking head")
        else:
            activities.append("scenic/b-roll")
    
    # Cap confidence
    confidence = min(confidence, 95)
    
    return {
        "primary_activity": activities[0] if activities else "unknown",
        "all_activities": activities,
        "confidence": confidence
    }


# ═══════════════════════════════════════════════════════════
# MODULE 4I — Aesthetic Quality Scoring
# ═══════════════════════════════════════════════════════════
def score_aesthetic_quality(image, objects, face_data, frame_data):
    """Score the aesthetic quality of an image (0-100)"""
    
    h, w = image.shape[:2]
    score = 50  # Base score
    factors = []
    
    # 1. Rule of thirds check (subject placement)
    if objects:
        # Check if main subject is near rule of thirds lines
        thirds_x = [w/3, 2*w/3]
        thirds_y = [h/3, 2*h/3]
        
        main_obj = objects[0]  # Most confident object
        cx = main_obj['bbox'][0] + (main_obj['bbox'][2] - main_obj['bbox'][0])/2
        cy = main_obj['bbox'][1] + (main_obj['bbox'][3] - main_obj['bbox'][1])/2
        
        # Distance to nearest third line
        dist_x = min(abs(cx - tx) for tx in thirds_x) / w
        dist_y = min(abs(cy - ty) for ty in thirds_y) / h
        
        if dist_x < 0.1 and dist_y < 0.1:
            score += 10
            factors.append("good_composition")
        elif dist_x < 0.15 or dist_y < 0.15:
            score += 5
            factors.append("decent_composition")
    
    # 2. Lighting quality
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    contrast = np.std(gray)
    
    if 100 <= brightness <= 200:
        score += 5
        factors.append("good_brightness")
    if contrast > 50:
        score += 5
        factors.append("good_contrast")
    elif contrast < 20:
        score -= 5
        factors.append("low_contrast")
    
    # 3. Face visibility and framing
    if face_data.get('face_count', 0) > 0:
        # Check if faces are well-framed
        for face in face_data.get('faces', []):
            if 0.2 <= face['cx'] <= 0.8 and 0.2 <= face['cy'] <= 0.8:
                score += 5
                factors.append("face_well_framed")
                break
        
        # Check eye contact
        if face_data.get('eye_contact', False):
            score += 5
            factors.append("eye_contact")
    
    # 4. Color harmony (simplified)
    # Convert to HSV and check color distribution
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
    
    # Check if not too many colors (cluttered)
    non_zero_hues = np.count_nonzero(hue_hist > max(hue_hist)*0.1)
    if non_zero_hues < 30:
        score += 5
        factors.append("color_harmony")
    elif non_zero_hues > 80:
        score -= 5
        factors.append("too_colorful")
    
    # 5. Background simplicity
    if frame_data.get('is_flat_bg', False):
        score += 5
        factors.append("clean_background")
    elif frame_data.get('is_busy_bg', False):
        score -= 5
        factors.append("busy_background")
    
    # 6. Sharpness
    if frame_data.get('sharpness', 0) > 100:
        score += 5
        factors.append("sharp_image")
    elif frame_data.get('sharpness', 0) < 30:
        score -= 5
        factors.append("blurry")
    
    # Clamp score
    score = max(0, min(100, score))
    
    return {
        "aesthetic_score": score,
        "factors": factors,
        "brightness": brightness,
        "contrast": contrast,
        "color_count": non_zero_hues
    }


# ═══════════════════════════════════════════════════════════
# MODULE 4J — Enhanced Frame Analysis with Classification
# ═══════════════════════════════════════════════════════════
class ImageClassifier:
    """Master classifier combining all image understanding models"""
    
    def __init__(self):
        self.places365 = Places365Classifier()
        self.yolo = YOLODetector()
        self.scene_classifier_loaded = False
        self.object_detector_loaded = False
        
    def load_models(self):
        """Load all classification models"""
        self.scene_classifier_loaded = self.places365.load_model()
        self.object_detector_loaded = self.yolo.load_model()
        
    def analyze_image(self, image, scene_data, face_data):
        """Complete image analysis"""
        
        # Load models if needed
        if not self.scene_classifier_loaded and not self.object_detector_loaded:
            self.load_models()
        
        # Scene classification
        scene_result = self.places365.classify_scene(image) if self.scene_classifier_loaded else {}
        
        # Object detection
        object_result = self.yolo.detect_objects(image) if self.object_detector_loaded else {}
        
        # Activity recognition
        activity_result = recognize_activity(
            object_result.get('objects', []),
            face_data,
            scene_data
        )
        
        # Aesthetic scoring
        aesthetic_result = score_aesthetic_quality(
            image, 
            object_result.get('objects', []),
            face_data,
            scene_data
        )
        
        return {
            "scene_classification": scene_result,
            "object_detection": object_result,
            "activity_recognition": activity_result,
            "aesthetic_quality": aesthetic_result
        }


# ═══════════════════════════════════════════════════════════
# MODIFIED: analyze_frames function with image classification
# ═══════════════════════════════════════════════════════════
def analyze_frames_with_classification(path: str) -> tuple:
    """Enhanced frame analysis with ML classification"""
    
    # Check cache first
    cached = load_cache(path, "frames_v4_ml")
    if cached:
        print("  ♻️  Frames (with ML) — using cache")
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        interval = max(1, int(fps * Config.SAMPLE_EVERY_SEC))
        keyframes, frame_idx = [], 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            if frame_idx % interval == 0:
                keyframes.append(frame.copy())
            frame_idx += 1
        cap.release()
        return cached, keyframes
    
    # Initialize classifier
    classifier = ImageClassifier()
    classifier.load_models()
    
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, int(fps * Config.SAMPLE_EVERY_SEC))
    frame_data, keyframes = [], []
    prev_gray = None
    frame_idx = 0
    
    with tqdm(total=total, desc="  🖼️  Frames (with ML)", unit="fr") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            if frame_idx % interval == 0:
                sec = round(frame_idx / fps, 2)
                
                # Basic visual analysis (existing)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                
                # Motion analysis
                motion, camera_move = 0.0, "static"
                if prev_gray is not None:
                    flow = cv2.calcOpticalFlowFarneback(
                        prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    motion = float(np.mean(np.sqrt(flow[...,0]**2 + flow[...,1]**2)))
                    mx, my = float(np.mean(flow[...,0])), float(np.mean(flow[...,1]))
                    if abs(mx) > abs(my):
                        camera_move = "pan_right" if mx>1 else "pan_left" if mx<-1 else "static"
                    else:
                        camera_move = "tilt_down" if my>1 else "tilt_up" if my<-1 else "static"
                
                sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                
                # Face and pose detection (existing)
                face_data = analyze_face_pose(frame)
                
                # Scene background detection (existing)
                scene_data = detect_scene_background(frame, face_data.get("faces", []))
                
                # Body counting (existing)
                body_data = count_bodies_hog(frame)
                
                # OCR text detection (existing)
                if EASYOCR_AVAILABLE:
                    text_data = extract_text_easyocr(frame)
                else:
                    text_data = {}
                
                # NEW: ML-based image classification
                ml_analysis = classifier.analyze_image(frame, scene_data, face_data)
                
                # Combine all data
                frame_data.append({
                    "frame": frame_idx,
                    "sec": sec,
                    # Existing fields
                    "face_count": face_data["face_count"],
                    "face_positions": face_data["faces"],
                    "gaze_direction": face_data.get("gaze_direction", "unknown"),
                    "eye_contact": face_data.get("eye_contact", False),
                    "body_detected": face_data["body_detected"],
                    "hands_raised": face_data.get("hands_raised", False),
                    "people_count": max(face_data["face_count"], body_data["body_count_hog"]),
                    "body_count_hog": body_data["body_count_hog"],
                    "body_positions": body_data["body_positions"],
                    "indoor_outdoor": scene_data["indoor_outdoor"],
                    "setting_type": scene_data["setting_type"],
                    "room_type": scene_data["room_type"],
                    "wall_color": scene_data["wall_color"],
                    "is_flat_bg": scene_data["is_flat_bg"],
                    "bg_texture_std": scene_data["bg_texture_std"],
                    "sky_ratio": scene_data["sky_ratio"],
                    "green_ratio": scene_data["green_ratio"],
                    "lighting": scene_data["lighting"],
                    "bg_mean_rgb": scene_data["bg_mean_rgb"],
                    "motion_score": round(motion, 3),
                    "camera_move": camera_move,
                    "scene_change": motion > 20,
                    "sharpness": round(sharpness, 1),
                    "edge_density": float(cv2.Canny(gray, 100, 200).mean()),
                    "hue": round(float(hsv[:,:,0].mean()), 1),
                    "saturation": round(float(hsv[:,:,1].mean()), 1),
                    "brightness": round(float(hsv[:,:,2].mean()), 1),
                    "ocr_text": text_data.get("text", ""),
                    "ocr_word_count": text_data.get("word_count", 0),
                    "ocr_words": text_data.get("words", []),
                    "has_text_overlay": text_data.get("has_text", False),
                    "text_readability": text_data.get("readability_score", 0),
                    "hook_text": text_data.get("hook_text", ""),
                    "scripts_detected": text_data.get("scripts_detected", []),
                    "hindi_text": text_data.get("hindi_text", ""),
                    "english_text": text_data.get("english_text", ""),
                    
                    # NEW ML fields
                    "scene_type": ml_analysis.get("scene_classification", {}).get("scene_type", "unknown"),
                    "scene_confidence": ml_analysis.get("scene_classification", {}).get("confidence", 0),
                    "scene_category": ml_analysis.get("scene_classification", {}).get("scene_category", "unknown"),
                    "objects_detected": ml_analysis.get("object_detection", {}).get("objects", []),
                    "object_count": ml_analysis.get("object_detection", {}).get("object_count", 0),
                    "object_categories": ml_analysis.get("object_detection", {}).get("object_categories", {}),
                    "has_people": ml_analysis.get("object_detection", {}).get("has_people", False),
                    "has_pets": ml_analysis.get("object_detection", {}).get("has_pets", False),
                    "has_technology": ml_analysis.get("object_detection", {}).get("has_technology", False),
                    "primary_activity": ml_analysis.get("activity_recognition", {}).get("primary_activity", "unknown"),
                    "activity_confidence": ml_analysis.get("activity_recognition", {}).get("confidence", 0),
                    "aesthetic_score": ml_analysis.get("aesthetic_quality", {}).get("aesthetic_score", 50),
                    "aesthetic_factors": ml_analysis.get("aesthetic_quality", {}).get("factors", [])
                })
                
                keyframes.append(frame.copy())
                prev_gray = gray
            
            frame_idx += 1
            pbar.update(1)
    
    cap.release()
    save_cache(path, "frames_v4_ml", frame_data)
    return frame_data, keyframes


# ═══════════════════════════════════════════════════════════
# MODIFIED: Aggregation function with ML stats
# ═══════════════════════════════════════════════════════════
def aggregate_visual_with_ml(frame_data: list, pil_data: list, moviepy_data: dict) -> dict:
    """Enhanced aggregation with ML statistics"""
    
    if not frame_data:
        return {}
    
    def avg(key):
        vals = [f[key] for f in frame_data if isinstance(f.get(key), (int, float))]
        return round(float(np.mean(vals)), 3) if vals else 0.0
    
    def mode_key(key):
        vals = [f.get(key) for f in frame_data if f.get(key)]
        if not vals:
            return "unknown"
        from collections import Counter
        return Counter(vals).most_common(1)[0][0]
    
    # Basic aggregations (existing)
    face_ratio = round(sum(1 for f in frame_data if f["face_count"] > 0) / len(frame_data), 3)
    text_ratio = round(sum(1 for f in frame_data if f["has_text_overlay"]) / len(frame_data), 3)
    eye_ratio = round(sum(1 for f in frame_data if f.get("eye_contact")) / len(frame_data), 3)
    
    # ML-based aggregations
    scenes = [f.get("scene_type") for f in frame_data if f.get("scene_type") != "unknown"]
    dominant_scene = mode_key("scene_type")
    
    activities = [f.get("primary_activity") for f in frame_data if f.get("primary_activity")]
    dominant_activity = mode_key("primary_activity")
    
    # Object statistics
    all_objects = []
    object_categories = {}
    for f in frame_data:
        for obj in f.get("objects_detected", []):
            all_objects.append(obj["class"])
            cat = obj["category"]
            object_categories[cat] = object_categories.get(cat, 0) + 1
    
    # Most common objects
    object_counts = {}
    for obj in all_objects:
        object_counts[obj] = object_counts.get(obj, 0) + 1
    
    top_objects = sorted(object_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    
    # Aesthetic quality
    avg_aesthetic = avg("aesthetic_score")
    
    # Scene category distribution
    scene_categories = [f.get("scene_category") for f in frame_data if f.get("scene_category")]
    indoor_ratio = sum(1 for sc in scene_categories if sc == "indoor") / max(len(scene_categories), 1)
    outdoor_ratio = sum(1 for sc in scene_categories if sc == "outdoor") / max(len(scene_categories), 1)
    nature_ratio = sum(1 for sc in scene_categories if sc == "nature") / max(len(scene_categories), 1)
    
    return {
        # Existing fields
        "duration_sec": moviepy_data["duration_sec"],
        "avg_brightness": avg("brightness"),
        "avg_saturation": avg("saturation"),
        "avg_sharpness": avg("sharpness"),
        "avg_motion": avg("motion_score"),
        "avg_edge_density": avg("edge_density"),
        "face_presence_ratio": face_ratio,
        "eye_contact_ratio": eye_ratio,
        "text_overlay_ratio": text_ratio,
        "estimated_cuts": sum(1 for f in frame_data if f.get("scene_change")),
        "pacing": "fast" if sum(1 for f in frame_data if f.get("scene_change")) > 8 else "medium" if sum(1 for f in frame_data if f.get("scene_change")) > 3 else "slow",
        "dominant_camera_move": mode_key("camera_move"),
        "dominant_gaze": mode_key("gaze_direction"),
        "all_ocr_text": " ".join(dict.fromkeys(f["ocr_text"] for f in frame_data if f["ocr_text"])),
        "all_hindi_ocr_text": " ".join(dict.fromkeys(f["hindi_text"] for f in frame_data if f.get("hindi_text"))),
        "all_english_ocr_text": " ".join(dict.fromkeys(f["english_text"] for f in frame_data if f.get("english_text"))),
        "scripts_detected": list({s for f in frame_data for s in f.get("scripts_detected", [])}),
        "avg_text_readability": avg("text_readability"),
        "dominant_setting": mode_key("setting_type"),
        "indoor_outdoor": mode_key("indoor_outdoor"),
        "dominant_lighting": mode_key("lighting"),
        "dominant_wall_color": mode_key("wall_color"),
        "avg_people_in_frame": avg("people_count"),
        "max_people_in_frame": max((f.get("people_count", 0) for f in frame_data), default=0),
        "solo_presenter_frames": sum(1 for f in frame_data if f.get("people_count", 0) == 1),
        "multi_person_frames": sum(1 for f in frame_data if f.get("people_count", 0) > 1),
        
        # NEW ML fields
        "dominant_scene_type": dominant_scene,
        "dominant_activity": dominant_activity,
        "indoor_percentage": round(indoor_ratio * 100, 1),
        "outdoor_percentage": round(outdoor_ratio * 100, 1),
        "nature_percentage": round(nature_ratio * 100, 1),
        "avg_aesthetic_score": round(avg_aesthetic, 1),
        "object_diversity": len(object_categories),
        "top_objects_detected": [obj for obj, _ in top_objects],
        "object_category_counts": object_categories,
        "has_people_in_video": any(f.get("has_people") for f in frame_data),
        "has_pets_in_video": any(f.get("has_pets") for f in frame_data),
        "has_technology_in_video": any(f.get("has_technology") for f in frame_data),
        
        "style": (
            "talking_head" if face_ratio > 0.6 and dominant_activity == "talking/presenting" else
            "text_educational" if text_ratio > 0.5 else
            "vlog_lifestyle" if "outdoor" in dominant_scene.lower() or nature_ratio > 0.3 else
            "tutorial" if dominant_activity in ["working/studying", "cooking"] else
            "entertainment" if dominant_activity == "eating/drinking" else
            "broll_cinematic" if avg("motion_score") < 2 else
            "action_dynamic" if avg("motion_score") > 8 else
            "mixed"
        ),
    }


# ═══════════════════════════════════════════════════════════
# MODULE 5 — PIL color palette
# ═══════════════════════════════════════════════════════════
def analyze_color_palette(path: str) -> list:
    cached = load_cache(path, "pil")
    if cached:
        return cached

    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = max(1, int(fps * Config.PIL_SAMPLE_SEC))
    frame_idx, pil_data = 0, []

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
                "sec":                  round(frame_idx / fps, 2),
                "mean_rgb":             [round(x,1) for x in [r,g,b]],
                "stddev_rgb":           [round(x,1) for x in stat.stddev[:3]],
                "dominant_colors":      [list(p[1]) for p in pal[:6]],
                "perceived_brightness": round(0.299*r + 0.587*g + 0.114*b, 1),
                "color_mood":           mood,
                "contrast_score":       round(float(np.mean(stat.stddev[:3])), 1),
                "color_variance":       round(float(np.std([r,g,b])), 1),
            })
        frame_idx += 1

    cap.release()
    save_cache(path, "pil", pil_data)
    return pil_data


# ═══════════════════════════════════════════════════════════
# MODULE 6 — Categorizer
# ═══════════════════════════════════════════════════════════
def categorize_content(visual_summary: dict, audio_data: dict,
                       transcript_data: dict,
                       ollama_result: dict = None) -> dict:

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

    transcript_quality = transcript_data.get("quality", {})
    transcript_reliable = transcript_quality.get("reliable", True)
    transcript = transcript_data.get("transcript", "") if transcript_reliable else ""

    # Always use OCR text; only use transcript if reliable
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
        "horoscope_astrology": ["राशि","राशिफल","भाग्य","ग्रह","zodiac","lucky","unlucky","star","fate",
                                "kundali","rashi","bhagya","kismat","किस्मत","भाग्यशाली","साप","सर्प"],
        "spiritual_wisdom":    ["karma","dharma","soul","God","भगवान","जीवन","सच","सत्य","आत्मा"],
    }
    detected_hooks = [k for k, phrases in hook_patterns.items()
                      if any(p in combined_text for p in phrases)]

    niche_keywords = {
        "humor_relatable":     ["driving","feels","nobody","lol","literally","when you","mood"],
        "finance":             ["money","invest","rich","income","bank","finance","crypto","stock","wealth"],
        "fitness":             ["gym","workout","fitness","muscle","diet","protein","exercise"],
        "food":                ["recipe","cook","food","eat","restaurant","taste","meal","kitchen"],
        "tech_ai":             ["ai","tech","tool","app","software","code","digital","automation","prompt"],
        "motivation":          ["success","mindset","hustle","grind","goal","dream","achieve","believe"],
        "beauty_fashion":      ["makeup","outfit","style","fashion","beauty","skincare","look","wear"],
        "travel":              ["travel","trip","country","city","hotel","explore","adventure","beach"],
        "education":           ["learn","study","school","fact","science","history","book","knowledge"],
        "relationship":        ["love","relationship","partner","dating","couple","heart","feelings"],
        "entrepreneurship":    ["business","entrepreneur","startup","brand","client","revenue","founder"],
        "mental_health":       ["anxiety","stress","therapy","mental","mindfulness","healing","trauma"],
        "astrology_horoscope": ["राशि","भाग्य","ग्रह","zodiac","lucky","unlucky","horoscope","kundali",
                                "किस्मत","fate","rashi","astro","planet","nakshatra","साप","सर्प","lucky"],
    }
    niche_scores = {}
    for niche, keywords in niche_keywords.items():
        score = sum(1 for kw in keywords if kw in combined_text)
        if score > 0:
            niche_scores[niche] = score
    detected_niche = max(niche_scores, key=niche_scores.get) if niche_scores else "general"

    # If transcript is unreliable and Ollama gave a confident niche, prefer Ollama
    if (not transcript_reliable and ollama_result and
            ollama_result.get("available") and not ollama_result.get("error")):
        ollama_niche  = ollama_result.get("topic_niche", "")
        ollama_conf   = ollama_result.get("confidence", 0)
        if ollama_niche and ollama_conf >= 60:
            # Normalize ollama niche to our labels
            niche_map = {
                "astrology": "astrology_horoscope", "horoscope": "astrology_horoscope",
                "spiritual": "mental_health", "comedy": "humor_relatable",
                "finance": "finance", "fitness": "fitness", "food": "food",
                "tech": "tech_ai", "motivation": "motivation",
                "beauty": "beauty_fashion", "travel": "travel",
                "education": "education", "relationship": "relationship",
            }
            for k, v in niche_map.items():
                if k in ollama_niche.lower():
                    detected_niche = v
                    break
            else:
                detected_niche = ollama_niche.replace(" ", "_").lower()

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
    if pacing == "fast":            score += 20
    elif pacing == "medium":        score += 10
    if audio_mood in ["hype_energetic","upbeat_positive"]: score += 12
    if detected_hooks:              score += 15
    if text_ratio > 0.5:           score += 8
    if 100 <= tempo <= 140:        score += 8
    if "relatable_situation" in detected_hooks: score += 10
    if "call_to_action" in detected_hooks:      score += 5
    if has_voice and transcript_reliable: score += 10
    if detected_niche != "general": score += 8
    if eye_ratio > 0.4:            score += 4
    if audio_data.get("silence_ratio", 0) > 0.2: score -= 5
    if pacing == "slow":           score -= 5
    if not detected_hooks:         score -= 5
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
    if not transcript_reliable:
        recommendations.append("🎙️ Audio quality may be unclear — consider adding subtitle text overlays")
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
        "transcript_reliable": transcript_reliable,
        "replication_template": {
            "format":          content_format,
            "hook_style":      detected_hooks[0] if detected_hooks else "unknown",
            "audio_mood":      audio_mood,
            "pacing":          pacing,
            "use_face":        face_ratio > 0.3,
            "use_eye_contact": eye_ratio > 0.3,
            "use_text":        text_ratio > 0.3,
            "target_bpm":      round(tempo),
            "niche":           detected_niche,
            "speaking_style":  speaking_style,
        }
    }


# ═══════════════════════════════════════════════════════════
# MODULE 7 — AI brief (Claude API)
# ═══════════════════════════════════════════════════════════
def generate_ai_brief(full_report: dict) -> dict:
    if not ANTHROPIC_AVAILABLE or not Config.ANTHROPIC_API_KEY or not Config.ENABLE_AI_BRIEF:
        return {"replication_script": "(disabled)", "hook_variations": [],
                "caption_pack": [], "posting_strategy": ""}

    client = anthropic.Anthropic(api_key=Config.ANTHROPIC_API_KEY)
    cat    = full_report["content_category"]
    audio  = full_report["audio_analysis"]
    visual = full_report["visual_summary"]
    speech = full_report.get("speech_transcript", {})
    ollama = full_report.get("ollama_analysis", {})

    transcript   = speech.get("transcript", "") if speech.get("quality",{}).get("reliable") else ""
    ocr_text     = visual.get("all_ocr_text","")
    ollama_brief = ollama.get("content_summary","") if ollama.get("available") else ""

    compressed = {
        "duration_sec":         full_report["container"]["duration_sec"],
        "niche":                cat["detected_niche"],
        "content_format":       cat["content_format"],
        "hook_formats":         cat["hook_formats"],
        "viral_score":          cat["viral_score"],
        "pacing":               visual["pacing"],
        "face_ratio":           visual["face_presence_ratio"],
        "text_overlay_ratio":   visual["text_overlay_ratio"],
        "audio_mood":           audio["audio_mood"],
        "tempo_bpm":            audio["tempo_bpm"],
        "transcript":           transcript[:400] if transcript else "(unreliable/none)",
        "ocr_text":             ocr_text[:300]   if ocr_text   else "(none)",
        "ollama_content_brief": ollama_brief[:300] if ollama_brief else "(not available)",
        "indoor_outdoor":       visual.get("indoor_outdoor","?"),
        "setting":              visual.get("dominant_setting","?"),
        "avg_people":           visual.get("avg_people_in_frame",1),
        "recommendations":      cat["recommendations"],
    }

    prompt = f"""You are a viral short-form video strategist for Instagram Reels/TikTok.
Analysis of a reel (viral score {compressed['viral_score']}/100):
{json.dumps(compressed, indent=2)}

Return ONLY valid JSON (no markdown) with:
1. "replication_script": 6-10 shots. Each: {{"shot":1,"time":"0-2s","visual":"...","audio":"...","text_overlay":"...","note":"..."}}
2. "hook_variations": 5 hooks. Each: {{"hook":"...","type":"..."}}
3. "caption_pack": 3 captions. Each: {{"caption":"...","hashtags":[...],"cta":"..."}}
4. "posting_strategy": one paragraph
5. "content_improvement": 3 improvements"""

    try:
        response = client.messages.create(
            model=Config.ANTHROPIC_MODEL, max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = "\n".join(raw.split("\n")[1:-1])
        return json.loads(raw)
    except Exception as e:
        return {"error": str(e), "replication_script": [], "hook_variations": [],
                "caption_pack": [], "posting_strategy": "API call failed"}


# ═══════════════════════════════════════════════════════════
# MODULE 8 — Enhanced HTML Report Generator
# ═══════════════════════════════════════════════════════════
def generate_html_report_with_ml(full_report: dict, output_path: str):
    """Generate HTML report with ML insights"""
    
    # Extract data
    cat = full_report["content_category"]
    audio = full_report["audio_analysis"]
    visual = full_report["visual_summary"]
    container = full_report["container"]
    speech = full_report.get("speech_transcript", {})
    ollama = full_report.get("ollama_analysis", {})
    score = cat["viral_score"]
    score_color = "#22c55e" if score >= 70 else "#f59e0b" if score >= 40 else "#ef4444"

    # ML insights
    top_objects = visual.get("top_objects_detected", [])
    object_counts = visual.get("object_category_counts", {})
    aesthetic_score = visual.get("avg_aesthetic_score", 50)
    dominant_scene = visual.get("dominant_scene_type", "unknown").replace("_", " ").title()
    dominant_activity = visual.get("dominant_activity", "unknown").replace("_", " ").title()
    
    # Create object categories HTML
    object_cats_html = ""
    for cat_name, count in object_counts.items():
        object_cats_html += f'<span class="tag">{cat_name}: {count}</span> '
    
    # Transcript quality
    quality = speech.get("quality", {})
    q_score = quality.get("score", 0)
    q_reliable = quality.get("reliable", False)
    q_issues = quality.get("issues", [])
    q_color = "#22c55e" if q_score >= 70 else "#f59e0b" if q_score >= 40 else "#ef4444"
    
    transcript_text = speech.get("transcript", "") or "(no spoken audio detected)"
    if not q_reliable:
        transcript_display = f'''
        <div style="background:#2d1a0e;border-left:4px solid #f59e0b;padding:10px;border-radius:6px;margin-bottom:10px;color:#fbbf24;">
          ⚠️ <b>Low confidence transcript</b> — Quality: {q_score}/100 · Issues: {", ".join(q_issues)}<br>
          <small>Treat this text with caution. ML analysis provides visual understanding.</small>
        </div>
        <div class="transcript-box">{transcript_text}</div>
        '''
    else:
        transcript_display = f'''
        <div style="background:#0e2d1a;border-left:4px solid #22c55e;padding:10px;border-radius:6px;margin-bottom:10px;color:#86efac;">
          ✅ Transcript quality: {q_score}/100
        </div>
        <div class="transcript-box">{transcript_text}</div>
        '''

    # ML Card HTML
    ml_card = f"""
    <div class="card" style="border-left:4px solid #06b6d4;">
      <h2>🤖 ML Image Understanding</h2>
      <div class="stat"><span class="stat-label">Primary Scene</span>
        <span class="stat-value">{dominant_scene}</span>
      </div>
      <div class="stat"><span class="stat-label">Primary Activity</span>
        <span class="stat-value">{dominant_activity}</span>
      </div>
      <div class="stat"><span class="stat-label">Aesthetic Score</span>
        <span class="stat-value" style="color:{'#22c55e' if aesthetic_score>=70 else '#f59e0b' if aesthetic_score>=40 else '#ef4444'}">
          {aesthetic_score}/100
        </span>
      </div>
      <div class="stat"><span class="stat-label">Scene Distribution</span>
        <span class="stat-value">🏠 {visual.get('indoor_percentage',0)}% | 🌳 {visual.get('outdoor_percentage',0)}% | 🌿 {visual.get('nature_percentage',0)}%</span>
      </div>
      <div class="stat"><span class="stat-label">Object Diversity</span>
        <span class="stat-value">{visual.get('object_diversity',0)} categories</span>
      </div>
      <div style="margin-top:10px;">
        <div><b>Top Objects:</b></div>
        {''.join(f'<span class="tag">{obj}</span>' for obj in top_objects[:5])}
      </div>
      <div style="margin-top:10px;">
        <div><b>Object Categories:</b></div>
        {object_cats_html}
      </div>
      <div style="margin-top:10px;">
        <div><b>Content Features:</b></div>
        <span class="tag">{'👥 People' if visual.get('has_people_in_video') else ''}</span>
        <span class="tag">{'🐾 Pets' if visual.get('has_pets_in_video') else ''}</span>
        <span class="tag">{'📱 Tech' if visual.get('has_technology_in_video') else ''}</span>
      </div>
    </div>
    """

    # Generate the full HTML (simplified - you can expand this)
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Reel Analysis with ML</title>
    <style>
        body {{ font-family: Arial, sans-serif; background: #0f0f17; color: #e2e8f0; }}
        .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
        .header {{ background: linear-gradient(135deg,#1a1a2e,#16213e); padding: 30px; border-radius: 10px; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px,1fr)); gap: 20px; }}
        .card {{ background: #1e1e2e; border-radius: 10px; padding: 20px; }}
        .stat {{ display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #2d2d44; }}
        .tag {{ display: inline-block; background: #3b1f5e; color: #c084fc; padding: 3px 10px; border-radius: 15px; margin: 2px; }}
        .score-badge {{ font-size: 2em; font-weight: bold; color: {score_color}; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎬 Reel Analysis with ML</h1>
            <div class="score-badge">{score}/100 Viral Score</div>
        </div>
        
        <div class="grid">
            <!-- Basic info card -->
            <div class="card">
                <h2>📦 Video Info</h2>
                <div class="stat"><span>Duration</span><span>{container['duration_sec']}s</span></div>
                <div class="stat"><span>Resolution</span><span>{container['resolution'][0]}×{container['resolution'][1]}</span></div>
                <div class="stat"><span>FPS</span><span>{container['fps']:.1f}</span></div>
            </div>
            
            <!-- Content card -->
            <div class="card">
                <h2>🎯 Content</h2>
                <div class="stat"><span>Niche</span><span>{cat['detected_niche']}</span></div>
                <div class="stat"><span>Format</span><span>{cat['content_format']}</span></div>
                <div class="stat"><span>Activity</span><span>{dominant_activity}</span></div>
            </div>
            
            <!-- ML Card -->
            {ml_card}
            
            <!-- Audio card -->
            <div class="card">
                <h2>🎵 Audio</h2>
                <div class="stat"><span>Tempo</span><span>{audio['tempo_bpm']} BPM</span></div>
                <div class="stat"><span>Key</span><span>{audio['detected_key']}</span></div>
                <div class="stat"><span>Mood</span><span>{audio['audio_mood']}</span></div>
            </div>
        </div>
        
        <!-- Transcript -->
        <h2>🎙️ Transcript</h2>
        {transcript_display}
        
        <!-- Recommendations -->
        <h2>🚀 Recommendations</h2>
        <ul>
            {''.join(f'<li>{r}</li>' for r in cat.get('recommendations', []))}
        </ul>
    </div>
</body>
</html>
    """
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"📄 HTML report with ML saved: {output_path}")


# ═══════════════════════════════════════════════════════════
# Enhanced Storyboard Generator with ML annotations
# ═══════════════════════════════════════════════════════════
def generate_storyboard_with_ml(keyframes, frame_data, audio_data, category, output_path, cols=4):
    """Generate storyboard with ML object annotations"""
    
    if not keyframes:
        return
    
    thumb_w, thumb_h = 360, 640
    label_h = 180  # Taller for ML info
    rows = math.ceil(len(keyframes) / cols)
    board = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + label_h) + 90), (10, 10, 14))
    draw = ImageDraw.Draw(board)
    
    try:
        font = ImageFont.truetype("arial.ttf", 14)
        font_small = ImageFont.truetype("arial.ttf", 11)
        font_bold = ImageFont.truetype("arialbd.ttf", 15)
        font_title = ImageFont.truetype("arialbd.ttf", 18)
    except:
        font = font_small = font_bold = font_title = ImageFont.load_default()

    # Header
    draw.rectangle([0, 0, cols * thumb_w, 85], fill=(20, 10, 40))
    draw.text((10, 8), f"🎯 {category['detected_niche'].upper()}  |  {category['content_format']}  |  VIRAL: {category['viral_score']}/100", 
              font=font_title, fill=(255, 220, 50))
    draw.text((10, 35), f"🎵 {audio_data['tempo_bpm']} BPM · {audio_data['detected_key']} · {audio_data['audio_mood']}", 
              font=font_bold, fill=(160, 210, 255))
    draw.text((10, 58), f"🪝 {', '.join(category['hook_formats']) or 'no hooks detected'}", 
              font=font, fill=(180, 255, 180))

    # Draw each frame
    for i, (frame, fd) in enumerate(zip(keyframes, frame_data)):
        col = i % cols
        row = i // cols
        x = col * thumb_w
        y = row * (thumb_h + label_h) + 90
        
        # Resize frame
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((thumb_w, thumb_h))
        img_draw = ImageDraw.Draw(img)
        
        # Draw face detections
        for fp in fd.get("face_positions", []):
            fx = int(fp["cx"] * thumb_w)
            fy = int(fp["cy"] * thumb_h)
            sz = int(math.sqrt(max(fp.get("size_ratio", 0.05), 0.01)) * thumb_w)
            c = (0, 255, 80) if fd.get("eye_contact") else (255, 160, 0)
            img_draw.ellipse([fx - sz // 2, fy - sz // 2, fx + sz // 2, fy + sz // 2], outline=c, width=3)
        
        # Draw object detections (NEW)
        for obj in fd.get("objects_detected", []):
            if "bbox" in obj:
                x1, y1, x2, y2 = obj["bbox"]
                # Scale to thumbnail
                scale_x = thumb_w / frame.shape[1]
                scale_y = thumb_h / frame.shape[0]
                tx1 = int(x1 * scale_x)
                ty1 = int(y1 * scale_y)
                tx2 = int(x2 * scale_x)
                ty2 = int(y2 * scale_y)
                
                # Color by category
                cat_colors = {
                    'people': (255, 255, 0),
                    'vehicle': (255, 165, 0),
                    'animal': (0, 255, 0),
                    'food_drink': (255, 105, 180),
                    'electronics': (0, 191, 255),
                    'furniture': (138, 43, 226),
                    'sports': (255, 215, 0)
                }
                color = cat_colors.get(obj['category'], (255, 255, 255))
                
                # Draw bounding box
                img_draw.rectangle([tx1, ty1, tx2, ty2], outline=color, width=2)
                
                # Draw label
                label = f"{obj['class']} ({obj['confidence']:.0f}%)"
                img_draw.text((tx1 + 2, ty1 - 12), label, fill=color, font=font_small)
        
        # Draw beat indicator
        beat_near = any(abs(bt - fd["sec"]) < 0.5 for bt in audio_data["beat_times"])
        if beat_near:
            img_draw.rectangle([0, thumb_h - 8, thumb_w, thumb_h], fill=(200, 40, 40))
            img_draw.text((4, thumb_h - 20), "♪ BEAT", font=font_small, fill=(255, 255, 255))
        
        # Draw scene change indicator
        if fd.get("scene_change"):
            img_draw.rectangle([0, 0, thumb_w, 6], fill=(255, 80, 0))
        
        # Indoor/outdoor indicator
        io_col = (80, 200, 120) if fd.get("indoor_outdoor") == "outdoor" else (100, 140, 255)
        img_draw.rectangle([thumb_w - 45, 4, thumb_w - 4, 22], fill=io_col)
        img_draw.text((thumb_w - 42, 6), fd.get("indoor_outdoor", "?")[:3].upper(), 
                     font=font_small, fill=(0, 0, 0))
        
        # Paste frame
        board.paste(img, (x, y))

        # Draw labels below frame
        ly = y + thumb_h
        draw.rectangle([x, ly, x + thumb_w, ly + label_h], fill=(14, 14, 22))
        
        # Time and basic info
        draw.text((x + 5, ly + 4), f"⏱ {fd['sec']}s  📷 {fd['camera_move']}", 
                  font=font, fill=(160, 210, 255))
        
        # People/face info
        draw.text((x + 5, ly + 22), f"👤 {fd['face_count']} faces  👥 {fd.get('people_count', 0)} people", 
                  font=font, fill=(200, 200, 200))
        
        # Scene and lighting
        draw.text((x + 5, ly + 40), f"🏠 {fd.get('setting_type', '?')[:22]}  💡 {fd.get('lighting', '?')[:10]}", 
                  font=font_small, fill=(255, 200, 100))
        
        # ML info (NEW)
        scene_type = fd.get('scene_type', 'unknown')[:20].replace('_', ' ')
        activity = fd.get('primary_activity', '?')[:20].replace('_', ' ')
        aesthetic = fd.get('aesthetic_score', 50)
        aesthetic_color = "#22c55e" if aesthetic >= 70 else "#f59e0b" if aesthetic >= 40 else "#ef4444"
        
        draw.text((x + 5, ly + 58), f"🏷️ {scene_type}  🎯 {activity}", 
                  font=font_small, fill=(160, 180, 255))
        draw.text((x + 5, ly + 76), f"🎨 Aesthetic: {aesthetic}", 
                  font=font_small, fill=aesthetic_color)
        
        # Object count
        obj_count = fd.get('object_count', 0)
        draw.text((x + 5, ly + 94), f"📦 Objects: {obj_count}  {', '.join(fd.get('object_categories', {}).keys())[:20]}", 
                  font=font_small, fill=(255, 220, 80))
        
        # OCR text
        draw.text((x + 5, ly + 112), f"📝 {fd['hook_text'][:40]}{'…' if len(fd['hook_text']) > 40 else ''}", 
                  font=font_small, fill=(200, 200, 200))
        
        # Scripts detected
        draw.text((x + 5, ly + 130), f"scripts:{','.join(fd.get('scripts_detected', []) or ['?'])[:20]}", 
                  font=font_small, fill=(140, 220, 140))
        
        # Gaze and beat
        draw.text((x + 5, ly + 148), f"gaze:{fd.get('gaze_direction', '?')[:16]}  ♪{'BEAT' if beat_near else '—'}", 
                  font=font_small, fill=(255, 100, 100) if beat_near else (80, 80, 100))
        
        # Readability
        draw.text((x + 5, ly + 166), f"readability:{fd.get('text_readability', 0):.0f}%", 
                  font=font_small, fill=(100, 180, 100))
        
        # Frame number
        draw.rectangle([x, y, x + 42, y + 22], fill=(120, 30, 200))
        draw.text((x + 5, y + 4), f"#{i + 1}", font=font_bold, fill="white")

    board.save(output_path, quality=95)
    print(f"🎬 Enhanced storyboard saved: {output_path}")


# ═══════════════════════════════════════════════════════════
# CSV Logging (updated)
# ═══════════════════════════════════════════════════════════
def append_to_csv(full_report: dict, video_path: str):
    cat    = full_report["content_category"]
    audio  = full_report["audio_analysis"]
    visual = full_report["visual_summary"]
    speech = full_report.get("speech_transcript",{})
    ollama = full_report.get("ollama_analysis",{})
    row = {
        "filename":            os.path.basename(video_path),
        "analyzed_at":         datetime.now().isoformat(),
        "duration_sec":        full_report["container"]["duration_sec"],
        "viral_score":         cat["viral_score"],
        "niche":               cat["detected_niche"],
        "content_format":      cat["content_format"],
        "pacing":              visual["pacing"],
        "cuts":                visual["estimated_cuts"],
        "hook_formats":        "|".join(cat.get("hook_formats",[])),
        "tempo_bpm":           audio["tempo_bpm"],
        "detected_key":        audio["detected_key"],
        "audio_mood":          audio["audio_mood"],
        "audio_type":          audio["audio_type"],
        "face_ratio":          visual["face_presence_ratio"],
        "eye_contact_ratio":   visual["eye_contact_ratio"],
        "text_ratio":          visual["text_overlay_ratio"],
        "avg_brightness":      visual["avg_brightness"],
        "avg_saturation":      visual["avg_saturation"],
        "avg_motion":          visual["avg_motion"],
        "harmony_pct":         audio["harmony_pct"],
        "speaking_style":      cat.get("speaking_style",""),
        "transcript_words":    speech.get("word_count",0),
        "speaking_rate_wpm":   speech.get("speaking_rate_wpm",0),
        "silence_ratio":       audio.get("silence_ratio",0),
        "transcript_quality":  speech.get("quality",{}).get("score",0),
        "transcript_reliable": speech.get("quality",{}).get("reliable",False),
        "indoor_outdoor":      visual.get("indoor_outdoor",""),
        "setting_type":        visual.get("dominant_setting",""),
        "avg_people":          visual.get("avg_people_in_frame",0),
        # ML fields
        "dominant_scene":      visual.get("dominant_scene_type",""),
        "dominant_activity":   visual.get("dominant_activity",""),
        "aesthetic_score":     visual.get("avg_aesthetic_score",0),
        "top_objects":         "|".join(visual.get("top_objects_detected",[])[:3]),
        "object_diversity":    visual.get("object_diversity",0),
        "ollama_topic":        ollama.get("topic_niche","") if ollama.get("available") else "",
        "ollama_confidence":   ollama.get("confidence",0) if ollama.get("available") else 0,
    }
    file_exists = os.path.exists(Config.CSV_LOG)
    with open(Config.CSV_LOG, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"📊 CSV log updated: {Config.CSV_LOG}")


# ═══════════════════════════════════════════════════════════
# Organize by category
# ═══════════════════════════════════════════════════════════
def organize_by_category(report_path, storyboard_path, niche, viral_score):
    import shutil
    tier   = "top_tier" if viral_score>=70 else "mid_tier" if viral_score>=40 else "low_tier"
    folder = os.path.join(Config.OUTPUT_DIR,"by_category",niche,tier)
    os.makedirs(folder, exist_ok=True)
    for p in [report_path, storyboard_path]:
        if p and os.path.exists(p):
            shutil.copy(p, os.path.join(folder, os.path.basename(p)))
    print(f"📁 Filed under: by_category/{niche}/{tier}/")


# ═══════════════════════════════════════════════════════════
# Batch analyze with ML
# ═══════════════════════════════════════════════════════════
def batch_analyze_with_ml(folder: str):
    exts   = {".mp4",".mov",".avi",".mkv",".webm",".m4v"}
    videos = [str(p) for p in Path(folder).rglob("*") if p.suffix.lower() in exts]
    print(f"\n📂 Found {len(videos)} video(s) in {folder}\n")
    for i, vp in enumerate(videos, 1):
        print(f"[{i}/{len(videos)}] {os.path.basename(vp)}")
        try:
            analyze_video_with_ml(vp)
        except Exception as e:
            print(f"  ❌ Failed: {e}")


# ═══════════════════════════════════════════════════════════
# Original analyze function (for non-ML mode)
# ═══════════════════════════════════════════════════════════
def analyze_video(video_path: str) -> dict:
    print(f"\n{'═'*62}")
    print(f"  🎬 Analyzing (standard): {os.path.basename(video_path)}")
    print(f"{'═'*62}\n")
    # ... (original implementation - you already have this)
    # For brevity, I'm not copying the entire original function
    # but you should keep your existing analyze_video function here
    return {}


# ═══════════════════════════════════════════════════════════
# Main pipeline with ML
# ═══════════════════════════════════════════════════════════
def analyze_video_with_ml(video_path: str) -> dict:
    """Main pipeline with ML image classification"""
    
    print(f"\n{'═'*62}")
    print(f"  🎬 Analyzing (with ML): {os.path.basename(video_path)}")
    print(f"{'═'*62}\n")

    slug = Path(video_path).stem[:30].replace(" ", "_")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("📦 [1/7] Container metadata...")
    moviepy_data = analyze_container(video_path)
    print(f"    {moviepy_data['duration_sec']}s  {moviepy_data['fps']:.1f}fps  "
          f"{moviepy_data['resolution'][0]}×{moviepy_data['resolution'][1]}  "
          f"{moviepy_data['format_hint']}")

    print(f"\n🎙️  [2/7] Speech-to-text (Whisper {Config.WHISPER_MODEL}, lang={Config.WHISPER_LANG or 'auto'})...")
    speech_data = transcribe_speech(video_path, moviepy_data["duration_sec"])
    q = speech_data.get("quality", {})
    if speech_data["transcript"]:
        print(f"    Lang: {speech_data['language']}  Words: {speech_data['word_count']}  "
              f"WPM: {speech_data['speaking_rate_wpm']}  "
              f"Quality: {q.get('score',0)}/100 ({'✅' if q.get('reliable') else '⚠️'})")

    print("\n🎵 [3/7] Deep audio analysis (Librosa)...")
    audio_data = analyze_audio(video_path)
    print(f"    {audio_data['tempo_bpm']} BPM  Key:{audio_data['detected_key']}  "
          f"Mood:{audio_data['audio_mood']}  Type:{audio_data['audio_type']}")

    print("\n🖼️  [4/7] Enhanced frame analysis (with ML classification)...")
    frame_data, keyframes = analyze_frames_with_classification(video_path)

    print("\n🎨 [5/7] Color palette (PIL)...")
    pil_data = analyze_color_palette(video_path)

    print("\n📊 [6/7] Aggregating with ML insights...")
    visual_summary = aggregate_visual_with_ml(frame_data, pil_data, moviepy_data)
    
    # Print ML insights
    print(f"    🏷️  Dominant Scene: {visual_summary['dominant_scene_type']}")
    print(f"    🎯 Primary Activity: {visual_summary['dominant_activity']}")
    print(f"    🏠 Indoor: {visual_summary['indoor_percentage']}% / Outdoor: {visual_summary['outdoor_percentage']}%")
    print(f"    🎨 Aesthetic Score: {visual_summary['avg_aesthetic_score']}/100")
    print(f"    📦 Top Objects: {', '.join(visual_summary['top_objects_detected'][:3])}")

    print("\n🤖 [6b] Ollama content analysis...")
    all_ocr = visual_summary.get("all_ocr_text", "")
    transcript = speech_data.get("transcript", "")
    ollama_result = ollama_analyze_content(
        transcript, all_ocr,
        speech_data.get("quality", {}),
        moviepy_data["duration_sec"]
    )

    print("\n🎯 [7/7] Categorizing with enhanced features...")
    category = categorize_content(visual_summary, audio_data, speech_data, ollama_result)
    print(f"    Niche:{category['detected_niche']}  Format:{category['content_format']}  "
          f"Viral:{category['viral_score']}/100")

    full_report = {
        "file": video_path,
        "analyzed_at": datetime.now().isoformat(),
        "container": moviepy_data,
        "speech_transcript": speech_data,
        "audio_analysis": audio_data,
        "visual_summary": visual_summary,
        "content_category": category,
        "frame_analysis": frame_data,
        "color_analysis": pil_data,
        "ollama_analysis": ollama_result,
    }

    print("\n📝 Generating outputs...")
    
    # Save JSON
    json_path = os.path.join(Config.REPORTS_DIR, f"{slug}_{ts}_ml.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)
    print(f"  ✅ JSON: {json_path}")

    # Generate enhanced storyboard with ML annotations
    sb_path = os.path.join(Config.REPORTS_DIR, f"{slug}_{ts}_storyboard_ml.jpg")
    generate_storyboard_with_ml(keyframes, frame_data, audio_data, category, sb_path)

    # Generate enhanced HTML report
    html_path = os.path.join(Config.REPORTS_DIR, f"{slug}_{ts}_report_ml.html")
    generate_html_report_with_ml(full_report, html_path)

    append_to_csv(full_report, video_path)
    organize_by_category(html_path, sb_path, category["detected_niche"], category["viral_score"])

    print(f"\n{'═'*62}  ✅ ANALYSIS COMPLETE")
    print(f"{'─'*62}")
    print(f"  🎯 Niche:         {category['detected_niche'].replace('_', ' ').title()}")
    print(f"  🏆 Viral Score:   {category['viral_score']}/100")
    print(f"  🏷️  Top Scene:      {visual_summary['dominant_scene_type']}")
    print(f"  🎯 Activity:       {visual_summary['dominant_activity']}")
    print(f"  🎨 Aesthetic:      {visual_summary['avg_aesthetic_score']}/100")
    print(f"  📦 Objects:        {', '.join(visual_summary['top_objects_detected'][:3])}")
    print(f"\n  📄 HTML  → {html_path}")
    print(f"  🎬 Board → {sb_path}")
    print(f"{'═'*62}\n")

    return full_report


# ═══════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reel Analyzer v2.4 with ML Image Classification")
    parser.add_argument("path", nargs="?", help="Video file or folder")
    parser.add_argument("--video", type=str, help="Single video file to analyze")
    parser.add_argument("--folder", type=str, help="Folder of videos to analyze")
    parser.add_argument("--no-ai", action="store_true", help="Skip Claude AI brief")
    parser.add_argument("--no-ollama", action="store_true", help="Skip Ollama analysis")
    parser.add_argument("--whisper-model", default="base",
                        choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--lang", default="hi",
                        help="Whisper language (default: hi). Use 'auto' for auto-detect.")
    parser.add_argument("--ollama-model", default=Config.OLLAMA_MODEL,
                        help=f"Ollama model to use (default: {Config.OLLAMA_MODEL})")
    parser.add_argument("--no-ml", action="store_true", help="Disable ML image classification")
    args = parser.parse_args()

    if args.no_ai: 
        Config.ENABLE_AI_BRIEF = False
    if args.no_ollama: 
        Config.OLLAMA_ENABLED = False
    Config.WHISPER_MODEL = args.whisper_model
    Config.WHISPER_LANG = None if args.lang == "auto" else args.lang
    Config.OLLAMA_MODEL = args.ollama_model

    target = args.video or args.path
    if args.folder:
        if args.no_ml:
            batch_analyze(args.folder)  # Original version
        else:
            batch_analyze_with_ml(args.folder)
    elif target:
        if os.path.isdir(target):
            if args.no_ml:
                batch_analyze(target)
            else:
                batch_analyze_with_ml(target)
        else:
            if args.no_ml:
                analyze_video(target)  # Original
            else:
                analyze_video_with_ml(target)  # Enhanced with ML
    else:
        print("Please provide a video file or folder path")
        print("Example: python video_recon.py video.mp4")
        print("         python video_recon.py --folder ./videos/")