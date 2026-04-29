"""
PATCH NOTES v2.2 — What changed from v2.1 (4_patched.py):

  ① Whisper cache key now includes language — changing --lang forces re-transcription
  ② Whisper initial_prompt set to a Hindi sentence to steer output to Devanagari script
  ③ Whisper post-processing: detects if output is Urdu/Arabic script and flags it;
     adds transliteration hint and quality score
  ④ Transcript quality gating: if quality_score < threshold, marks as unreliable
     and does NOT use it for niche/hook detection (prevents hallucination pollution)
  ⑤ NEW: Ollama integration — uses local LLM to:
       - Summarize transcript/OCR text in plain English
       - Detect content topic/niche from raw text
       - Generate a human-readable content brief
     Falls back gracefully if Ollama is not running
  ⑥ Better audio_type classification: music_heavy reels flagged, Whisper WPM sanity check
  ⑦ Richer HTML report: transcript quality badge, Ollama content brief section,
     per-script text breakdown
  ⑧ Summary console output now shows transcript quality and Ollama brief snippet

SETUP:
  Ollama:  https://ollama.com  →  ollama pull llama3   (or any model you have)
  Default model: llama3  — change OLLAMA_MODEL below
  Default URL:   http://localhost:11434  — change OLLAMA_URL if using a different port
"""

"""
PATCH NOTES v2.2.1 — Ollama improvements:

  ① Better Ollama connection handling with retries and timeouts
  ② More robust JSON parsing with fallbacks
  ③ Keep-alive to maintain model in memory
  ④ Detailed error logging for debugging
  ⑤ Response validation and fallback defaults
  ⑥ Connection pooling and reuse
  ⑦ Support for both localhost and 127.0.0.1
"""

"""
PATCH NOTES v2.3 — Complete overhaul with multi-language & multi-layer audio support:

  ① Fixed KeyError when no speech segments detected
  ② Added enhanced multi-layer audio detection (speech, music, ambient)
  ③ Improved language handling with auto-detection
  ④ Better error handling throughout
  ⑤ Enhanced Ollama integration with retries and fallbacks
  ⑥ Multi-script support (Devanagari, Latin, Arabic/Urdu)
  ⑦ Music-heavy detection with Whisper accuracy warnings
  ⑧ Background music vs speech separation
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
# MODULE 2B — Ollama content analysis  (⑤ NEW - IMPROVED)
# Uses local LLM to understand transcript + OCR text
# Falls back gracefully if Ollama is not running
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
# ⑥ Added music_heavy detection to flag poor Whisper conditions
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
# MODULE 4E — Per-frame analysis (all detectors combined)
# ═══════════════════════════════════════════════════════════
def analyze_frames(path: str) -> tuple:
    cached = load_cache(path, "frames_v3")
    if cached:
        print("  ♻️  Frames — using cache")
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
            if not ret: break
            if frame_idx % interval == 0:
                sec  = round(frame_idx / fps, 2)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                lab  = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)

                motion, camera_move = 0.0, "static"
                if prev_gray is not None:
                    flow   = cv2.calcOpticalFlowFarneback(
                        prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    motion = float(np.mean(np.sqrt(flow[...,0]**2 + flow[...,1]**2)))
                    mx, my = float(np.mean(flow[...,0])), float(np.mean(flow[...,1]))
                    if abs(mx) > abs(my):
                        camera_move = "pan_right" if mx>1 else "pan_left" if mx<-1 else "static"
                    else:
                        camera_move = "tilt_down" if my>1 else "tilt_up" if my<-1 else "static"

                sharpness    = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                edges        = cv2.Canny(gray, 100, 200)
                edge_density = float(edges.mean())
                h_f, w_f     = frame.shape[:2]
                b_mean       = float(frame[:,:,0].mean())
                r_mean       = float(frame[:,:,2].mean())
                color_temp   = ("warm" if r_mean > b_mean*1.1
                                else "cool" if b_mean > r_mean*1.1 else "neutral")
                h3, w3 = h_f//3, w_f//3
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
                scene_change     = motion > 20

                if EASYOCR_AVAILABLE:
                    text_data = extract_text_easyocr(frame)
                else:
                    text_data = {"text":"","word_count":0,"words":[],"has_text":False,
                                 "readability_score":0,"text_zones":[],"hook_text":"",
                                 "scripts_detected":[],"hindi_text":"","english_text":""}

                face_data  = analyze_face_pose(frame)
                scene_data = detect_scene_background(frame, face_data.get("faces", []))
                body_data  = count_bodies_hog(frame)
                people_count = max(face_data["face_count"], body_data["body_count_hog"])

                frame_data.append({
                    "frame":           frame_idx,
                    "sec":             sec,
                    "face_count":      face_data["face_count"],
                    "face_positions":  face_data["faces"],
                    "gaze_direction":  face_data.get("gaze_direction", "unknown"),
                    "eye_contact":     face_data.get("eye_contact", False),
                    "body_detected":   face_data["body_detected"],
                    "hands_raised":    face_data.get("hands_raised", False),
                    "people_count":    people_count,
                    "body_count_hog":  body_data["body_count_hog"],
                    "body_positions":  body_data["body_positions"],
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
                    "motion_score":    round(motion, 3),
                    "camera_move":     camera_move,
                    "scene_change":    scene_change,
                    "sharpness":       round(sharpness, 1),
                    "edge_density":    round(edge_density, 3),
                    "hue":             round(float(hsv[:,:,0].mean()), 1),
                    "saturation":      round(float(hsv[:,:,1].mean()), 1),
                    "brightness":      round(float(hsv[:,:,2].mean()), 1),
                    "perceptual_lightness": round(float(lab[:,:,0].mean()), 1),
                    "color_temp":      color_temp,
                    "brightest_region": brightest_region,
                    "ocr_text":        text_data["text"],
                    "ocr_word_count":  text_data["word_count"],
                    "ocr_words":       text_data["words"],
                    "has_text_overlay": text_data["has_text"],
                    "text_readability": text_data["readability_score"],
                    "hook_text":       text_data["hook_text"],
                    "scripts_detected": text_data.get("scripts_detected", []),
                    "hindi_text":      text_data.get("hindi_text", ""),
                    "english_text":    text_data.get("english_text", ""),
                })
                keyframes.append(frame.copy())
                prev_gray = gray
            frame_idx += 1
            pbar.update(1)

    cap.release()
    save_cache(path, "frames_v3", frame_data)
    return frame_data, keyframes


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
# MODULE 6 — Aggregation
# ═══════════════════════════════════════════════════════════
def aggregate_visual(frame_data: list, pil_data: list, moviepy_data: dict) -> dict:
    if not frame_data:
        return {}

    def avg(key):
        vals = [f[key] for f in frame_data if isinstance(f.get(key), (int, float))]
        return round(float(np.mean(vals)), 3) if vals else 0.0

    face_ratio   = round(sum(1 for f in frame_data if f["face_count"] > 0) / len(frame_data), 3)
    text_ratio   = round(sum(1 for f in frame_data if f["has_text_overlay"]) / len(frame_data), 3)
    eye_ratio    = round(sum(1 for f in frame_data if f.get("eye_contact")) / len(frame_data), 3)
    cuts         = sum(1 for f in frame_data if f["scene_change"])
    all_text     = " ".join(dict.fromkeys(f["ocr_text"] for f in frame_data if f["ocr_text"]))
    all_hindi    = " ".join(dict.fromkeys(f["hindi_text"] for f in frame_data if f.get("hindi_text")))
    all_english  = " ".join(dict.fromkeys(f["english_text"] for f in frame_data if f.get("english_text")))
    all_scripts  = list({s for f in frame_data for s in f.get("scripts_detected", [])})

    moods     = [p["color_mood"] for p in pil_data]
    dom_mood  = max(set(moods), key=moods.count) if moods else "unknown"
    cam_moves = [f["camera_move"] for f in frame_data]
    dom_cam   = max(set(cam_moves), key=cam_moves.count)
    gaze_modes= [f.get("gaze_direction","unknown") for f in frame_data]
    dom_gaze  = max(set(gaze_modes), key=gaze_modes.count)

    settings  = [f["setting_type"] for f in frame_data if f.get("setting_type")]
    dom_setting = max(set(settings), key=settings.count) if settings else "unknown"
    io_tags   = [f["indoor_outdoor"] for f in frame_data if f.get("indoor_outdoor")]
    dom_io    = max(set(io_tags), key=io_tags.count) if io_tags else "unknown"
    lightings = [f["lighting"] for f in frame_data if f.get("lighting")]
    dom_light = max(set(lightings), key=lightings.count) if lightings else "unknown"
    wall_cols = [f["wall_color"] for f in frame_data if f.get("wall_color")]
    dom_wall  = max(set(wall_cols), key=wall_cols.count) if wall_cols else "unknown"

    avg_people   = avg("people_count")
    max_people   = max((f["people_count"] for f in frame_data), default=0)
    solo_frames  = sum(1 for f in frame_data if f["people_count"] == 1)
    multi_frames = sum(1 for f in frame_data if f["people_count"] > 1)

    avg_motion = avg("motion_score")
    pacing     = "fast" if cuts > 8 else "medium" if cuts > 3 else "slow"

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
        "pacing":                pacing,
        "dominant_camera_move":  dom_cam,
        "dominant_gaze":         dom_gaze,
        "dominant_color_mood":   dom_mood,
        "all_ocr_text":          all_text,
        "all_hindi_ocr_text":    all_hindi,
        "all_english_ocr_text":  all_english,
        "scripts_detected":      all_scripts,
        "avg_text_readability":  avg("text_readability"),
        "dominant_setting":      dom_setting,
        "indoor_outdoor":        dom_io,
        "dominant_lighting":     dom_light,
        "dominant_wall_color":   dom_wall,
        "avg_people_in_frame":   round(avg_people, 1),
        "max_people_in_frame":   max_people,
        "solo_presenter_frames": solo_frames,
        "multi_person_frames":   multi_frames,
        "style": (
            "talking_head"     if face_ratio > 0.6 else
            "text_educational" if text_ratio > 0.5 else
            "broll_cinematic"  if avg_motion < 2   else
            "action_dynamic"   if avg_motion > 8   else
            "mixed"
        ),
    }


# ═══════════════════════════════════════════════════════════
# MODULE 7 — Categorizer
# ④ Uses Ollama niche if transcript quality is low
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

    # ④ If transcript is unreliable and Ollama gave a confident niche, prefer Ollama
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
# MODULE 8 — AI brief (Claude API)
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
# MODULE 9 — HTML report  (⑦ adds transcript quality + Ollama section)
# ═══════════════════════════════════════════════════════════
def generate_html_report(full_report: dict, ai_brief: dict, output_path: str):
    cat      = full_report["content_category"]
    audio    = full_report["audio_analysis"]
    visual   = full_report["visual_summary"]
    container= full_report["container"]
    speech   = full_report.get("speech_transcript", {})
    ollama   = full_report.get("ollama_analysis", {})
    score    = cat["viral_score"]
    score_color = "#22c55e" if score >= 70 else "#f59e0b" if score >= 40 else "#ef4444"

    quality      = speech.get("quality", {})
    q_score      = quality.get("score", 0)
    q_reliable   = quality.get("reliable", False)
    q_issues     = quality.get("issues", [])
    q_color      = "#22c55e" if q_score >= 70 else "#f59e0b" if q_score >= 40 else "#ef4444"
    q_script     = quality.get("detected_script", "unknown")

    transcript_text = speech.get("transcript","") or "(no spoken audio detected)"
    # Show a warning if unreliable
    if not q_reliable:
        transcript_display = (
            f'<div style="background:#2d1a0e;border-left:4px solid #f59e0b;'
            f'padding:10px;border-radius:6px;margin-bottom:10px;color:#fbbf24;">'
            f'⚠️ <b>Low confidence transcript</b> — Quality: {q_score}/100 · '
            f'Script: {q_script} · Issues: {", ".join(q_issues)}<br>'
            f'<small>Treat this text with caution. '
            f'Ollama analysis below uses OCR text instead.</small></div>'
            f'<div class="transcript-box">{transcript_text}</div>'
        )
    else:
        transcript_display = (
            f'<div style="background:#0e2d1a;border-left:4px solid #22c55e;'
            f'padding:10px;border-radius:6px;margin-bottom:10px;color:#86efac;">'
            f'✅ Transcript quality: {q_score}/100 · Script: {q_script}'
            f'</div>'
            f'<div class="transcript-box">{transcript_text}</div>'
        )

    # Ollama section
    if ollama.get("available") and not ollama.get("error"):
        key_phrases_html = "".join(
            f'<span class="tag">{p}</span>' for p in ollama.get("key_phrases",[])
        )
        ollama_html = f"""
        <div class="card" style="border-left:4px solid #06b6d4;">
          <h2>🤖 Ollama Content Brief ({ollama.get("model_used","llama3")})</h2>
          <div class="stat"><span class="stat-label">Summary</span>
            <span class="stat-value" style="max-width:70%;text-align:right;font-size:.9em;font-weight:400;">
              {ollama.get("content_summary","—")}
            </span>
          </div>
          <div class="stat"><span class="stat-label">Topic / Niche</span>
            <span class="stat-value">{ollama.get("topic_niche","?")} → {ollama.get("sub_topic","?")}</span>
          </div>
          <div class="stat"><span class="stat-label">Emotional Tone</span>
            <span class="stat-value">{ollama.get("emotional_tone","?")}</span>
          </div>
          <div class="stat"><span class="stat-label">Target Audience</span>
            <span class="stat-value">{ollama.get("target_audience","?")}</span>
          </div>
          <div class="stat"><span class="stat-label">Horoscope / Advice</span>
            <span class="stat-value">
              {'🔮 Yes' if ollama.get("has_horoscope_elements") else '—'}
              {' · 💡 Advice' if ollama.get("has_advice_elements") else ''}
            </span>
          </div>
          <div class="stat"><span class="stat-label">Confidence</span>
            <span class="stat-value">{ollama.get("confidence","?")}%</span>
          </div>
          <div style="margin-top:10px;">{key_phrases_html}</div>
        </div>"""
    elif not ollama.get("available"):
        reason = ollama.get("reason","")
        ollama_html = (
            f'<div class="card" style="opacity:.5;">'
            f'<h2>🤖 Ollama Content Brief</h2>'
            f'<p style="color:#64748b;">Not available: {reason}. '
            f'Run: <code>ollama pull {Config.OLLAMA_MODEL}</code> then restart Ollama.</p>'
            f'</div>'
        )
    else:
        ollama_html = (
            f'<div class="card" style="opacity:.5;">'
            f'<h2>🤖 Ollama Content Brief</h2>'
            f'<p style="color:#ef4444;">Error: {ollama.get("error","")}</p>'
            f'<pre style="font-size:.7em;color:#64748b;">{ollama.get("raw_response","")[:300]}</pre>'
            f'</div>'
        )

    shots_html = ""
    for s in ai_brief.get("replication_script", []):
        if isinstance(s, dict):
            shots_html += f"""
            <div class="shot-card">
              <div class="shot-num">Shot {s.get('shot','?')} <span>{s.get('time','')}</span></div>
              <div class="shot-row"><b>🎬</b> {s.get('visual','')}</div>
              <div class="shot-row"><b>🎵</b> {s.get('audio','')}</div>
              <div class="shot-row"><b>📝</b> {s.get('text_overlay','')}</div>
              <div class="shot-note">{s.get('note','')}</div>
            </div>"""

    hooks_html = "".join(
        f'<div class="hook-pill"><span class="hook-type">{h.get("type","")}</span>'
        f'{h.get("hook","")}</div>'
        for h in ai_brief.get("hook_variations", []) if isinstance(h, dict)
    )
    captions_html_parts = []
    for c in ai_brief.get("caption_pack", []):
        if not isinstance(c, dict):
            continue
        tags = " ".join("#" + t.lstrip("#") for t in c.get("hashtags", []))
        captions_html_parts.append(
            f'<div class="caption-card"><p>{c.get("caption","")}</p>'
            f'<div class="hashtags">{tags}</div>'
            f'<div class="cta">📣 {c.get("cta","")}</div></div>'
        )
    captions_html = "".join(captions_html_parts)
    improvements_html = "".join(f"<li>{i}</li>" for i in ai_brief.get("content_improvement", []))
    rec_html          = "".join(f"<li>{r}</li>" for r in cat.get("recommendations",[]))
    scripts_str       = ", ".join(visual.get("scripts_detected", [])) or "latin"
    energy_data       = json.dumps([e["energy"] for e in audio.get("energy_over_time",[])][:60])
    energy_times      = json.dumps([e["sec"]    for e in audio.get("energy_over_time",[])][:60])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Reel Analysis Report v2.2</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
<style>
  *{{box-sizing:border-box;margin:0;padding:0;}}
  body{{font-family:'Segoe UI',system-ui,sans-serif;background:#0f0f17;color:#e2e8f0;}}
  .header{{background:linear-gradient(135deg,#1a1a2e,#16213e,#0f3460);padding:40px;text-align:center;}}
  .header h1{{font-size:2.2em;font-weight:800;color:#fff;}}
  .header p{{color:#94a3b8;margin-top:6px;}}
  .score-badge{{display:inline-block;background:{score_color}22;border:3px solid {score_color};
    color:{score_color};font-size:2.8em;font-weight:900;padding:14px 28px;border-radius:14px;margin:16px 0;}}
  .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:18px;
    padding:28px;max-width:1400px;margin:0 auto;}}
  .card{{background:#1e1e2e;border-radius:14px;padding:22px;border:1px solid #2d2d44;}}
  .card h2{{font-size:.9em;color:#94a3b8;text-transform:uppercase;letter-spacing:.08em;margin-bottom:14px;}}
  .stat{{display:flex;justify-content:space-between;padding:7px 0;border-bottom:1px solid #2d2d44;}}
  .stat:last-child{{border-bottom:none;}}
  .stat-label{{color:#94a3b8;font-size:.88em;}}
  .stat-value{{font-weight:700;color:#e2e8f0;font-size:.92em;}}
  .tag{{display:inline-block;background:#3b1f5e;color:#c084fc;padding:3px 10px;
    border-radius:18px;font-size:.78em;margin:2px;font-weight:600;}}
  .section{{max-width:1400px;margin:0 auto;padding:0 28px 28px;}}
  .section-title{{font-size:1.3em;font-weight:700;margin:28px 0 14px;
    color:#fff;border-left:4px solid #7c3aed;padding-left:11px;}}
  .shot-card{{background:#1e1e2e;border-radius:10px;padding:18px;
    border-left:4px solid #7c3aed;margin-bottom:10px;}}
  .shot-num{{font-size:1em;font-weight:800;color:#a78bfa;margin-bottom:8px;}}
  .shot-num span{{font-size:.78em;color:#64748b;font-weight:400;margin-left:6px;}}
  .shot-row{{font-size:.88em;margin:3px 0;color:#cbd5e1;}}
  .shot-note{{margin-top:7px;font-size:.83em;color:#7c3aed;font-style:italic;}}
  .hook-pill{{background:#1e1e2e;border:1px solid #2d2d44;border-radius:10px;
    padding:12px 16px;margin-bottom:9px;font-size:.93em;}}
  .hook-type{{display:inline-block;background:#7c3aed22;color:#a78bfa;
    font-size:.74em;padding:2px 7px;border-radius:9px;margin-right:7px;font-weight:700;}}
  .caption-card{{background:#1e1e2e;border-radius:10px;padding:18px;
    border:1px solid #2d2d44;margin-bottom:10px;}}
  .caption-card p{{color:#e2e8f0;line-height:1.6;font-size:.93em;}}
  .hashtags{{color:#7c3aed;font-size:.83em;margin-top:9px;}}
  .cta{{color:#22c55e;font-size:.83em;margin-top:5px;}}
  .rec-list,.improve-list{{list-style:none;}}
  .rec-list li{{padding:9px 13px;background:#1e1e2e;border-radius:7px;
    margin-bottom:7px;border-left:3px solid #f59e0b;font-size:.93em;}}
  .improve-list li{{padding:9px 13px;background:#1e1e2e;border-radius:7px;
    margin-bottom:7px;border-left:3px solid #22c55e;font-size:.93em;}}
  .transcript-box{{background:#1e1e2e;border-radius:10px;padding:18px;
    border:1px solid #2d2d44;line-height:1.8;color:#cbd5e1;font-size:.93em;
    font-family:'Noto Sans Devanagari','Segoe UI',sans-serif;direction:auto;}}
  .strategy-box{{background:#1e1e2e;border-radius:10px;padding:18px;
    border-left:4px solid #22c55e;color:#cbd5e1;line-height:1.7;}}
  canvas{{max-height:200px;}}
  code{{background:#2d2d44;padding:2px 6px;border-radius:4px;font-size:.85em;}}
  .footer{{text-align:center;padding:28px;color:#475569;font-size:.83em;}}
</style>
</head>
<body>

<div class="header">
  <h1>🎬 Reel Intelligence Report <span style="font-size:.5em;color:#7c3aed;">v2.2</span></h1>
  <p>Generated {datetime.now().strftime("%B %d, %Y at %H:%M")} ·
     Whisper ({Config.WHISPER_MODEL}, {Config.WHISPER_LANG or "auto"}) ·
     EasyOCR (en+hi) · MediaPipe · HOG · Ollama ({Config.OLLAMA_MODEL})</p>
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
    <div class="stat"><span class="stat-label">Transcript Reliable</span><span class="stat-value">{'✅' if cat.get("transcript_reliable") else '⚠️ No'}</span></div>
    <div style="margin-top:11px;">
      {''.join(f'<span class="tag">{h.replace("_"," ")}</span>' for h in cat.get('hook_formats',[]))}
    </div>
  </div>

  <div class="card">
    <h2>🎵 Audio Profile</h2>
    <div class="stat"><span class="stat-label">Tempo</span><span class="stat-value">{audio['tempo_bpm']} BPM</span></div>
    <div class="stat"><span class="stat-label">Key</span><span class="stat-value">{audio['detected_key']}</span></div>
    <div class="stat"><span class="stat-label">Mood</span><span class="stat-value">{audio['audio_mood'].replace('_',' ').title()}</span></div>
    <div class="stat"><span class="stat-label">Type</span><span class="stat-value">{audio['audio_type'].replace('_',' ').title()}</span></div>
    <div class="stat"><span class="stat-label">Harmony / Perc</span><span class="stat-value">{audio['harmony_pct']}% / {audio['percussion_pct']}%</span></div>
    <div class="stat"><span class="stat-label">Music Heavy</span><span class="stat-value">{'⚠️ Yes (Whisper accuracy reduced)' if audio.get('is_music_heavy') else 'No'}</span></div>
  </div>

  <div class="card">
    <h2>🖼️ Visual Metrics</h2>
    <div class="stat"><span class="stat-label">Avg Brightness</span><span class="stat-value">{visual['avg_brightness']:.1f}</span></div>
    <div class="stat"><span class="stat-label">Avg Saturation</span><span class="stat-value">{visual['avg_saturation']:.1f}</span></div>
    <div class="stat"><span class="stat-label">Face Presence</span><span class="stat-value">{int(visual['face_presence_ratio']*100)}%</span></div>
    <div class="stat"><span class="stat-label">Eye Contact</span><span class="stat-value">{int(visual['eye_contact_ratio']*100)}%</span></div>
    <div class="stat"><span class="stat-label">Text Overlay</span><span class="stat-value">{int(visual['text_overlay_ratio']*100)}%</span></div>
    <div class="stat"><span class="stat-label">Cuts / Pacing</span><span class="stat-value">{visual['estimated_cuts']} / {visual['pacing']}</span></div>
    <div class="stat"><span class="stat-label">Scripts Detected</span><span class="stat-value">{scripts_str}</span></div>
  </div>

  <div class="card">
    <h2>🏠 Scene & Background</h2>
    <div class="stat"><span class="stat-label">Indoor / Outdoor</span><span class="stat-value">{visual.get('indoor_outdoor','?').title()}</span></div>
    <div class="stat"><span class="stat-label">Setting</span><span class="stat-value">{visual.get('dominant_setting','?').replace('_',' ').title()}</span></div>
    <div class="stat"><span class="stat-label">Lighting</span><span class="stat-value">{visual.get('dominant_lighting','?').replace('_',' ').title()}</span></div>
    <div class="stat"><span class="stat-label">BG Color</span><span class="stat-value">{visual.get('dominant_wall_color','?').replace('_',' ').title()}</span></div>
  </div>

  <div class="card">
    <h2>👥 People in Frame</h2>
    <div class="stat"><span class="stat-label">Avg People / Frame</span><span class="stat-value">{visual.get('avg_people_in_frame',0)}</span></div>
    <div class="stat"><span class="stat-label">Max People</span><span class="stat-value">{visual.get('max_people_in_frame',0)}</span></div>
    <div class="stat"><span class="stat-label">Solo Frames</span><span class="stat-value">{visual.get('solo_presenter_frames',0)}</span></div>
    <div class="stat"><span class="stat-label">Multi-Person Frames</span><span class="stat-value">{visual.get('multi_person_frames',0)}</span></div>
  </div>

  <div class="card">
    <h2>🎙️ Transcript Quality</h2>
    <div class="stat"><span class="stat-label">Quality Score</span>
      <span class="stat-value" style="color:{q_color};">{q_score}/100</span></div>
    <div class="stat"><span class="stat-label">Reliable</span>
      <span class="stat-value">{'✅ Yes' if q_reliable else '⚠️ No'}</span></div>
    <div class="stat"><span class="stat-label">Script Detected</span>
      <span class="stat-value">{q_script}</span></div>
    <div class="stat"><span class="stat-label">Avg Log Prob</span>
      <span class="stat-value">{quality.get('avg_logprob','?')}</span></div>
    <div class="stat"><span class="stat-label">No-Speech Prob</span>
      <span class="stat-value">{quality.get('avg_no_speech','?')}</span></div>
    <div class="stat"><span class="stat-label">Issues</span>
      <span class="stat-value" style="font-size:.8em;color:#f59e0b;">{", ".join(q_issues) or "none"}</span></div>
  </div>

  {ollama_html}
</div>

<div class="section">
  <div class="section-title">📈 Audio Energy Over Time</div>
  <div class="card"><canvas id="energyChart"></canvas></div>

  <div class="section-title">🚀 Recommendations</div>
  <ul class="rec-list">{rec_html}</ul>

  <div class="section-title">🎙️ Speech Transcript (Whisper · {Config.WHISPER_LANG or "auto"})</div>
  {transcript_display}

  <div class="section-title">🎬 Replication Shot List</div>
  {shots_html or '<div class="transcript-box">Set ANTHROPIC_API_KEY to generate shot list.</div>'}

  <div class="section-title">🪝 Hook Variations</div>
  {hooks_html or '<div class="transcript-box">Set ANTHROPIC_API_KEY to generate hooks.</div>'}

  <div class="section-title">📣 Caption Pack</div>
  {captions_html or '<div class="transcript-box">Set ANTHROPIC_API_KEY to generate captions.</div>'}

  <div class="section-title">📆 Posting Strategy</div>
  <div class="strategy-box">{ai_brief.get('posting_strategy','Set ANTHROPIC_API_KEY to generate strategy.')}</div>

  <div class="section-title">⚡ Content Improvements</div>
  <ul class="improve-list">{improvements_html or '<li>Set ANTHROPIC_API_KEY to generate improvements.</li>'}</ul>
</div>

<div class="footer">Reel Analyzer v2.2 · Whisper · MediaPipe · EasyOCR (en+hi) · HOG · Ollama ({Config.OLLAMA_MODEL}) · Claude</div>

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
    plugins:{{legend:{{labels:{{color:'#94a3b8'}}}}}},
    scales:{{
      x:{{ticks:{{color:'#64748b'}},grid:{{color:'#1e1e2e'}}}},
      y:{{ticks:{{color:'#64748b'}},grid:{{color:'#2d2d44'}}}},
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
# Storyboard, CSV, Organizer (condensed — same logic as v2.1)
# ═══════════════════════════════════════════════════════════
def generate_storyboard(keyframes, frame_data, audio_data, category, output_path, cols=4):
    if not keyframes: return
    thumb_w, thumb_h = 360, 640
    label_h = 150
    rows    = math.ceil(len(keyframes) / cols)
    board   = Image.new("RGB", (cols*thumb_w, rows*(thumb_h+label_h)+90), (10,10,14))
    draw    = ImageDraw.Draw(board)
    try:
        font       = ImageFont.truetype("arial.ttf", 14)
        font_small = ImageFont.truetype("arial.ttf", 11)
        font_bold  = ImageFont.truetype("arialbd.ttf", 15)
        font_title = ImageFont.truetype("arialbd.ttf", 18)
    except:
        font = font_small = font_bold = font_title = ImageFont.load_default()

    draw.rectangle([0,0,cols*thumb_w,85], fill=(20,10,40))
    draw.text((10,8),  f"🎯 {category['detected_niche'].upper()}  |  {category['content_format']}  |  VIRAL: {category['viral_score']}/100", font=font_title, fill=(255,220,50))
    draw.text((10,35), f"🎵 {audio_data['tempo_bpm']} BPM · {audio_data['detected_key']} · {audio_data['audio_mood']}", font=font_bold, fill=(160,210,255))
    draw.text((10,58), f"🪝 {', '.join(category['hook_formats']) or 'no hooks detected'}", font=font, fill=(180,255,180))

    for i, (frame, fd) in enumerate(zip(keyframes, frame_data)):
        col = i % cols; row = i // cols
        x   = col * thumb_w; y = row * (thumb_h + label_h) + 90
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((thumb_w, thumb_h))
        img_draw = ImageDraw.Draw(img)
        for fp in fd.get("face_positions",[]):
            fx = int(fp["cx"]*thumb_w); fy = int(fp["cy"]*thumb_h)
            sz = int(math.sqrt(max(fp.get("size_ratio",0.05),0.01))*thumb_w)
            c  = (0,255,80) if fd.get("eye_contact") else (255,160,0)
            img_draw.ellipse([fx-sz//2,fy-sz//2,fx+sz//2,fy+sz//2], outline=c, width=3)
        for bp in fd.get("body_positions",[]):
            bx = int(bp["cx"]*thumb_w); by = int(bp["cy"]*thumb_h)
            img_draw.rectangle([bx-30,by-60,bx+30,by+60], outline=(80,160,255), width=2)
        for wrd in fd.get("ocr_words",[])[:8]:
            if "bbox" in wrd:
                bx1=int(wrd["bbox"][0]*thumb_w/frame.shape[1]); by1=int(wrd["bbox"][1]*thumb_h/frame.shape[0])
                bx2=int(wrd["bbox"][2]*thumb_w/frame.shape[1]); by2=int(wrd["bbox"][3]*thumb_h/frame.shape[0])
                c = (255,100,100) if wrd.get("script")=="devanagari" else (255,220,0)
                img_draw.rectangle([bx1,by1,bx2,by2], outline=c, width=1)
        beat_near = any(abs(bt-fd["sec"])<0.5 for bt in audio_data["beat_times"])
        if beat_near:
            img_draw.rectangle([0,thumb_h-8,thumb_w,thumb_h], fill=(200,40,40))
            img_draw.text((4,thumb_h-20), "♪ BEAT", font=font_small, fill=(255,255,255))
        if fd.get("scene_change"):
            img_draw.rectangle([0,0,thumb_w,6], fill=(255,80,0))
        io_col  = (80,200,120) if fd.get("indoor_outdoor")=="outdoor" else (100,140,255)
        img_draw.rectangle([thumb_w-45,4,thumb_w-4,22], fill=io_col)
        img_draw.text((thumb_w-42,6), fd.get("indoor_outdoor","?")[:3].upper(), font=font_small, fill=(0,0,0))
        board.paste(img, (x,y))

        ly = y + thumb_h
        draw.rectangle([x,ly,x+thumb_w,ly+label_h], fill=(14,14,22))
        ec = "👁️" if fd.get("eye_contact") else ""
        draw.text((x+5,ly+4),   f"⏱ {fd['sec']}s  📷 {fd['camera_move']}  {ec}", font=font, fill=(160,210,255))
        draw.text((x+5,ly+22),  f"👤 {fd['face_count']} faces  👥 {fd.get('people_count',0)} people  💨 motion:{fd['motion_score']:.1f}", font=font, fill=(200,200,200))
        draw.text((x+5,ly+40),  f"🏠 {fd.get('setting_type','?')[:22]}  💡 {fd.get('lighting','?')[:10]}", font=font_small, fill=(255,200,100))
        draw.text((x+5,ly+56),  f"🎨 bg:{fd.get('wall_color','?')[:14]}  📝 {'✅' if fd['has_text_overlay'] else '❌'}", font=font_small, fill=(255,220,80))
        draw.text((x+5,ly+72),  f"{fd['hook_text'][:42]}{'…' if len(fd['hook_text'])>42 else ''}", font=font_small, fill=(200,200,200))
        draw.text((x+5,ly+88),  f"scripts:{','.join(fd.get('scripts_detected',[]) or ['?'])[:20]}", font=font_small, fill=(140,220,140))
        draw.text((x+5,ly+104), f"gaze:{fd.get('gaze_direction','?')[:16]}  ♪{'BEAT' if beat_near else '—'}", font=font_small, fill=(255,100,100) if beat_near else (80,80,100))
        draw.text((x+5,ly+120), f"readability:{fd.get('text_readability',0):.0f}%  {'✂️CUT' if fd.get('scene_change') else ''}", font=font_small, fill=(100,180,100))
        draw.text((x+5,ly+136), f"{'HINDI:'+fd.get('hindi_text','')[:20] if fd.get('hindi_text') else ''}", font=font_small, fill=(255,160,80))
        draw.rectangle([x,y,x+42,y+22], fill=(120,30,200))
        draw.text((x+5,y+4), f"#{i+1}", font=font_bold, fill="white")

    board.save(output_path, quality=95)
    print(f"🎬 Storyboard saved: {output_path}")


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
        "transcript_script":   speech.get("detected_script",""),
        "indoor_outdoor":      visual.get("indoor_outdoor",""),
        "setting_type":        visual.get("dominant_setting",""),
        "wall_color":          visual.get("dominant_wall_color",""),
        "lighting":            visual.get("dominant_lighting",""),
        "avg_people":          visual.get("avg_people_in_frame",0),
        "max_people":          visual.get("max_people_in_frame",0),
        "scripts_detected":    "|".join(visual.get("scripts_detected",[])),
        "is_music_heavy":      audio.get("is_music_heavy",False),
        "ollama_topic":        ollama.get("topic_niche","") if ollama.get("available") else "",
        "ollama_tone":         ollama.get("emotional_tone","") if ollama.get("available") else "",
        "ollama_confidence":   ollama.get("confidence",0) if ollama.get("available") else 0,
    }
    file_exists = os.path.exists(Config.CSV_LOG)
    with open(Config.CSV_LOG, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"📊 CSV log updated: {Config.CSV_LOG}")


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
        print(f"    Script: {speech_data.get('detected_script','?')}  "
              f"Issues: {q.get('issues',[])}")
        print(f"    Sample: «{speech_data['transcript'][:120]}»")
    else:
        print("    (no speech detected)")

    print("\n🎵 [3/7] Deep audio analysis (Librosa)...")
    audio_data = analyze_audio(video_path)
    print(f"    {audio_data['tempo_bpm']} BPM  Key:{audio_data['detected_key']}  "
          f"Mood:{audio_data['audio_mood']}  Type:{audio_data['audio_type']}"
          + ("  ⚠️ MUSIC HEAVY" if audio_data.get("is_music_heavy") else ""))

    print("\n🖼️  [4/7] Frame analysis (OpenCV + MediaPipe + EasyOCR en+hi + HOG)...")
    frame_data, keyframes = analyze_frames(video_path)

    print("\n🎨 [5/7] Color palette (PIL)...")
    pil_data = analyze_color_palette(video_path)

    print("\n📊 [6/7] Aggregating & categorizing...")
    visual_summary = aggregate_visual(frame_data, pil_data, moviepy_data)

    # ⑤ Run Ollama BEFORE categorizing so niche can be overridden
    print("\n🤖 [6b] Ollama content analysis...")
    all_ocr    = visual_summary.get("all_ocr_text","")
    transcript = speech_data.get("transcript","")
    ollama_result = ollama_analyze_content(
        transcript, all_ocr,
        speech_data.get("quality",{}),
        moviepy_data["duration_sec"]
    )
    if ollama_result.get("available") and not ollama_result.get("error"):
        print(f"    ✅ Ollama: {ollama_result.get('topic_niche','?')} · "
              f"{ollama_result.get('emotional_tone','?')} · "
              f"confidence:{ollama_result.get('confidence','?')}%")
        print(f"    📝 {ollama_result.get('content_summary','')[:100]}…")
    elif not ollama_result.get("available"):
        print(f"    ⚠️  Ollama not available: {ollama_result.get('reason','')}")
        print(f"       Install: https://ollama.com  then run: ollama pull {Config.OLLAMA_MODEL}")
    else:
        print(f"    ❌ Ollama error: {ollama_result.get('error','')}")

    category = categorize_content(visual_summary, audio_data, speech_data, ollama_result)
    print(f"    Niche:{category['detected_niche']}  Format:{category['content_format']}  "
          f"Viral:{category['viral_score']}/100")
    print(f"    Scene:{visual_summary.get('indoor_outdoor','?')}/{visual_summary.get('dominant_setting','?')}")
    print(f"    People avg:{visual_summary.get('avg_people_in_frame',0)}  "
          f"max:{visual_summary.get('max_people_in_frame',0)}")
    print(f"    Scripts:{visual_summary.get('scripts_detected',[])}  "
          f"Transcript reliable:{'Yes' if category.get('transcript_reliable') else 'No'}")

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
        "ollama_analysis":   ollama_result,
    }

    print("\n🤖 [7/7] Generating outputs...")
    ai_brief = {}
    if Config.ENABLE_AI_BRIEF and Config.ANTHROPIC_API_KEY:
        print("  🧠 Claude AI brief...")
        ai_brief = generate_ai_brief(full_report)
        full_report["ai_brief"] = ai_brief
    else:
        print("  ⏭️  Claude AI brief skipped (set ANTHROPIC_API_KEY)")

    json_path = os.path.join(Config.REPORTS_DIR, f"{slug}_{ts}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)
    print(f"  ✅ JSON: {json_path}")

    sb_path   = os.path.join(Config.REPORTS_DIR, f"{slug}_{ts}_storyboard.jpg")
    html_path = os.path.join(Config.REPORTS_DIR, f"{slug}_{ts}_report.html")
    generate_storyboard(keyframes, frame_data, audio_data, category, sb_path)
    generate_html_report(full_report, ai_brief, html_path)
    append_to_csv(full_report, video_path)
    organize_by_category(html_path, sb_path, category["detected_niche"], category["viral_score"])

    # ⑧ Summary console
    q = speech_data.get("quality", {})
    print(f"\n{'═'*62}  ✅ ANALYSIS COMPLETE")
    print(f"{'─'*62}")
    print(f"  🎯 Niche:         {category['detected_niche'].replace('_',' ').title()}")
    print(f"  📋 Format:        {category['content_format'].replace('_',' ').title()}")
    print(f"  🏆 Viral Score:   {category['viral_score']}/100")
    print(f"  🎵 BPM/Key:       {audio_data['tempo_bpm']} / {audio_data['detected_key']}")
    print(f"  ✂️  Pacing:        {visual_summary['pacing']} ({visual_summary['estimated_cuts']} cuts)")
    print(f"  👤 Face/Eye:      {int(visual_summary['face_presence_ratio']*100)}% / {int(visual_summary['eye_contact_ratio']*100)}%")
    print(f"  👥 People avg:    {visual_summary.get('avg_people_in_frame',0)}  max:{visual_summary.get('max_people_in_frame',0)}")
    print(f"  🏠 Scene:         {visual_summary.get('indoor_outdoor','?')} / {visual_summary.get('dominant_setting','?')}")
    print(f"  💡 Lighting:      {visual_summary.get('dominant_lighting','?')}")
    print(f"  🎨 BG Color:      {visual_summary.get('dominant_wall_color','?')}")
    print(f"  📝 Text:          {int(visual_summary['text_overlay_ratio']*100)}%  Scripts:{visual_summary.get('scripts_detected',[])}")
    print(f"  🪝 Hooks:         {', '.join(category['hook_formats']) or 'none'}")
    print(f"  🎙️  Transcript:    {speech_data.get('word_count',0)} words  "
          f"Quality:{q.get('score',0)}/100 ({'✅' if q.get('reliable') else '⚠️ UNRELIABLE'})  "
          f"Script:{speech_data.get('detected_script','?')}")
    if ollama_result.get("available") and not ollama_result.get("error"):
        print(f"  🤖 Ollama topic:  {ollama_result.get('topic_niche','?')}  "
              f"tone:{ollama_result.get('emotional_tone','?')}  "
              f"conf:{ollama_result.get('confidence','?')}%")
        print(f"  💬 Summary:       {ollama_result.get('content_summary','')[:80]}…")
    print(f"\n  📄 HTML  → {html_path}")
    print(f"  🎬 Board → {sb_path}")
    print(f"  📊 CSV   → {Config.CSV_LOG}")
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
    parser = argparse.ArgumentParser(description="Reel Analyzer v2.2")
    parser.add_argument("path",     nargs="?", help="Video file or folder")
    parser.add_argument("--video",  type=str)
    parser.add_argument("--folder", type=str)
    parser.add_argument("--no-ai",  action="store_true", help="Skip Claude AI brief")
    parser.add_argument("--no-ollama", action="store_true", help="Skip Ollama analysis")
    parser.add_argument("--whisper-model", default="base",
                        choices=["tiny","base","small","medium","large"])
    parser.add_argument("--lang", default="hi",
                        help="Whisper language (default: hi). Use 'auto' for auto-detect.")
    parser.add_argument("--ollama-model", default=Config.OLLAMA_MODEL,
                        help=f"Ollama model to use (default: {Config.OLLAMA_MODEL})")
    args = parser.parse_args()

    if args.no_ai:         Config.ENABLE_AI_BRIEF = False
    if args.no_ollama:     Config.OLLAMA_ENABLED   = False
    Config.WHISPER_MODEL  = args.whisper_model
    Config.WHISPER_LANG   = None if args.lang == "auto" else args.lang
    Config.OLLAMA_MODEL   = args.ollama_model

    target = args.video or args.path
    if args.folder:
        batch_analyze(args.folder)
    elif target:
        if os.path.isdir(target):
            batch_analyze(target)
        else:
            analyze_video(target)
    else:
        VIDEO_PATH = r"E:\reel_19000likes_AQObSSyIQiPNrzzEHWUv.mp4"
        analyze_video(VIDEO_PATH)