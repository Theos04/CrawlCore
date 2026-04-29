"""
REEL ANALYZER v3.1 — ULTRA SMART EDITION (ENHANCED)
Fully merged from v3.0 with new libraries:
  • Surya OCR & DocTR (additional OCR engines)
  • SpeechBrain (emotion, diarization)
  • Demucs (source separation)
  • Pyannote (speaker diarization)
  • SceneDetect (smart scene cuts)
  • Segment Anything (SAM) for object segmentation
All new modules are optional and gracefully degrade if not installed.
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
from collections import defaultdict, Counter
import sys
warnings.filterwarnings("ignore")

# Additional imports for new features
try:
    from scipy.spatial.distance import cosine
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("⚠️  scipy not installed — pip install scipy (needed for anomaly detection)")

# ========== NEW LIBRARY IMPORTS (with graceful fallback) ==========
# --- OCR: Surya ---
SURYA_AVAILABLE = False
try:
    from surya.ocr import run_ocr
    from surya.model.detection.model import load_model as load_det_model
    from surya.model.recognition.model import load_model as load_rec_model
    from surya.model.recognition.processor import load_processor as load_rec_processor
    SURYA_AVAILABLE = True
    print("✅ Surya OCR ready")
except ImportError:
    print("⚠️  Surya OCR not installed — pip install surya-ocr")

# --- OCR: DocTR ---
DOCTR_AVAILABLE = False
try:
    from doctr.io import DocumentFile
    from doctr.models import ocr_predictor
    DOCTR_AVAILABLE = True
    print("✅ DocTR ready")
except ImportError:
    print("⚠️  DocTR not installed — pip install python-doctr")

# --- SpeechBrain ---
SPEECHBRAIN_AVAILABLE = False
try:
    import speechbrain as sb
    from speechbrain.pretrained import SpeakerRecognition, EmotionRecognition
    SPEECHBRAIN_AVAILABLE = True
    print("✅ SpeechBrain ready")
except Exception as e:
    SPEECHBRAIN_AVAILABLE = False
    print(f"⚠️  SpeechBrain not usable: {e} — install/update speechbrain and torchaudio")

# --- Demucs ---
DEMUCS_AVAILABLE = False
try:
    import demucs.api
    DEMUCS_AVAILABLE = True
    print("✅ Demucs ready")
except ImportError:
    print("⚠️  Demucs not installed — pip install demucs")

# --- Pyannote ---
PYANNOTE_AVAILABLE = False
try:
    from pyannote.audio import Pipeline
    PYANNOTE_AVAILABLE = True
    print("✅ Pyannote ready")
except ImportError:
    print("⚠️  Pyannote not installed — pip install pyannote.audio")

# --- SceneDetect ---
SCENEDETECT_AVAILABLE = False
try:
    from scenedetect import VideoManager, SceneManager
    from scenedetect.detectors import ContentDetector
    SCENEDETECT_AVAILABLE = True
    print("✅ SceneDetect ready")
except ImportError:
    print("⚠️  SceneDetect not installed — pip install scenedetect[opencv]")

# --- Segment Anything (SAM) ---
SAM_AVAILABLE = False
try:
    import segment_anything as sam
    from segment_anything import sam_model_registry, SamPredictor
    SAM_AVAILABLE = True
    print("✅ Segment Anything ready")
except ImportError:
    print("⚠️  Segment Anything not installed — pip install git+https://github.com/facebookresearch/segment-anything.git")

# --- ffmpeg-python ---
FFMPEG_PYTHON_AVAILABLE = False
try:
    import ffmpeg
    FFMPEG_PYTHON_AVAILABLE = True
    print("✅ ffmpeg-python ready")
except ImportError:
    print("⚠️  ffmpeg-python not installed — pip install ffmpeg-python")

# ── Existing imports ────────────────────────────────────────────
try:
    import torch
    import torchvision.transforms as transforms
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
    print("✅ PyTorch ready")
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️  PyTorch not installed — pip install torch torchvision")

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    print("✅ YOLOv8 ready")
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️  YOLOv8 not installed — pip install ultralytics")

try:
    import whisper
    WHISPER_AVAILABLE = True
    print("✅ Whisper ready")
except ImportError:
    WHISPER_AVAILABLE = False
    print("⚠️  Whisper not installed — pip install openai-whisper")

try:
    import easyocr
    EASYOCR_AVAILABLE = True
    EASYOCR_READER    = None
    print("✅ EasyOCR ready")
except ImportError:
    EASYOCR_AVAILABLE = False
    print("⚠️  EasyOCR not installed — pip install easyocr")

try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
    PADDLE_READER    = None
    PADDLE_INIT_FAILED = False
    print("✅ PaddleOCR ready")
except Exception as e:
    PADDLE_AVAILABLE   = False
    PADDLE_INIT_FAILED = True
    print(f"⚠️  PaddleOCR not usable: {e}. Falling back to EasyOCR.")

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
    print("✅ Anthropic SDK ready")
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("⚠️  Anthropic SDK not installed — pip install anthropic")

try:
    import tensorflow as tf
    import tensorflow_hub as hub
    YAMNET_AVAILABLE = True
    YAMNET_MODEL = None
    print("✅ TensorFlow / YAMNet ready")
except ImportError:
    YAMNET_AVAILABLE = False
    print("⚠️  TensorFlow not installed — pip install tensorflow tensorflow-hub")

try:
    import clip
    CLIP_AVAILABLE = True
    CLIP_MODEL, CLIP_PREPROCESS = None, None
    print("✅ CLIP ready")
except ImportError:
    CLIP_AVAILABLE = False
    print("⚠️  CLIP not installed — pip install git+https://github.com/openai/CLIP.git")

# ========== MEDIAPIPE GRACEFUL DEGRADATION (FIXED for v0.10.32) ==========
MEDIAPIPE_WORKING = True
try:
    import mediapipe as mp
    # Try standard import first (works for most versions)
    try:
        mp_face_mesh      = mp.solutions.face_mesh
        mp_face_detection = mp.solutions.face_detection
        mp_pose           = mp.solutions.pose
        from mediapipe.framework.formats import detection_pb2
    except AttributeError:
        # Fallback for different MediaPipe versions
        try:
            from mediapipe.python.solutions import face_mesh as mp_face_mesh
            from mediapipe.python.solutions import face_detection as mp_face_detection
            from mediapipe.python.solutions import pose as mp_pose
            from mediapipe.framework.formats import detection_pb2
        except ImportError:
            raise ImportError("MediaPipe solutions module not found")
    print("✅ MediaPipe ready")
except Exception as e:
    MEDIAPIPE_WORKING = False
    print(f"⚠️  MediaPipe not usable: {e}. Face/pose analysis disabled.")
    mp = None
    mp_face_mesh = None
    mp_face_detection = None
    mp_pose = None

# ══════════════════════════════════════════════════════════
# CONFIGURATION (updated with new flags)
# ══════════════════════════════════════════════════════════
class Config:
    OUTPUT_DIR        = "analysis_output"
    AUDIO_DIR         = os.path.join(OUTPUT_DIR, "audio")
    REPORTS_DIR       = os.path.join(OUTPUT_DIR, "reports")
    CACHE_DIR         = os.path.join(OUTPUT_DIR, ".cache")
    TRENDS_LOG        = os.path.join(OUTPUT_DIR, "trends.jsonl")
    CSV_LOG           = os.path.join(OUTPUT_DIR, "all_reels_log.csv")
    SAMPLE_EVERY_SEC  = 0.5
    PIL_SAMPLE_SEC    = 3
    STORYBOARD_COLS   = 4
    WHISPER_MODEL     = "base"
    WHISPER_LANG      = "hi"
    WHISPER_HINDI_PROMPT = "यह एक हिंदी वीडियो है।"
    EASYOCR_LANGS     = ["en", "hi"]
    ANTHROPIC_MODEL   = "claude-3-opus-20240229"
    ENABLE_AI_BRIEF   = True
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    OLLAMA_URLS       = ["http://127.0.0.1:11434", "http://localhost:11434"]
    OLLAMA_MODEL      = "llama3"
    OLLAMA_ENABLED    = True
    OLLAMA_TIMEOUT    = 120
    OLLAMA_MAX_RETRIES= 2
    OLLAMA_RETRY_DELAY= 10
    OLLAMA_KEEP_ALIVE = "10m"
    TRANSCRIPT_QUALITY_THRESHOLD = 40
    ML_ENABLED        = True
    YOLO_CONFIDENCE   = 0.2
    SCENE_CONFIDENCE  = 0.3
    OCR_CONFIDENCE    = 0.2
    # v3.0 toggles
    USE_PADDLE        = True
    USE_YAMNET        = True
    USE_CLIP          = False
    # v3.1 new toggles
    USE_SURYA         = False   # heavy, enable manually
    USE_DOCTR         = False
    USE_SPEECHBRAIN   = False
    USE_DEMUCS        = False
    USE_PYANNOTE      = False
    USE_SCENEDETECT   = True    # better scene detection, enabled by default
    USE_SAM           = False   # heavy segmentation
    USE_FFMPEG_PYTHON = False   # not used yet
    # v3.1+ new feature toggles
    USE_TEMPORAL_ANALYSIS = True   # temporal smoothing and activity tracking
    USE_ANOMALY_DETECTION = False  # requires CLIP embeddings
    USE_WEAK_SUPERVISION = True    # weak supervision label suggestion
    USE_ENSEMBLE_CLASSIFIER = True # ensemble predictions from multiple models
    USE_AUTO_CLIPS = False         # automatic clip extraction (requires ffmpeg-python)
    USE_CONTENT_REPURPOSING = True # multi-platform metadata generation
    USE_TRANSFER_LEARNING = False  # prepare dataset for fine-tuning

# ========== NEW CLASSES FOR v3.1+ IMPROVEMENTS ==========

class TemporalAnalyzer:
    """Temporal consistency and activity smoothing for video frames."""
    def __init__(self, window=5):
        self.window = window

    def smooth_predictions(self, predictions):
        """Majority vote smoothing for categorical predictions."""
        smoothed = []
        for i in range(len(predictions)):
            start = max(0, i - self.window // 2)
            end = min(len(predictions), i + self.window // 2 + 1)
            window = predictions[start:end]
            if all(isinstance(p, str) for p in window):
                most_common = Counter(window).most_common(1)[0][0]
                smoothed.append(most_common)
            else:
                smoothed.append(predictions[i])  # fallback
        return smoothed

    def detect_activity_changes(self, frame_activities):
        """Identify when the primary activity changes (scene breaks)."""
        changes = []
        for i in range(1, len(frame_activities)):
            if frame_activities[i] != frame_activities[i-1]:
                changes.append(i)
        return changes


class VideoAnomalyDetector:
    """Detect frames that deviate from the video's dominant style."""
    def __init__(self, threshold=0.7):
        self.threshold = threshold

    def detect_style_breaks(self, frame_embeddings):
        """Use CLIP embeddings (if available) to find outliers."""
        if not frame_embeddings or not SCIPY_AVAILABLE:
            return []
        try:
            emb = np.array(frame_embeddings)
            if len(emb) == 0:
                return []
            median_emb = np.median(emb, axis=0)
            anomalies = []
            for i, e in enumerate(emb):
                sim = 1 - cosine(e, median_emb)
                if sim < self.threshold:
                    anomalies.append(i)
            return anomalies
        except Exception as e:
            print(f"⚠️  Anomaly detection error: {e}")
            return []


class WeakSupervisionLabeler:
    """Weak supervision approach to suggest content labels with confidence scores."""
    def __init__(self):
        self.label_heuristics = {
            'tutorial': {
                'keywords': ['step', 'guide', 'how to', 'number 1', 'first', 'next', 'tip'],
                'visual': ['text_overlay_ratio > 0.4', 'cuts > 3'],
                'weight': 0.8
            },
            'entertainment': {
                'keywords': ['funny', 'laugh', 'joke', 'hilarious', 'reaction'],
                'visual': ['facial_expressions_varied', 'music_energetic'],
                'weight': 0.7
            },
            'educational': {
                'keywords': ['fact', 'learn', 'explain', 'research', 'study', 'science'],
                'visual': ['text_heavy', 'calm_pace'],
                'weight': 0.75
            },
            'promotional': {
                'keywords': ['shop', 'buy', 'link', 'available', 'order', 'discount'],
                'visual': ['product_visible', 'cta_present'],
                'weight': 0.6
            },
            'motivational': {
                'keywords': ['believe', 'achieve', 'success', 'mindset', 'goal', 'dream'],
                'visual': ['inspirational_imagery', 'uplifting_music'],
                'weight': 0.65
            }
        }

    def suggest_labels(self, report, top_k=3):
        """Suggest labels based on multiple signals (transcript, OCR, audio, visual)."""
        text = (report.get('visual_summary', {}).get('all_ocr_text', '') + ' ' +
                report.get('speech_transcript', {}).get('transcript', '')).lower()
        scores = {}
        for label, h in self.label_heuristics.items():
            score = 0.0
            # keyword matches
            matches = [kw for kw in h['keywords'] if kw in text]
            if matches:
                score += (len(matches) / len(h['keywords'])) * h['weight']
            # visual signals (simplified)
            vis = report.get('visual_summary', {})
            if 'text_overlay_ratio > 0.4' in h['visual'] and vis.get('text_overlay_ratio', 0) > 0.4:
                score += 0.2
            if 'cuts > 3' in h['visual'] and vis.get('estimated_cuts', 0) > 3:
                score += 0.2
            scores[label] = {'score': round(score, 2), 'keywords_matched': matches}
        top = sorted(scores.items(), key=lambda x: x[1]['score'], reverse=True)[:top_k]
        return [{'label': l, 'confidence': s['score'], 'evidence': s['keywords_matched']} for l, s in top]


class LabelCleaner:
    """Co-teaching for noisy labels - filter low-confidence predictions."""
    @staticmethod
    def filter_low_confidence(predictions, threshold=0.7):
        """Filter predictions below confidence threshold."""
        return [p for p in predictions if p.get('confidence', 0) >= threshold]

    @staticmethod
    def detect_disagreements(predictions_dict):
        """Compare predictions from different models for the same frame."""
        disagreements = []
        for frame_idx, preds in predictions_dict.items():
            if isinstance(preds, dict):
                models_agree = len(set(preds.values())) == 1
                if not models_agree:
                    disagreements.append(frame_idx)
        return disagreements


class AutoClipGenerator:
    """Automatic clip extraction for TikTok/Reels (hook, peak, CTA)."""
    def __init__(self, video_path):
        self.video_path = video_path

    def identify_clips(self, frame_data, audio_data=None):
        """Identify key moments for clip extraction."""
        if not frame_data:
            return []
        clips = []
        # Hook: first 2 seconds
        clips.append({
            'name': 'hook',
            'start': 0.0,
            'end': min(2.0, frame_data[-1]['sec']),
            'score': 0.95
        })
        # Peak moment: highest aesthetic score
        peak = max(frame_data, key=lambda f: f.get('aesthetic_score', 0))
        start = max(0, peak['sec'] - 1.5)
        end = min(frame_data[-1]['sec'], peak['sec'] + 1.5)
        clips.append({
            'name': 'peak',
            'start': start,
            'end': end,
            'score': peak.get('aesthetic_score', 50) / 100.0
        })
        # CTA: last 5 seconds
        dur = frame_data[-1]['sec']
        clips.append({
            'name': 'cta',
            'start': max(0, dur - 5),
            'end': dur,
            'score': 0.7
        })
        return clips

    def extract_clips(self, clips, output_dir):
        """Extract clips using ffmpeg-python."""
        if not FFMPEG_PYTHON_AVAILABLE:
            print("⚠️  ffmpeg-python not available — clips not extracted")
            return []
        os.makedirs(output_dir, exist_ok=True)
        extracted = []
        for clip in clips:
            out = os.path.join(output_dir, f"{clip['name']}.mp4")
            try:
                stream = ffmpeg.input(self.video_path, ss=clip['start'], t=clip['end'] - clip['start'])
                stream = ffmpeg.output(stream, out, vcodec='libx264', acodec='aac', **{'shortest': None})
                ffmpeg.run(stream, overwrite_output=True, quiet=True)
                extracted.append(out)
            except Exception as e:
                print(f"⚠️  Clip extraction failed for {clip['name']}: {e}")
        return extracted


class ContentRepurposingEngine:
    """Generate multi-platform metadata for content distribution."""
    @staticmethod
    def generate_packages(report, clips=None):
        """Generate platform-specific packages."""
        niche = report.get('content_category', {}).get('detected_niche', 'general')
        packages = {
            'tiktok': {
                'duration': (8, 15),
                'aspect': '9:16',
                'clips': clips[:1] if clips else [],
                'hashtags': [f"#{niche.replace('_', '')}", '#foryou', '#viral']
            },
            'instagram': {
                'duration': (15, 90),
                'aspect': '9:16',
                'clips': clips if clips else [],
                'hashtags': [f"#{niche.replace('_', '')}", '#reels']
            },
            'youtube_shorts': {
                'duration': (15, 60),
                'aspect': '9:16',
                'clips': clips if clips else [],
                'hashtags': [f"#{niche.replace('_', '')}", '#shorts']
            },
            'linkedin': {
                'duration': (30, 120),
                'aspect': '16:9',
                'clips': None,  # professional only
                'hashtags': [f"#{niche.replace('_', '')}", '#professional']
            }
        }
        # Add scheduling recommendations
        schedule_map = {
            'astrology': {'days': ['Mon', 'Wed'], 'times': ['7am', '7pm']},
            'fitness': {'days': ['Mon', 'Thu', 'Fri'], 'times': ['6am', '6pm']},
            'food': {'days': ['Wed', 'Sat', 'Sun'], 'times': ['12pm', '6pm']},
            'motivation': {'days': ['Mon', 'Wed', 'Fri'], 'times': ['8am', '8pm']},
            'education': {'days': ['Tue', 'Thu'], 'times': ['10am', '7pm']},
            'entertainment': {'days': ['Fri', 'Sat', 'Sun'], 'times': ['7pm', '10pm']},
        }
        niche_lower = niche.lower()
        for niche_key, schedule in schedule_map.items():
            if niche_key in niche_lower:
                packages['schedule'] = schedule
                break
        return packages


class EnsembleClassifier:
    """Combine predictions from multiple models via weighted voting."""
    def __init__(self, weights=None):
        self.weights = weights or {'yolo': 0.4, 'places365': 0.35, 'clip': 0.25}

    def predict(self, predictions_dict):
        """Combine predictions from multiple models."""
        final = {}
        for model, pred in predictions_dict.items():
            if model not in self.weights:
                continue
            if isinstance(pred, dict):
                label = pred.get('class', pred.get('scene_type', 'unknown'))
                conf = pred.get('confidence', 0)
            else:
                label = str(pred)
                conf = 0.5  # default confidence for string predictions
            if label not in final:
                final[label] = 0
            final[label] += conf * self.weights[model]
        if not final:
            return {'class': 'unknown', 'confidence': 0}
        best = max(final.items(), key=lambda x: x[1])
        return {'class': best[0], 'confidence': best[1]}


class TransferLearningOptimizer:
    """Prepare dataset for transfer learning/fine-tuning."""
    @staticmethod
    def prepare_dataset(frame_data, output_dir='finetune_data'):
        """Organize frames into train/val structure for fine-tuning."""
        import shutil
        os.makedirs(output_dir, exist_ok=True)
        categories = {}
        for fd in frame_data:
            cat = fd.get('scene_category', fd.get('scene_type', 'unknown'))
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(fd)
        for cat, items in categories.items():
            train_dir = os.path.join(output_dir, 'train', cat)
            val_dir = os.path.join(output_dir, 'val', cat)
            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(val_dir, exist_ok=True)
            split = int(len(items) * 0.8)
            # Note: frame_data doesn't store images by default
            # This is a placeholder - you'd need to save frames during analysis
            print(f"📁 Category '{cat}': {len(items)} frames (train: {split}, val: {len(items)-split})")
        print(f"✅ Dataset structure ready at {output_dir}")
        return output_dir

for d in [Config.OUTPUT_DIR, Config.AUDIO_DIR, Config.REPORTS_DIR, Config.CACHE_DIR]:
    os.makedirs(d, exist_ok=True)

# ========== CACHE HELPERS ==========
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

# ══════════════════════════════════════════════════════════
# v3.0 — TREND STORAGE
# ══════════════════════════════════════════════════════════
def log_trends(full_report: dict):
    """Append key trend indicators to a JSON lines file."""
    try:
        entry = {
            "timestamp":        full_report["analyzed_at"],
            "filename":         os.path.basename(full_report["file"]),
            "duration":         full_report["container"]["duration_sec"],
            "niche":            full_report["content_category"]["detected_niche"],
            "viral_score":      full_report["content_category"]["viral_score"],
            "audio_events":     full_report["audio_analysis"].get("audio_events", []),
            "all_ocr_text":     full_report["visual_summary"].get("all_ocr_text", ""),
            "transcript":       full_report["speech_transcript"].get("transcript", ""),
            "dominant_activity":full_report["visual_summary"].get("dominant_activity", ""),
            "dominant_scene":   full_report["visual_summary"].get("dominant_scene_type", ""),
            "clip_meme":        full_report.get("clip_meme", None),
        }
        with open(Config.TRENDS_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"⚠️  Failed to log trends: {e}")

def analyze_trends():
    """Read trends.jsonl and print top recurring patterns."""
    if not os.path.exists(Config.TRENDS_LOG):
        print("No trends log found.")
        return
    niches, audio_events, phrases, activities, scenes = [], [], [], [], []
    with open(Config.TRENDS_LOG, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                niches.append(entry["niche"])
                activities.append(entry["dominant_activity"])
                scenes.append(entry["dominant_scene"])
                for ev in entry.get("audio_events", []):
                    audio_events.append(ev["class"])
                for word in entry.get("all_ocr_text", "").split():
                    if len(word) > 3:
                        phrases.append(word.lower())
            except:
                continue
    print("\n📈 TREND ANALYSIS")
    print("Top Niches:",       Counter(niches).most_common(5))
    print("Top Activities:",   Counter(activities).most_common(5))
    print("Top Scenes:",       Counter(scenes).most_common(5))
    print("Top Audio Events:", Counter(audio_events).most_common(10))
    print("Top OCR Phrases:",  Counter(phrases).most_common(10))

# ══════════════════════════════════════════════════════════
# v3.0 — ENHANCED OCR  (FIX 2: try/except in get_paddle_reader)
# ══════════════════════════════════════════════════════════
_PADDLE_INIT_FAILED = False   # permanent flag — set True once, NEVER retried again

def get_paddle_reader():
    """
    Return PaddleOCR instance, or None if unavailable/broken.
    After the FIRST failure (for any reason), _PADDLE_INIT_FAILED is set True
    and this function immediately returns None forever — no per-frame retries.

    Newer PaddleOCR versions raise 'Unknown argument: show_log' as a non-TypeError
    exception, so we detect it by checking the error message string rather than
    relying on exception type alone.
    """
    global PADDLE_READER, PADDLE_AVAILABLE, _PADDLE_INIT_FAILED
    if _PADDLE_INIT_FAILED:          # fast-exit — already failed, no retry
        return None
    if PADDLE_READER is not None:    # already initialised successfully
        return PADDLE_READER
    if not PADDLE_AVAILABLE:
        _PADDLE_INIT_FAILED = True
        return None

    # Attempt 1: with show_log=False (older PaddleOCR)
    try:
        PADDLE_READER = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
        return PADDLE_READER
    except Exception as e:
        err = str(e)
        if "show_log" not in err and "Unknown argument" not in err:
            # Real failure, not just an unsupported argument — give up entirely
            print(f"    ⚠️  PaddleOCR init failed: {e}. Permanently using EasyOCR.")
            _PADDLE_INIT_FAILED = True
            PADDLE_AVAILABLE = False
            return None
        # Otherwise: 'show_log' is unsupported in this version → retry without it

    # Attempt 2: without show_log (newer PaddleOCR)
    try:
        PADDLE_READER = PaddleOCR(use_angle_cls=True, lang='en')
        print("    ✅ PaddleOCR initialised (show_log not supported in this version — skipped)")
        return PADDLE_READER
    except Exception as e:
        print(f"    ⚠️  PaddleOCR init failed on 2nd attempt: {e}. Permanently using EasyOCR.")
        _PADDLE_INIT_FAILED = True
        PADDLE_AVAILABLE = False
        return None

def correct_hinglish_ocr(text: str) -> str:
    """Simple post-correction for common Hinglish mis-reads."""
    corrections = {
        'Tlalaadle': 'M kha laadle', 'Mkhar eadle': 'M kha laadle',
        'Mkhalaadle': 'M kha laadle', 'Mtla laadlle': 'M kha laadle',
        'Mkia laadle': 'M kha laadle', 'Alha laadlle': 'M kha laadle',
        'Mkhallaadle': 'M kha laadle', 'Mkha laadle': 'M kha laadle',
        'Tlha laadle': 'M kha laadle', 'Nikha ladle': 'M kha laadle',
        'kia aadle': 'M kha laadle', 'l kha laadle': 'M kha laadle',
        'Mkha Jaadlle': 'M kha laadle', 'Mlha laadle': 'M kha laadle',
        'Lkhalaadle': 'M kha laadle', 'Mkha laade': 'M kha laadle',
    }
    for wrong, correct in corrections.items():
        text = text.replace(wrong, correct)
    return text

def _fallback_easyocr_simple(frame):
    """Return (text, has_text) — simplified version for the frame loop."""
    if not EASYOCR_AVAILABLE:
        return "", False
    global EASYOCR_READER
    if EASYOCR_READER is None:
        EASYOCR_READER = easyocr.Reader(Config.EASYOCR_LANGS, gpu=False)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    results = EASYOCR_READER.readtext(binary, detail=1, paragraph=False)
    texts = [r[1] for r in results if r[2] >= Config.OCR_CONFIDENCE]
    full_text = ' '.join(texts).strip()
    full_text = correct_hinglish_ocr(full_text)
    return full_text, bool(full_text)

# ========== NEW OCR FUNCTIONS (Surya & DocTR) ==========
def get_surya_ocr():
    """Load Surya OCR models (detection, recognition) once."""
    if not SURYA_AVAILABLE or not Config.USE_SURYA:
        return None
    try:
        det_model = load_det_model()
        rec_model = load_rec_model()
        rec_processor = load_rec_processor()
        return det_model, rec_model, rec_processor
    except Exception as e:
        print(f"    ⚠️  Surya OCR init failed: {e}")
        return None

def extract_text_surya(frame):
    """Run Surya OCR on a frame (PIL Image)."""
    models = get_surya_ocr()
    if models is None:
        return "", False
    try:
        det_model, rec_model, rec_processor = models
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # Surya expects list of images
        predictions = run_ocr([image], [image.size], det_model, rec_model, rec_processor)
        if predictions and predictions[0].text_lines:
            texts = [line.text for line in predictions[0].text_lines]
            return " ".join(texts).strip(), True
    except Exception as e:
        print(f"    ⚠️  Surya OCR failed: {e}")
    return "", False

def get_doctr_predictor():
    """Load DocTR predictor."""
    if not DOCTR_AVAILABLE or not Config.USE_DOCTR:
        return None
    try:
        predictor = ocr_predictor(pretrained=True)
        return predictor
    except Exception as e:
        print(f"    ⚠️  DocTR init failed: {e}")
        return None

def extract_text_doctr(frame):
    """Run DocTR on a frame."""
    predictor = get_doctr_predictor()
    if predictor is None:
        return "", False
    try:
        # DocTR expects a list of pages; we can pass the frame as a numpy array
        doc = DocumentFile.from_images(frame)
        result = predictor(doc)
        text = result.render()
        if text.strip():
            return text.strip(), True
    except Exception as e:
        print(f"    ⚠️  DocTR failed: {e}")
    return "", False

def extract_text_enhanced(frame):
    """
    Try multiple OCR engines in order:
      1. PaddleOCR (if enabled and available)
      2. EasyOCR (fallback)
      3. Surya OCR (if enabled and previous failed)
      4. DocTR (if enabled and previous failed)
    Returns (text: str, has_text: bool, used_engine: str)
    """
    # Try Paddle
    if Config.USE_PADDLE and not _PADDLE_INIT_FAILED:
        try:
            reader = get_paddle_reader()
            if reader is not None:
                # Newer PaddleOCR versions don't accept cls parameter in ocr() call
                # use_angle_cls is set during initialization instead
                result = reader.ocr(frame)
                if result and result[0]:
                    texts = [line[1][0] for line in result[0] if line and len(line) > 1]
                    full_text = ' '.join(texts).strip()
                    full_text = correct_hinglish_ocr(full_text)
                    if full_text:
                        return full_text, True, "paddle"
        except Exception as e:
            print(f"    ⚠️  PaddleOCR failed on frame: {e}")

    # Try EasyOCR fallback
    if EASYOCR_AVAILABLE:
        text, has_text = _fallback_easyocr_simple(frame)
        if has_text:
            return text, True, "easyocr"

    # Try Surya if enabled
    if Config.USE_SURYA and SURYA_AVAILABLE:
        text, has_text = extract_text_surya(frame)
        if has_text:
            return text, True, "surya"

    # Try DocTR if enabled
    if Config.USE_DOCTR and DOCTR_AVAILABLE:
        text, has_text = extract_text_doctr(frame)
        if has_text:
            return text, True, "doctr"

    return "", False, "none"

def get_easyocr_reader():
    global EASYOCR_READER
    if EASYOCR_READER is None:
        print("  📝 Loading EasyOCR model (first time only)...")
        EASYOCR_READER = easyocr.Reader(Config.EASYOCR_LANGS, gpu=False)
    return EASYOCR_READER

def preprocess_for_ocr(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def extract_text_easyocr(frame) -> dict:
    """
    Full EasyOCR extraction with word-level detail (used for storyboard labels).
    Only called when needed — not on every frame when PaddleOCR succeeded.
    """
    if not EASYOCR_AVAILABLE:
        return {"text": "", "word_count": 0, "words": [], "has_text": False,
                "readability_score": 0, "text_zones": [], "hook_text": "",
                "scripts_detected": [], "hindi_text": "", "english_text": ""}
    reader = get_easyocr_reader()
    processed = preprocess_for_ocr(frame)
    results = reader.readtext(processed, detail=1, paragraph=False)
    h, w = frame.shape[:2]
    words, zones, full_text = [], [], ""
    hindi_parts, english_parts = [], []
    for (bbox, text, conf) in results:
        if conf < Config.OCR_CONFIDENCE or len(text.strip()) < 1: continue
        x1=int(bbox[0][0]); y1=int(bbox[0][1]); x2=int(bbox[2][0]); y2=int(bbox[2][1])
        cx, cy = (x1+x2)/2/w, (y1+y2)/2/h
        font_size_est = abs(y2-y1)
        script = "latin"
        for ch in text:
            cp = ord(ch)
            if 0x0900 <= cp <= 0x097F:  script = "devanagari"; break
            elif 0x0600 <= cp <= 0x06FF: script = "arabic_urdu"; break
        words.append({"text": text.strip(), "confidence": round(conf,2),
                      "bbox": [x1,y1,x2,y2],
                      "pos_v": "top" if cy<0.33 else "bottom" if cy>0.66 else "middle",
                      "pos_h": "left" if cx<0.33 else "right" if cx>0.66 else "center",
                      "font_size": font_size_est, "is_large": font_size_est>40, "script": script})
        zones.append({"x1":x1,"y1":y1,"x2":x2,"y2":y2,"text":text.strip(),"script":script})
        full_text += " " + text.strip()
        if script == "devanagari": hindi_parts.append(text.strip())
        else: english_parts.append(text.strip())
    full_text = full_text.strip()
    large_words = [w for w in words if w["is_large"]]
    readability = round(len(large_words)/max(len(words),1)*100, 1)
    return {"text": full_text, "word_count": len(full_text.split()), "words": words,
            "has_text": len(full_text)>1, "readability_score": readability,
            "text_zones": zones, "hook_text": full_text[:150],
            "scripts_detected": list({w["script"] for w in words}),
            "hindi_text": " ".join(hindi_parts), "english_text": " ".join(english_parts)}

def _empty_easyocr_detail(ocr_text: str, has_text: bool) -> dict:
    """Return a minimal EasyOCR-shaped dict when we skip the full EasyOCR pass."""
    scripts = []
    for ch in ocr_text:
        cp = ord(ch)
        if 0x0900 <= cp <= 0x097F:   scripts.append("devanagari"); break
        elif 0x0600 <= cp <= 0x06FF: scripts.append("arabic_urdu"); break
    if not scripts: scripts = ["latin"] if ocr_text else []
    hindi  = " ".join(w for w in ocr_text.split() if any(0x0900<=ord(c)<=0x097F for c in w))
    english = " ".join(w for w in ocr_text.split() if all(ord(c)<0x0900 or ord(c)>0x097F for c in w))
    return {"text": ocr_text, "word_count": len(ocr_text.split()), "words": [],
            "has_text": has_text, "readability_score": 80 if has_text else 0,
            "text_zones": [], "hook_text": ocr_text[:150],
            "scripts_detected": scripts, "hindi_text": hindi, "english_text": english}

# ══════════════════════════════════════════════════════════
# v3.0 — YAMNET AUDIO EVENT DETECTION (FIX 3: class_names + CSV fallback)
# ══════════════════════════════════════════════════════════
def get_yamnet_model():
    global YAMNET_MODEL
    if YAMNET_MODEL is None and YAMNET_AVAILABLE:
        YAMNET_MODEL = hub.load('https://tfhub.dev/google/yamnet/1')
    return YAMNET_MODEL

def load_yamnet_class_names_csv() -> list:
    """Download YAMNet class map CSV if needed and return list of class name strings."""
    url = "https://storage.googleapis.com/audioset/yamnet_class_map.csv"
    local_path = os.path.join(Config.CACHE_DIR, "yamnet_class_map.csv")
    if not os.path.exists(local_path):
        try:
            print("    Downloading YAMNet class names...")
            urllib.request.urlretrieve(url, local_path)
        except Exception as e:
            print(f"    ⚠️  Failed to download YAMNet class names: {e}")
            return None
    class_names = []
    try:
        with open(local_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # skip header row
            for row in reader:
                if len(row) >= 3:
                    class_names.append(row[2])   # display_name column
    except Exception as e:
        print(f"    ⚠️  Failed to read YAMNet class names: {e}")
        return None
    return class_names

def classify_audio_events(audio_path: str) -> list:
    if not YAMNET_AVAILABLE or not Config.USE_YAMNET:
        return []
    try:
        model = get_yamnet_model()
        waveform, sr = librosa.load(audio_path, sr=16000, mono=True)
        scores, embeddings, spectrogram = model(waveform)

        # FIX: class_names() may return bytes or strings depending on TF version
        # Also some TF Hub versions don't expose class_names() at all — fall back to CSV
        class_names = None
        if hasattr(model, 'class_names'):
            try:
                raw = model.class_names()
                # Decode bytes → str if necessary
                class_names = [
                    (n.decode('utf-8') if isinstance(n, bytes) else n)
                    for n in raw
                ]
            except Exception:
                class_names = None
        if class_names is None:
            class_names = load_yamnet_class_names_csv()
        if class_names is None:
            print("    ⚠️  YAMNet class names unavailable — skipping event detection")
            return []

        scores_np   = scores.numpy()
        top_classes = np.argmax(scores_np, axis=1)
        unique, counts = np.unique(top_classes, return_counts=True)
        total_frames = len(top_classes)
        events = []
        for idx, cnt in zip(unique, counts):
            if cnt / total_frames > 0.1 and idx < len(class_names):
                events.append({
                    'class':      class_names[idx],
                    'confidence': round(cnt / total_frames, 2)
                })
        return events
    except Exception as e:
        print(f"    ⚠️  YAMNet failed: {e}")
        return []

# ══════════════════════════════════════════════════════════
# v3.0 — CLIP MEME DETECTION
# ══════════════════════════════════════════════════════════
def init_clip():
    global CLIP_MODEL, CLIP_PREPROCESS
    if CLIP_AVAILABLE and Config.USE_CLIP and CLIP_MODEL is None:
        try:
            CLIP_MODEL, CLIP_PREPROCESS = clip.load("ViT-B/32")
        except Exception as e:
            print(f"    ⚠️  CLIP init failed: {e}")

def classify_with_clip(frame):
    if not CLIP_AVAILABLE or not Config.USE_CLIP or CLIP_MODEL is None:
        return None
    try:
        image = CLIP_PREPROCESS(
            Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ).unsqueeze(0)
        with torch.no_grad():
            image_features = CLIP_MODEL.encode_image(image)
        texts = [
            "a person making a deadpan face",
            "ghop ghop drinking sound",
            "m kha laadle text",
            "take this L",
            "accepting defeat humorously"
        ]
        text_tokens   = clip.tokenize(texts)
        text_features = CLIP_MODEL.encode_text(text_tokens)
        similarity    = (image_features @ text_features.T).softmax(dim=-1)
        best_idx      = similarity.argmax().item()
        best_score    = similarity[0][best_idx].item()
        if best_score > 0.5:
            return texts[best_idx]
    except Exception as e:
        print(f"    ⚠️  CLIP failed: {e}")
    return None

# ══════════════════════════════════════════════════════════
# MODULE 1 — CONTAINER METADATA
# ══════════════════════════════════════════════════════════
def analyze_container(path: str) -> dict:
    vc = VideoFileClip(path)
    data = {
        "duration_sec": round(vc.duration, 2),
        "fps":          vc.fps,
        "resolution":   list(vc.size),
        "aspect_ratio": round(vc.size[0] / vc.size[1], 3),
        "total_frames": int(vc.duration * vc.fps),
        "has_audio":    vc.audio is not None,
        "is_vertical":  vc.size[1] > vc.size[0],
        "format_hint":  "reel/story" if vc.size[1] > vc.size[0] else "landscape",
    }
    vc.close()
    return data

# ══════════════════════════════════════════════════════════
# MODULE 2 — WHISPER STT
# ══════════════════════════════════════════════════════════
def _detect_script(text: str) -> str:
    if not text: return "unknown"
    devanagari = arabic = latin = 0
    for ch in text:
        cp = ord(ch)
        if 0x0900 <= cp <= 0x097F:  devanagari += 1
        elif 0x0600 <= cp <= 0x06FF: arabic += 1
        elif (0x0041 <= cp <= 0x007A) or (0x0061 <= cp <= 0x007A): latin += 1
    total = devanagari + arabic + latin
    if total == 0: return "unknown"
    if devanagari / total > 0.3: return "devanagari"
    if arabic / total > 0.3:     return "arabic_urdu"
    if latin / total > 0.3:      return "latin"
    return "mixed"

def _score_transcript_quality(result: dict, duration_sec: float, expected_lang: str) -> dict:
    segments = result.get("segments", [])
    if not segments or not result.get("text", "").strip():
        return {"score": 0, "issues": ["no_speech_detected"], "reliable": False,
                "avg_logprob": -10.0, "avg_no_speech": 1.0, "avg_compression": 1.0,
                "detected_script": "none", "wpm_estimated": 0}
    issues = []
    try:
        avg_logprob = float(np.mean([s.get("avg_logprob", -1.0) for s in segments
                                     if s.get("avg_logprob") is not None]))
    except: avg_logprob = -2.0
    try:
        avg_compression = float(np.mean([s.get("compression_ratio", 1.0) for s in segments
                                         if s.get("compression_ratio") is not None]))
    except: avg_compression = 1.0
    try:
        avg_no_speech = float(np.mean([s.get("no_speech_prob", 0.0) for s in segments
                                       if s.get("no_speech_prob") is not None]))
    except: avg_no_speech = 0.5

    transcript  = result.get("text", "")
    word_count  = len(transcript.split())
    wpm = word_count / max(duration_sec / 60, 0.01) if duration_sec > 0 else 0
    if wpm > 250: issues.append("wpm_too_high")
    if wpm < 10 and duration_sec > 10: issues.append("wpm_too_low")

    detected_script = _detect_script(transcript)
    if expected_lang == "hi" and detected_script == "arabic_urdu": issues.append("wrong_script_urdu")
    elif expected_lang == "hi" and detected_script == "latin":     issues.append("wrong_script_latin")
    if avg_compression > 2.4: issues.append("high_compression_ratio")
    if avg_no_speech > 0.6:   issues.append("mostly_no_speech")
    if avg_logprob < -0.8:    issues.append("low_confidence")

    score = 100
    score += avg_logprob * 30
    score -= avg_no_speech * 40
    if avg_compression > 2.4:                      score -= 20
    if "wpm_too_high" in issues:                   score -= 20
    if "wrong_script_urdu" in issues:              score -= 15
    if "wrong_script_latin" in issues and expected_lang == "hi": score -= 10
    if "low_confidence" in issues:                 score -= 10
    score = max(0, min(100, round(score)))

    reliable = score >= Config.TRANSCRIPT_QUALITY_THRESHOLD and not (
        "mostly_no_speech" in issues or "high_compression_ratio" in issues)
    return {
        "score": score, "issues": issues, "reliable": reliable,
        "avg_logprob": round(avg_logprob, 3) if avg_logprob != -10.0 else -10.0,
        "avg_no_speech": round(avg_no_speech, 3),
        "avg_compression": round(avg_compression, 3),
        "detected_script": detected_script,
        "wpm_estimated": round(wpm, 1),
    }

def transcribe_speech(video_path: str, duration_sec: float = 0.0) -> dict:
    if not WHISPER_AVAILABLE:
        return {"transcript": "", "language": "unknown", "segments": [],
                "word_timestamps": [], "quality": {"score": 0, "reliable": False, "detected_script": "unknown"}}
    lang_tag  = Config.WHISPER_LANG or "auto"
    cache_key = f"whisper_{lang_tag}"
    cached = load_cache(video_path, cache_key)
    if cached:
        print("  ♻️  Whisper — using cache")
        return cached
    try:
        vc = VideoFileClip(video_path)
        audio_path = os.path.join(Config.AUDIO_DIR, "whisper_temp.wav")
        vc.audio.write_audiofile(audio_path, fps=16000, logger=None)
        vc.close()
    except Exception as e:
        print(f"  ❌ Audio extraction failed: {e}")
        return {"transcript": "", "language": "unknown", "segments": [], "word_timestamps": [],
                "quality": {"score": 0, "reliable": False, "detected_script": "unknown"}}
    print(f"  🎙️  Whisper ({Config.WHISPER_MODEL}) — transcribing (lang={lang_tag})...")
    try:
        model = whisper.load_model(Config.WHISPER_MODEL)
        kwargs = {"word_timestamps": True, "verbose": False, "task": "transcribe"}
        if Config.WHISPER_LANG and Config.WHISPER_LANG != "auto":
            kwargs["language"] = Config.WHISPER_LANG
        if Config.WHISPER_LANG == "hi" and Config.WHISPER_HINDI_PROMPT:
            kwargs["initial_prompt"] = Config.WHISPER_HINDI_PROMPT
        result = model.transcribe(audio_path, **kwargs)
    except Exception as e:
        print(f"  ❌ Transcription failed: {e}")
        if os.path.exists(audio_path): os.remove(audio_path)
        return {"transcript": "", "language": "unknown", "segments": [], "word_timestamps": [],
                "quality": {"score": 0, "reliable": False, "detected_script": "unknown"}}

    quality = _score_transcript_quality(result, duration_sec, lang_tag)
    segments, word_ts = [], []
    for seg in result.get("segments", []):
        segments.append({"start": round(seg.get("start", 0), 2), "end": round(seg.get("end", 0), 2),
                         "text": seg.get("text", "").strip(),
                         "avg_logprob": round(seg.get("avg_logprob", -1.0), 3),
                         "no_speech_prob": round(seg.get("no_speech_prob", 0.0), 3),
                         "compression_ratio": round(seg.get("compression_ratio", 1.0), 3)})
        for w in seg.get("words", []):
            word_ts.append({"word": w.get("word", "").strip(),
                            "start": round(w.get("start", 0), 2),
                            "end": round(w.get("end", 0), 2)})
    transcript  = result.get("text", "").strip()
    word_count  = len(transcript.split())
    last_end    = segments[-1]["end"] if segments and segments[-1].get("end", 0) > 0 else 0
    wpm         = round(word_count / max(last_end / 60, 0.01), 1) if last_end > 0 else 0
    if os.path.exists(audio_path): os.remove(audio_path)
    if not quality["reliable"]:
        print(f"  ⚠️  Transcript quality LOW ({quality['score']}/100): {quality['issues']}")
    else:
        print(f"  ✅ Transcript quality: {quality['score']}/100  Script: {quality['detected_script']}")
    data = {"transcript": transcript, "transcript_display": transcript,
            "language": result.get("language", lang_tag), "segments": segments,
            "word_timestamps": word_ts, "word_count": word_count,
            "speaking_rate_wpm": wpm, "quality": quality,
            "detected_script": quality["detected_script"]}
    save_cache(video_path, cache_key, data)
    return data

# ========== NEW AUDIO FUNCTIONS (Demucs, SpeechBrain, Pyannote) ==========

def demucs_separate(audio_path: str) -> dict:
    """
    Use Demucs to separate vocals and accompaniment.
    Returns paths to separated files or None if failed.
    """
    if not DEMUCS_AVAILABLE or not Config.USE_DEMUCS:
        return None
    try:
        separator = demucs.api.Separator()
        # Separate the audio file
        # Demucs returns a tuple: (origin, sources) where sources is a dict
        origin, sources = separator.separate_audio_file(audio_path)
        # Save vocals and accompaniment to temp files
        vocals_path = os.path.join(Config.AUDIO_DIR, "demucs_vocals.wav")
        accompaniment_path = os.path.join(Config.AUDIO_DIR, "demucs_accompaniment.wav")
        # sources is a dict with keys like 'drums', 'bass', 'other', 'vocals'
        # We want 'vocals' and sum of others for accompaniment
        vocals = sources.get('vocals')
        if vocals is not None:
            # Save vocals
            import torchaudio
            torchaudio.save(vocals_path, vocals.cpu(), separator.samplerate)
        # Sum other stems for accompaniment
        others = []
        for stem in ['drums', 'bass', 'other']:
            if stem in sources:
                others.append(sources[stem])
        if others:
            accompaniment = sum(others)
            torchaudio.save(accompaniment_path, accompaniment.cpu(), separator.samplerate)
        else:
            accompaniment_path = None
        return {"vocals": vocals_path, "accompaniment": accompaniment_path}
    except Exception as e:
        print(f"    ⚠️  Demucs separation failed: {e}")
        return None

def speechbrain_emotion(audio_path: str) -> dict:
    """Use SpeechBrain emotion recognition."""
    if not SPEECHBRAIN_AVAILABLE or not Config.USE_SPEECHBRAIN:
        return {}
    try:
        emotion_classifier = EmotionRecognition.from_hparams(
            source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
            savedir=os.path.join(Config.CACHE_DIR, "speechbrain-emotion")
        )
        # EmotionRecognition expects a file path
        out = emotion_classifier.classify_file(audio_path)
        # out is a dictionary with 'emotion' and 'score'
        return {"emotion": out['emotion'], "emotion_confidence": out['score']}
    except Exception as e:
        print(f"    ⚠️  SpeechBrain emotion failed: {e}")
        return {}

def pyannote_diarization(audio_path: str) -> list:
    """Use Pyannote for speaker diarization."""
    if not PYANNOTE_AVAILABLE or not Config.USE_PYANNOTE:
        return []
    try:
        # Load pretrained pipeline
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
        # Apply to audio file
        diarization = pipeline(audio_path)
        # Convert to list of segments with speaker labels
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker
            })
        return segments
    except Exception as e:
        print(f"    ⚠️  Pyannote diarization failed: {e}")
        return []

# ========== MODULE 2B — AUDIO ANALYSIS (v3.0 + new features) ==========
def analyze_audio(video_path: str) -> dict:
    cached = load_cache(video_path, "audio_enhanced_v3_1")
    if cached:
        print("  ♻️  Enhanced audio — using cache")
        return cached
    print("  🎵 Enhanced audio analysis (Librosa + YAMNet + optional Demucs/SpeechBrain/Pyannote)...")
    # Extract audio
    try:
        vc = VideoFileClip(video_path)
        audio_path = os.path.join(Config.AUDIO_DIR, "temp_audio_enhanced.wav")
        vc.audio.write_audiofile(audio_path, fps=22050, logger=None)
        vc.close()
    except Exception as e:
        print(f"  ❌ Audio extraction failed: {e}")
        return _empty_audio()

    # Basic librosa analysis (same as before)
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    # Ensure tempo is a scalar (librosa may return array in some versions)
    tempo = float(np.asarray(tempo).item()) if isinstance(tempo, (np.ndarray, np.generic)) else float(tempo)
    beat_times = librosa.frames_to_time(beats, sr=sr).tolist() if len(beats) > 0 else []
    onset_env  = librosa.onset.onset_strength(y=y, sr=sr)
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spectral_rolloff   = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    rms      = librosa.feature.rms(y=y)[0]
    rms_mean = float(np.mean(rms))
    voice_activity   = np.mean(rms > np.mean(rms) * 0.5) * 100
    voice_confidence = np.mean(spectral_centroids < 3000) * 100
    music_confidence = (len(beats) / (len(y)/sr) * 10) if len(beats) > 0 else 0
    music_confidence = min(music_confidence, 100)
    layers = []
    if voice_confidence > 30 or voice_activity > 20: layers.append("speech")
    if music_confidence > 20:                        layers.append("music")
    if len(beats) > 20:                              layers.append("rhythmic")
    if np.mean(spectral_rolloff) < 4000:             layers.append("bass_heavy")

    # YAMNet events
    audio_events = classify_audio_events(audio_path)

    # Energy over time
    chunk_sec  = 0.5
    chunk_size = int(sr * chunk_sec)
    energy_over_time = []
    for i in range(0, len(y) - chunk_size, chunk_size):
        chunk = y[i:i+chunk_size]
        energy_over_time.append({"sec": round(i/sr, 2),
                                 "energy": round(float(np.sqrt(np.mean(chunk**2))), 4)})

    # Chroma and key detection
    chroma      = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)
    note_names  = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    dominant_note = note_names[int(np.argmax(chroma_mean))] if len(chroma_mean) > 0 else "unknown"
    major_template = np.array([1,0,1,0,1,1,0,1,0,1,0,1], dtype=float)
    minor_template = np.array([1,0,1,1,0,1,0,1,1,0,1,0], dtype=float)
    if len(chroma_mean) == 12:
        major_scores = [np.dot(np.roll(chroma_mean, -i), major_template) for i in range(12)]
        minor_scores = [np.dot(np.roll(chroma_mean, -i), minor_template) for i in range(12)]
        best_major = int(np.argmax(major_scores))
        best_minor = int(np.argmax(minor_scores))
        detected_key = (f"{note_names[best_major]} major"
                        if max(major_scores) >= max(minor_scores)
                        else f"{note_names[best_minor]} minor")
    else:
        detected_key = "unknown"

    harmonic_energy   = float(np.mean(np.abs(y_harmonic)))
    percussive_energy = float(np.mean(np.abs(y_percussive)))
    total_energy      = harmonic_energy + percussive_energy + 1e-9

    result = {
        "tempo_bpm":            round(tempo, 1),
        "detected_key":         detected_key,
        "dominant_note":        dominant_note,
        "layers_detected":      layers,
        "layer_count":          len(layers),
        "primary_layer":        layers[0] if layers else "unknown",
        "voice_confidence":     round(voice_confidence, 1),
        "music_confidence":     round(music_confidence, 1),
        "has_speech":           "speech" in layers,
        "has_music":            "music" in layers,
        "is_music_only":        "music" in layers and "speech" not in layers,
        "is_speech_only":       "speech" in layers and "music" not in layers,
        "is_speech_with_music": "speech" in layers and "music" in layers,
        "beat_strength":        round(float(np.mean(onset_env)), 4) if len(onset_env) > 0 else 0,
        "beat_times":           beat_times[:20],
        "beat_count":           len(beats),
        "harmonic_content":     round(harmonic_energy, 4),
        "percussive_content":   round(percussive_energy, 4),
        "harmony_pct":          round(harmonic_energy/total_energy*100, 1),
        "percussion_pct":       round(percussive_energy/total_energy*100, 1),
        "spectral_centroid_mean": round(float(np.mean(spectral_centroids)), 1),
        "spectral_rolloff_mean":  round(float(np.mean(spectral_rolloff)), 1),
        "audio_complexity":     len(set(layers)),
        "rms_mean":             round(rms_mean, 4),
        "energy_over_time":     energy_over_time,
        "audio_events":         audio_events,
    }

    # --- New audio features ---
    # Demucs source separation
    if Config.USE_DEMUCS and DEMUCS_AVAILABLE:
        sep = demucs_separate(audio_path)
        if sep:
            result["demucs"] = sep

    # SpeechBrain emotion
    if Config.USE_SPEECHBRAIN and SPEECHBRAIN_AVAILABLE:
        emotion = speechbrain_emotion(audio_path)
        if emotion:
            result["emotion"] = emotion

    # Pyannote diarization
    if Config.USE_PYANNOTE and PYANNOTE_AVAILABLE:
        diarization = pyannote_diarization(audio_path)
        if diarization:
            result["speaker_diarization"] = diarization
            result["num_speakers"] = len(set(seg["speaker"] for seg in diarization))

    if os.path.exists(audio_path): os.remove(audio_path)
    save_cache(video_path, "audio_enhanced_v3_1", result)
    return result

def _empty_audio() -> dict:
    return {"tempo_bpm": 0, "detected_key": "unknown", "dominant_note": "unknown",
            "layers_detected": [], "layer_count": 0, "primary_layer": "unknown",
            "voice_confidence": 0, "music_confidence": 0, "has_speech": False,
            "has_music": False, "is_music_only": False, "is_speech_only": False,
            "is_speech_with_music": False, "beat_strength": 0, "beat_times": [],
            "beat_count": 0, "harmonic_content": 0, "percussive_content": 0,
            "harmony_pct": 0, "percussion_pct": 0, "spectral_centroid_mean": 0,
            "spectral_rolloff_mean": 0, "audio_complexity": 0, "rms_mean": 0,
            "energy_over_time": [], "audio_events": []}

# ══════════════════════════════════════════════════════════
# MODULE 2C — OLLAMA CONTENT ANALYSIS
# ══════════════════════════════════════════════════════════
def _check_port_open(host: str, port: int, timeout: float = 2.0) -> bool:
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        if host.startswith("http://"): host = host.replace("http://","").split(":")[0]
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except: return False

def _ollama_available() -> tuple:
    for url in Config.OLLAMA_URLS:
        try:
            host_part = url.replace("http://","").split(":")
            host = host_part[0]; port = int(host_part[1]) if len(host_part) > 1 else 11434
            if not _check_port_open(host, port): continue
            req = urllib.request.Request(f"{url}/api/tags")
            with urllib.request.urlopen(req, timeout=3) as resp:
                if resp.getcode() == 200: return True, url
        except: continue
    return False, None

def _ollama_warmup(working_url: str) -> bool:
    payload = json.dumps({"model": Config.OLLAMA_MODEL, "prompt": "Hi", "stream": False,
                          "options": {"num_predict": 1}, "keep_alive": Config.OLLAMA_KEEP_ALIVE}).encode()
    try:
        req = urllib.request.Request(f"{working_url}/api/generate", data=payload,
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=Config.OLLAMA_TIMEOUT) as resp:
            return bool(json.loads(resp.read().decode()).get("response"))
    except Exception as e:
        print(f"    ⚠️  Ollama warm-up failed: {e}")
        return False

def _clean_json_response(response_text: str) -> str:
    if not response_text: return "{}"
    import re
    fence_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
    if fence_match:
        candidate = fence_match.group(1).strip()
        try: json.loads(candidate); return candidate
        except: pass
    best = ""
    for start_idx, ch in enumerate(response_text):
        if ch != '{': continue
        depth = 0; in_str = False; escape_next = False
        for end_idx in range(start_idx, len(response_text)):
            c = response_text[end_idx]
            if escape_next: escape_next = False; continue
            if c == '\\' and in_str: escape_next = True; continue
            if c == '"': in_str = not in_str; continue
            if in_str: continue
            if c == '{': depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    candidate = response_text[start_idx:end_idx+1]
                    if len(candidate) > len(best): best = candidate
                    break
    return best if best else response_text

def _get_default_ollama_response(reason: str = None) -> dict:
    return {"available": False, "reason": reason or "ollama_unavailable",
            "content_summary": "(Ollama analysis unavailable)", "topic_niche": "unknown",
            "sub_topic": "", "emotional_tone": "unknown", "key_phrases": [],
            "target_audience": "unknown", "content_language": "unknown",
            "secondary_languages": [], "has_horoscope_elements": False,
            "has_advice_elements": False, "has_background_music": False,
            "is_multilingual": False, "confidence": 0}

def ollama_analyze_content(transcript, ocr_text, transcript_quality, audio_layers, duration_sec) -> dict:
    if not Config.OLLAMA_ENABLED: return _get_default_ollama_response("disabled_in_config")
    available, working_url = _ollama_available()
    if not available: return _get_default_ollama_response("ollama_not_running")
    print(f"    🔥 Warming up {Config.OLLAMA_MODEL}...")
    warmup_ok = _ollama_warmup(working_url)
    if not warmup_ok: print("    ⚠️  Warm-up timed out — proceeding anyway")

    audio_context = ""
    if audio_layers:
        audio_context = f"\nAudio composition: {', '.join(audio_layers)}"
        if "speech" in audio_layers and "music" in audio_layers:
            audio_context += " (speech with background music)"
        elif "music" in audio_layers and "speech" not in audio_layers:
            audio_context += " (music only, no speech)"
        elif "speech" in audio_layers:
            audio_context += " (speech only)"

    if transcript_quality.get("reliable", False):
        text_block = f"SPOKEN CONTENT:\n{transcript}\n\nON-SCREEN TEXT:\n{ocr_text}"
    else:
        quality_note = (f"[Note: Speech transcription quality is LOW "
                        f"({transcript_quality.get('score', 0)}/100). "
                        f"Focus more on on-screen text.]")
        text_block = (f"SPOKEN CONTENT (low quality):\n{transcript}\n{quality_note}\n\n"
                      f"ON-SCREEN TEXT:\n{ocr_text}")

    prompt = f"""Analyze this short video ({duration_sec:.0f} seconds) and respond with ONLY a JSON object.{audio_context}

Content:
{text_block}

JSON response (fill in the values, return ONLY the JSON):
{{
  "content_summary": "2-3 sentence plain English summary of what this video is about",
  "topic_niche": "one of: astrology, motivation, comedy, education, finance, beauty, food, travel, relationship, spiritual, news, fitness, tech, relationship, humor",
  "sub_topic": "more specific topic",
  "emotional_tone": "e.g. playful, serious, romantic, inspirational, funny",
  "key_phrases": ["phrase1", "phrase2", "phrase3"],
  "target_audience": "describe the intended audience",
  "content_language": "primary language e.g. Hindi, English, Hinglish",
  "secondary_languages": [],
  "has_horoscope_elements": false,
  "has_advice_elements": false,
  "has_background_music": false,
  "is_multilingual": false,
  "confidence": 75
}}"""

    payload = json.dumps({"model": Config.OLLAMA_MODEL,
                          "system": ("You are a JSON-only API endpoint. "
                                     "Your entire response must be a single valid JSON object. "
                                     "No preamble, no explanation, no markdown fences, no trailing text. "
                                     "Start your response with { and end with }."),
                          "prompt": prompt, "stream": False,
                          "options": {"temperature": 0.1, "num_predict": 250},
                          "keep_alive": Config.OLLAMA_KEEP_ALIVE}).encode()

    for attempt in range(Config.OLLAMA_MAX_RETRIES):
        try:
            print(f"    🤖 Ollama attempt {attempt+1}/{Config.OLLAMA_MAX_RETRIES} (timeout={Config.OLLAMA_TIMEOUT}s)...")
            req = urllib.request.Request(f"{working_url}/api/generate", data=payload,
                                         headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=Config.OLLAMA_TIMEOUT) as resp:
                raw = json.loads(resp.read().decode())
            response_text = raw.get("response", "").strip()
            if not response_text:
                if attempt < Config.OLLAMA_MAX_RETRIES - 1:
                    time.sleep(Config.OLLAMA_RETRY_DELAY); continue
                return _get_default_ollama_response("empty_response")
            cleaned = _clean_json_response(response_text)
            try: parsed = json.loads(cleaned)
            except json.JSONDecodeError:
                if attempt < Config.OLLAMA_MAX_RETRIES - 1:
                    time.sleep(Config.OLLAMA_RETRY_DELAY); continue
                return _get_default_ollama_response("json_parse_failed")
            defaults = {"content_summary": "(No summary)", "topic_niche": "unknown",
                        "sub_topic": "", "emotional_tone": "unknown", "key_phrases": [],
                        "target_audience": "unknown", "content_language": "unknown",
                        "secondary_languages": [], "has_horoscope_elements": False,
                        "has_advice_elements": False,
                        "has_background_music": "music" in (audio_layers or []),
                        "is_multilingual": False, "confidence": 0}
            for field, val in defaults.items():
                if field not in parsed: parsed[field] = val
            parsed["available"]  = True
            parsed["model_used"] = Config.OLLAMA_MODEL
            return parsed
        except urllib.error.URLError as e:
            if attempt < Config.OLLAMA_MAX_RETRIES - 1:
                print(f"    ⚠️  Attempt {attempt+1} failed: {e.reason}, retrying...")
                time.sleep(Config.OLLAMA_RETRY_DELAY)
            else: return _get_default_ollama_response(f"connection_failed: {e.reason}")
        except socket.timeout:
            if attempt < Config.OLLAMA_MAX_RETRIES - 1:
                print(f"    ⚠️  Attempt {attempt+1} timed out, retrying...")
                time.sleep(Config.OLLAMA_RETRY_DELAY)
            else: return _get_default_ollama_response(f"timeout_after_{Config.OLLAMA_TIMEOUT}s")
        except Exception as e:
            if attempt < Config.OLLAMA_MAX_RETRIES - 1:
                time.sleep(Config.OLLAMA_RETRY_DELAY)
            else: return _get_default_ollama_response(str(e))
    return _get_default_ollama_response("max_retries_exceeded")

# ══════════════════════════════════════════════════════════
# MODULE 4B — SCENE BACKGROUND DETECTION
# ══════════════════════════════════════════════════════════
def detect_scene_background(frame, face_boxes: list) -> dict:
    h, w = frame.shape[:2]
    mask = np.ones((h,w), dtype=np.uint8)*255
    for fb in face_boxes:
        cx=int(fb["cx"]*w); cy=int(fb["cy"]*h)
        sz=int(math.sqrt(max(fb.get("size_ratio",0.05),0.01))*w*1.5)
        x1=max(0,cx-sz); y1=max(0,cy-sz); x2=min(w,cx+sz); y2=min(h,cy+sz)
        mask[y1:y2,x1:x2]=0
    bg_pixels = frame[mask>0]
    if len(bg_pixels)<100: bg_pixels = frame.reshape(-1,3)
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    top_zone  = hsv_frame[:h//4,:,:]
    sky_blue  = np.sum((top_zone[:,:,0]>95)&(top_zone[:,:,0]<135)&
                       (top_zone[:,:,1]>40)&(top_zone[:,:,2]>80))
    sky_ratio   = sky_blue/max(top_zone.shape[0]*top_zone.shape[1],1)
    green_px    = np.sum((hsv_frame[:,:,0]>35)&(hsv_frame[:,:,0]<85)&
                         (hsv_frame[:,:,1]>40)&(hsv_frame[:,:,2]>40))
    green_ratio = green_px/(h*w)
    bg_gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bg_masked = cv2.bitwise_and(bg_gray, bg_gray, mask=mask)
    nonzero   = bg_masked[mask>0]
    texture_std = float(np.std(nonzero)) if len(nonzero)>0 else 0
    if len(bg_pixels)>0:
        mean_bgr = bg_pixels.mean(axis=0)
        r,g,b = float(mean_bgr[2]),float(mean_bgr[1]),float(mean_bgr[0])
    else: r,g,b = 128.,128.,128.
    if r>200 and g>200 and b>200:   wall_color="white"
    elif r<60 and g<60 and b<60:    wall_color="black"
    elif r>g+30 and r>b+30:         wall_color="red_warm"
    elif b>r+30 and b>g+20:         wall_color="blue_cool"
    elif g>r+20 and g>b+20:         wall_color="green"
    elif r>160 and g>130 and b<100: wall_color="golden_warm"
    elif r>180 and g>180:           wall_color="bright_yellow"
    else:
        val=int((r+g+b)/3); wall_color=f"grey_{val//50*50}"
    is_outdoor   = sky_ratio>0.08 or green_ratio>0.15
    is_nature    = green_ratio>0.25
    is_studio_flat = texture_std<18 and not is_outdoor
    is_busy_bg   = texture_std>45
    if is_nature and is_outdoor:         setting="outdoor_nature"
    elif is_outdoor and sky_ratio>0.08:  setting="outdoor_street_or_open"
    elif is_outdoor:                     setting="outdoor_other"
    elif is_studio_flat:                 setting="studio_or_plain_wall"
    elif is_busy_bg:                     setting="indoor_busy_background"
    else:                                setting="indoor_room"
    room_type = None
    if not is_outdoor:
        brightness = float(np.mean(nonzero)) if len(nonzero)>0 else 128
        if brightness>200 and texture_std<20:               room_type="studio_white"
        elif wall_color in ["red_warm","golden_warm","bright_yellow"]: room_type="warm_decorated_room"
        elif wall_color=="blue_cool":                        room_type="cool_toned_room"
        elif texture_std>40:                                 room_type="cluttered_or_bookshelf"
        else:                                                room_type="generic_indoor"
    bg_bright = float(np.mean(bg_pixels)) if len(bg_pixels)>0 else 128
    if bg_bright>170:   lighting="bright_well_lit"
    elif bg_bright>100: lighting="normal"
    elif bg_bright>50:  lighting="dim"
    else:               lighting="dark_low_light"
    return {"indoor_outdoor":"outdoor" if is_outdoor else "indoor", "setting_type":setting,
            "room_type":room_type, "wall_color":wall_color, "bg_texture_std":round(texture_std,1),
            "is_flat_bg":is_studio_flat, "is_busy_bg":is_busy_bg,
            "sky_ratio":round(sky_ratio,3), "green_ratio":round(green_ratio,3),
            "lighting":lighting, "bg_mean_rgb":[round(r,1),round(g,1),round(b,1)]}

# ══════════════════════════════════════════════════════════
# MODULE 4C — HOG BODY/PEOPLE COUNTER
# ══════════════════════════════════════════════════════════
_HOG = cv2.HOGDescriptor()
_HOG.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

def count_bodies_hog(frame) -> dict:
    h,w = frame.shape[:2]
    scale = 480/h if h>480 else 1.0
    small = cv2.resize(frame,(int(w*scale),int(h*scale)))
    rects, weights = _HOG.detectMultiScale(small, winStride=(8,8), padding=(4,4), scale=1.05)
    people = []
    for (bx,by,bw,bh) in rects:
        cx=round((bx+bw/2)/small.shape[1],3); cy=round((by+bh/2)/small.shape[0],3)
        people.append({"cx":cx,"cy":cy,
                       "pos_v":"top" if cy<0.33 else "bottom" if cy>0.66 else "center",
                       "pos_h":"left" if cx<0.33 else "right" if cx>0.66 else "center"})
    return {"body_count_hog":len(people),"body_positions":people}

# ══════════════════════════════════════════════════════════
# MODULE 4D — MEDIAPIPE FACE + POSE  (FIX 4: MEDIAPIPE_WORKING guard)
# ══════════════════════════════════════════════════════════
def analyze_face_pose(frame) -> dict:
    """Returns face/pose data. Safely disabled if MEDIAPIPE_WORKING=False."""
    default = {"face_count":0, "faces":[], "body_detected":False,
               "gaze_direction":"unknown", "upper_body_visible":False,
               "eye_contact":False, "hands_raised":False}
    if not MEDIAPIPE_WORKING:
        return default

    h,w = frame.shape[:2]
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = dict(default)  # copy
    try:
        with mp_face_detection.FaceDetection(min_detection_confidence=0.4) as fd:
            det = fd.process(img_rgb)
            if det.detections:
                result["face_count"] = len(det.detections)
                for d in det.detections:
                    bbox = d.location_data.relative_bounding_box
                    cx=round(bbox.xmin+bbox.width/2,3); cy=round(bbox.ymin+bbox.height/2,3)
                    size=round(bbox.width*bbox.height,4); score=round(d.score[0],2)
                    result["faces"].append({"cx":cx,"cy":cy,"size_ratio":size,"confidence":score,
                                            "is_close_up":size>0.15,
                                            "position":"top" if cy<0.33 else "bottom" if cy>0.66 else "center"})
    except Exception as e:
        print(f"    ⚠️  FaceDetection failed: {e}")

    try:
        with mp_face_mesh.FaceMesh(static_image_mode=True,max_num_faces=1,
                                    refine_landmarks=True,min_detection_confidence=0.4) as fm:
            mesh = fm.process(img_rgb)
            if mesh.multi_face_landmarks:
                lm = mesh.multi_face_landmarks[0].landmark
                nose_x=lm[1].x; eye_center_x=(lm[33].x+lm[263].x)/2; diff=nose_x-eye_center_x
                if diff>0.03:    result["gaze_direction"]="looking_right"
                elif diff<-0.03: result["gaze_direction"]="looking_left"
                else:            result["gaze_direction"]="looking_at_camera"
                result["eye_contact"] = result["gaze_direction"]=="looking_at_camera"
    except Exception as e:
        print(f"    ⚠️  FaceMesh failed: {e}")

    try:
        with mp_pose.Pose(static_image_mode=True,min_detection_confidence=0.4) as pose:
            pr = pose.process(img_rgb)
            if pr.pose_landmarks:
                result["body_detected"]=True; result["upper_body_visible"]=True
                lms=pr.pose_landmarks.landmark
                lw=lms[mp_pose.PoseLandmark.LEFT_WRIST]; rw=lms[mp_pose.PoseLandmark.RIGHT_WRIST]
                result["hands_raised"] = lw.y<0.5 or rw.y<0.5
    except Exception as e:
        print(f"    ⚠️  Pose failed: {e}")

    return result

# ══════════════════════════════════════════════════════════
# MODULE 4F — PLACES365 SCENE CLASSIFIER
# ══════════════════════════════════════════════════════════
class Places365Classifier:
    def __init__(self):
        self.model=None; self.transform=None; self.labels=None; self.available=False

    def load_model(self):
        if not TORCH_AVAILABLE: return False
        try:
            print("    Loading Places365 scene classifier...")
            import torch.hub
            torch.hub.set_dir(os.path.join(Config.CACHE_DIR,"torch_hub"))
            self.model = torch.hub.load('zhoubolei/places365','resnet50_places365',pretrained=True,trust_repo=True)
            self.model.eval()
            self.labels = []
            try:
                with open('categories_places365.txt','r') as f:
                    for line in f: self.labels.append(line.strip().split(' ')[0][3:])
            except: self.labels = ['unknown']*365
            self.transform = transforms.Compose([
                transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])])
            self.available = True; return True
        except Exception as e:
            print(f"    ⚠️  Places365 not available (using fallback). Reason: {e}")
            self.available = False; return False

    def classify_scene(self, image):
        if not self.available: return {"scene_type":"unknown","confidence":0,"scene_category":"unknown"}
        try:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            input_tensor = self.transform(pil_img).unsqueeze(0)
            with torch.no_grad():
                logits = self.model(input_tensor)
                probs = F.softmax(logits,dim=1)
            top_probs,top_indices = torch.topk(probs,5)
            top_probs=top_probs[0].tolist(); top_indices=top_indices[0].tolist()
            predictions = [{"scene": self.labels[idx] if idx<len(self.labels) else "unknown",
                            "confidence": top_probs[i],
                            "category": self._map_category(self.labels[idx] if idx<len(self.labels) else "")}
                           for i,idx in enumerate(top_indices)]
            best = predictions[0]
            return {"scene_type":best["scene"],"confidence":round(best["confidence"]*100,1),
                    "scene_category":best["category"],"top_predictions":predictions[:3]}
        except: return {"scene_type":"unknown","confidence":0,"scene_category":"unknown"}

    def _map_category(self, scene):
        s = scene.lower()
        if any(k in s for k in ['forest','mountain','beach','desert','field','canyon','lake','river','waterfall','glacier']): return "nature"
        if any(k in s for k in ['room','house','office','shop','mall','restaurant','kitchen','bathroom','bedroom','corridor','library','classroom']): return "indoor"
        if any(k in s for k in ['street','road','park','sky','coast']): return "outdoor"
        return "unknown"

# ══════════════════════════════════════════════════════════
# MODULE 4G — YOLO OBJECT DETECTION
# ══════════════════════════════════════════════════════════
class YOLODetector:
    def __init__(self): self.model=None

    def load_model(self):
        if not YOLO_AVAILABLE: return False
        try:
            print("    Loading YOLOv8 object detector...")
            self.model = YOLO('yolov8n.pt'); return True
        except Exception as e:
            print(f"    ⚠️  Failed to load YOLO: {e}"); return False

    def detect_objects(self, image, confidence_threshold=0.25):
        if self.model is None:
            if not self.load_model(): return {"objects":[],"object_count":0,"object_categories":{}}
        try:
            results = self.model(image, verbose=False)[0]
            objects=[]; object_categories={}
            category_map = {
                'person':'people','bicycle':'vehicle','car':'vehicle','motorcycle':'vehicle',
                'airplane':'vehicle','bus':'vehicle','train':'vehicle','truck':'vehicle','boat':'vehicle',
                'traffic light':'street','fire hydrant':'street','stop sign':'street','parking meter':'street','bench':'furniture',
                'bird':'animal','cat':'animal','dog':'animal','horse':'animal','sheep':'animal',
                'cow':'animal','elephant':'animal','bear':'animal','zebra':'animal','giraffe':'animal',
                'backpack':'accessory','umbrella':'accessory','handbag':'accessory','tie':'accessory','suitcase':'accessory',
                'frisbee':'sports','skis':'sports','snowboard':'sports','sports ball':'sports','kite':'sports',
                'baseball bat':'sports','baseball glove':'sports','skateboard':'sports','surfboard':'sports','tennis racket':'sports',
                'bottle':'food_drink','wine glass':'food_drink','cup':'food_drink','fork':'food_drink',
                'knife':'food_drink','spoon':'food_drink','bowl':'food_drink','banana':'food_drink',
                'apple':'food_drink','sandwich':'food_drink','orange':'food_drink','broccoli':'food_drink',
                'carrot':'food_drink','hot dog':'food_drink','pizza':'food_drink','donut':'food_drink','cake':'food_drink',
                'chair':'furniture','couch':'furniture','potted plant':'plant','bed':'furniture',
                'dining table':'furniture','toilet':'furniture','tv':'electronics','laptop':'electronics',
                'mouse':'electronics','remote':'electronics','keyboard':'electronics','cell phone':'electronics',
                'microwave':'appliance','oven':'appliance','toaster':'appliance','sink':'appliance','refrigerator':'appliance',
                'book':'stationery','clock':'electronics','vase':'decor','scissors':'tool',
                'teddy bear':'toy','hair drier':'personal_care','toothbrush':'personal_care'
            }
            if results.boxes is not None:
                for box in results.boxes:
                    conf = float(box.conf[0])
                    if conf < confidence_threshold: continue
                    cls_id = int(box.cls[0]); cls_name = results.names[cls_id]
                    x1,y1,x2,y2 = box.xyxy[0].tolist()
                    hh,ww = image.shape[:2]
                    cx=((x1+x2)/2)/ww; cy=((y1+y2)/2)/hh
                    category = category_map.get(cls_name,'other')
                    objects.append({"class":cls_name,"confidence":round(conf*100,1),
                                    "category":category,"position":{"cx":round(cx,3),"cy":round(cy,3)},
                                    "bbox":[int(x1),int(y1),int(x2),int(y2)],
                                    "size_ratio":((x2-x1)*(y2-y1))/(ww*hh)})
                    object_categories[category]=object_categories.get(category,0)+1
            objects.sort(key=lambda x:x['confidence'],reverse=True)
            return {"objects":objects,"object_count":len(objects),"object_categories":object_categories,
                    "has_people":any(o['class']=='person' for o in objects),
                    "has_pets":any(o['category']=='animal' for o in objects),
                    "has_technology":any(o['category']=='electronics' for o in objects)}
        except Exception as e:
            print(f"    ⚠️  Object detection failed: {e}")
            return {"objects":[],"object_count":0,"object_categories":{}}

# ══════════════════════════════════════════════════════════
# MODULE 4H — ACTIVITY RECOGNITION
# ══════════════════════════════════════════════════════════
def recognize_activity(objects, face_data, scene_data):
    activities=[]; confidence=0
    cls = [o['class'] for o in objects]
    if any(i in cls for i in ['bottle','cup','fork','knife','bowl','sandwich','pizza']): activities.append("eating/drinking"); confidence+=20
    if any(i in cls for i in ['laptop','book','keyboard','mouse','tv']): activities.append("working/studying"); confidence+=25
    if any(i in cls for i in ['sports ball','skis','snowboard','frisbee']): activities.append("exercising/sports"); confidence+=30
    if face_data.get('face_count',0)>0 and scene_data.get('upper_body_visible',False): activities.append("talking/presenting"); confidence+=35
    if any(i in cls for i in ['microwave','oven','refrigerator','sink','bowl']): activities.append("cooking"); confidence+=25
    if any(i in cls for i in ['car','truck','bus','motorcycle','bicycle']): activities.append("driving/traveling"); confidence+=30
    if any(i in cls for i in ['handbag','suitcase','backpack']): activities.append("shopping"); confidence+=15
    if len([o for o in objects if o['class']=='person'])>2: activities.append("concert/event"); confidence+=40
    if not activities:
        activities.append("talking head" if face_data.get('face_count',0)>0 else "scenic/b-roll")
    return {"primary_activity":activities[0],"all_activities":activities,"confidence":min(confidence,95)}

# ══════════════════════════════════════════════════════════
# MODULE 4I — AESTHETIC SCORING
# ══════════════════════════════════════════════════════════
def score_aesthetic_quality(image, objects, face_data, frame_data):
    h,w = image.shape[:2]; score=50; factors=[]
    if objects:
        thirds_x=[w/3,2*w/3]; thirds_y=[h/3,2*h/3]
        mo=objects[0]; cx=mo['bbox'][0]+(mo['bbox'][2]-mo['bbox'][0])/2; cy=mo['bbox'][1]+(mo['bbox'][3]-mo['bbox'][1])/2
        dx=min(abs(cx-tx) for tx in thirds_x)/w; dy=min(abs(cy-ty) for ty in thirds_y)/h
        if dx<0.1 and dy<0.1: score+=10; factors.append("good_composition")
        elif dx<0.15 or dy<0.15: score+=5; factors.append("decent_composition")
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    brightness=np.mean(gray); contrast=np.std(gray)
    if 100<=brightness<=200: score+=5; factors.append("good_brightness")
    if contrast>50: score+=5; factors.append("good_contrast")
    elif contrast<20: score-=5; factors.append("low_contrast")
    if face_data.get('face_count',0)>0:
        for face in face_data.get('faces',[]):
            if 0.2<=face['cx']<=0.8 and 0.2<=face['cy']<=0.8:
                score+=5; factors.append("face_well_framed"); break
        if face_data.get('eye_contact',False): score+=5; factors.append("eye_contact")
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    hue_hist=cv2.calcHist([hsv],[0],None,[180],[0,180])
    non_zero_hues=np.count_nonzero(hue_hist>max(hue_hist)*0.1)
    if non_zero_hues<30: score+=5; factors.append("color_harmony")
    elif non_zero_hues>80: score-=5; factors.append("too_colorful")
    if frame_data.get('is_flat_bg',False): score+=5; factors.append("clean_background")
    elif frame_data.get('is_busy_bg',False): score-=5; factors.append("busy_background")
    if frame_data.get('sharpness',0)>100: score+=5; factors.append("sharp_image")
    elif frame_data.get('sharpness',0)<30: score-=5; factors.append("blurry")
    return {"aesthetic_score":max(0,min(100,score)),"factors":factors,
            "brightness":brightness,"contrast":contrast,"color_count":non_zero_hues}

# ══════════════════════════════════════════════════════════
# MODULE 4J — MASTER IMAGE CLASSIFIER
# ══════════════════════════════════════════════════════════
class ImageClassifier:
    def __init__(self):
        self.places365=Places365Classifier(); self.yolo=YOLODetector()
        self.scene_classifier_loaded=False; self.object_detector_loaded=False

    def load_models(self):
        self.scene_classifier_loaded = self.places365.load_model()
        self.object_detector_loaded  = self.yolo.load_model()

    def analyze_image(self, image, scene_data, face_data):
        scene_result    = self.places365.classify_scene(image) if self.scene_classifier_loaded else {}
        object_result   = self.yolo.detect_objects(image, Config.YOLO_CONFIDENCE) if self.object_detector_loaded else {}
        activity_result = recognize_activity(object_result.get('objects',[]), face_data, scene_data)
        aesthetic_result= score_aesthetic_quality(image, object_result.get('objects',[]), face_data, scene_data)
        return {"scene_classification":scene_result,"object_detection":object_result,
                "activity_recognition":activity_result,"aesthetic_quality":aesthetic_result}

# ══════════════════════════════════════════════════════════
# MODULE 4K — PERSON TRACKING ACROSS FRAMES
# ══════════════════════════════════════════════════════════
def track_people_across_frames(frame_data):
    track_history=[]; next_id=0
    for i,fd in enumerate(frame_data):
        people=[o for o in fd.get('objects_detected',[]) if o['class']=='person']
        frame_people=[]
        for p in people:
            matched=False
            if i>0 and track_history:
                for prev in track_history[-1]:
                    b1=p['bbox']; b2=prev['bbox']
                    x1=max(b1[0],b2[0]); y1=max(b1[1],b2[1]); x2=min(b1[2],b2[2]); y2=min(b1[3],b2[3])
                    if x2>x1 and y2>y1:
                        inter=(x2-x1)*(y2-y1)
                        area1=(b1[2]-b1[0])*(b1[3]-b1[1]); area2=(b2[2]-b2[0])*(b2[3]-b2[1])
                        iou=inter/(area1+area2-inter)
                        if iou>0.3: p['person_id']=prev['person_id']; matched=True; break
            if not matched: p['person_id']=next_id; next_id+=1
            frame_people.append(p)
        track_history.append(frame_people)
    distinct_ids=set()
    for fp in track_history:
        for p in fp: distinct_ids.add(p['person_id'])
    return len(distinct_ids), track_history

# ══════════════════════════════════════════════════════════
# MODULE 5 — PIL COLOR PALETTE
# ══════════════════════════════════════════════════════════
def analyze_color_palette(path: str) -> list:
    cached = load_cache(path, "pil")
    if cached: return cached
    cap=cv2.VideoCapture(path); fps=cap.get(cv2.CAP_PROP_FPS)
    interval=max(1,int(fps*Config.PIL_SAMPLE_SEC))
    frame_idx=0; pil_data=[]
    while cap.isOpened():
        ret,frame=cap.read()
        if not ret: break
        if frame_idx%interval==0:
            img=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
            stat=ImageStat.Stat(img)
            small=img.resize((50,50)); q=small.quantize(colors=8).convert("RGB")
            pal=sorted(q.getcolors(maxcolors=2500) or [],key=lambda x:-x[0])
            r,g,b=stat.mean[:3]
            if r>150 and g<100 and b<100:   mood="energetic_red"
            elif b>150 and r<120:            mood="calm_cool"
            elif g>140 and r<130:            mood="natural_green"
            elif r>160 and g>140 and b<100:  mood="warm_golden"
            elif stat.mean[0]<60:            mood="dark_moody"
            elif stat.mean[0]>200:           mood="bright_clean"
            else:                            mood="neutral"
            pil_data.append({"sec":round(frame_idx/fps,2),"mean_rgb":[round(x,1) for x in [r,g,b]],
                             "stddev_rgb":[round(x,1) for x in stat.stddev[:3]],
                             "dominant_colors":[list(p[1]) for p in pal[:6]],
                             "perceived_brightness":round(0.299*r+0.587*g+0.114*b,1),
                             "color_mood":mood,"contrast_score":round(float(np.mean(stat.stddev[:3])),1),
                             "color_variance":round(float(np.std([r,g,b])),1)})
        frame_idx+=1
    cap.release()
    save_cache(path,"pil",pil_data)
    return pil_data

# ========== SCENE DETECTION USING SCENEDETECT ==========
def detect_scenes(video_path: str) -> list:
    """Return list of scene cut times (seconds)."""
    if not SCENEDETECT_AVAILABLE or not Config.USE_SCENEDETECT:
        # Fallback to motion-based detection? We'll just return empty.
        return []
    try:
        video_manager = VideoManager([video_path])
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=30.0))
        video_manager.set_duration()  # detect whole video
        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)
        scenes = scene_manager.get_scene_list()
        # Convert scene list to list of start times (end times maybe later)
        cut_times = [scene[0].get_seconds() for scene in scenes] + [scene[-1].get_seconds() for scene in scenes[-1:]]
        return cut_times
    except Exception as e:
        print(f"    ⚠️  SceneDetect failed: {e}")
        return []

def get_scene_cuts(video_path: str) -> list:
    """Return list of frame indices where scenes change."""
    cuts = detect_scenes(video_path)
    if cuts:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cut_frames = [int(t * fps) for t in cuts]
        cap.release()
        return cut_frames
    return []

# ========== SEGMENT ANYTHING (SAM) ==========
class SAMPredictor:
    def __init__(self):
        self.model = None
        self.predictor = None
        self.available = False

    def load_model(self):
        if not SAM_AVAILABLE or not Config.USE_SAM:
            return False
        try:
            # Download model if not present (you can specify path)
            checkpoint = os.path.join(Config.CACHE_DIR, "sam_vit_h_4b8939.pth")
            if not os.path.exists(checkpoint):
                print("    Downloading SAM model (approx 2.4GB)...")
                urllib.request.urlretrieve(
                    "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
                    checkpoint
                )
            model_type = "vit_h"
            sam_model = sam_model_registry[model_type](checkpoint=checkpoint)
            self.predictor = SamPredictor(sam_model)
            self.available = True
            return True
        except Exception as e:
            print(f"    ⚠️  SAM init failed: {e}")
            return False

    def segment_objects(self, image, boxes=None):
        """Run SAM segmentation. If boxes provided, segment within those boxes."""
        if not self.available:
            return None
        try:
            self.predictor.set_image(image)
            if boxes is not None:
                # boxes are in xyxy format
                masks, scores, _ = self.predictor.predict(box=boxes, multimask_output=False)
                return masks, scores
            else:
                # For full image segmentation without prompts, we'd need to generate points, but that's complex.
                # Return None for now.
                return None
        except Exception as e:
            print(f"    ⚠️  SAM segmentation failed: {e}")
            return None

# ══════════════════════════════════════════════════════════
# MODULE 6 — VISUAL AGGREGATION (updated with new fields)
# ══════════════════════════════════════════════════════════
def aggregate_visual(frame_data: list, pil_data: list, moviepy_data: dict) -> dict:
    if not frame_data: return {}
    def avg(key):
        vals=[f[key] for f in frame_data if isinstance(f.get(key),(int,float))]
        return round(float(np.mean(vals)),3) if vals else 0.0
    def mode_key(key):
        vals=[f.get(key) for f in frame_data if f.get(key)]
        return Counter(vals).most_common(1)[0][0] if vals else "unknown"

    face_ratio=round(sum(1 for f in frame_data if f["face_count"]>0)/len(frame_data),3)
    text_ratio=round(sum(1 for f in frame_data if f["has_text_overlay"])/len(frame_data),3)
    eye_ratio =round(sum(1 for f in frame_data if f.get("eye_contact"))/len(frame_data),3)
    all_objects=[]; object_categories={}
    for f in frame_data:
        for obj in f.get("objects_detected",[]):
            all_objects.append(obj["class"])
            cat=obj["category"]; object_categories[cat]=object_categories.get(cat,0)+1
    object_counts=Counter(all_objects); top_objects=object_counts.most_common(5)
    avg_aesthetic=avg("aesthetic_score")
    scene_categories=[f.get("scene_category") for f in frame_data if f.get("scene_category")]
    indoor_ratio=sum(1 for sc in scene_categories if sc=="indoor")/max(len(scene_categories),1)
    outdoor_ratio=sum(1 for sc in scene_categories if sc=="outdoor")/max(len(scene_categories),1)
    nature_ratio=sum(1 for sc in scene_categories if sc=="nature")/max(len(scene_categories),1)
    distinct_people,_=track_people_across_frames(frame_data)
    moods=[p["color_mood"] for p in pil_data]
    dom_mood=max(set(moods),key=moods.count) if moods else "unknown"
    # Use scene cuts from frame_data if available (we'll set scene_change flag in frame analysis)
    estimated_cuts=sum(1 for f in frame_data if f.get("scene_change"))
    pacing="fast" if estimated_cuts>8 else "medium" if estimated_cuts>3 else "slow"
    return {
        "duration_sec":          moviepy_data["duration_sec"],
        "avg_brightness":        avg("brightness"), "avg_saturation":avg("saturation"),
        "avg_sharpness":         avg("sharpness"),  "avg_motion":avg("motion_score"),
        "avg_edge_density":      avg("edge_density"),
        "face_presence_ratio":   face_ratio, "eye_contact_ratio":eye_ratio,
        "text_overlay_ratio":    text_ratio, "estimated_cuts":estimated_cuts, "pacing":pacing,
        "dominant_camera_move":  mode_key("camera_move"), "dominant_gaze":mode_key("gaze_direction"),
        "dominant_color_mood":   dom_mood,
        "all_ocr_text":          " ".join(dict.fromkeys(f["ocr_text"] for f in frame_data if f["ocr_text"])),
        "all_hindi_ocr_text":    " ".join(dict.fromkeys(f.get("hindi_text","") for f in frame_data if f.get("hindi_text"))),
        "all_english_ocr_text":  " ".join(dict.fromkeys(f.get("english_text","") for f in frame_data if f.get("english_text"))),
        "scripts_detected":      list({s for f in frame_data for s in f.get("scripts_detected",[])}),
        "avg_text_readability":  avg("text_readability"),
        "dominant_setting":      mode_key("setting_type"), "indoor_outdoor":mode_key("indoor_outdoor"),
        "dominant_lighting":     mode_key("lighting"),     "dominant_wall_color":mode_key("wall_color"),
        "avg_people_in_frame":   avg("people_count"),
        "max_people_in_frame":   max((f.get("people_count",0) for f in frame_data),default=0),
        "solo_presenter_frames": sum(1 for f in frame_data if f.get("people_count",0)==1),
        "multi_person_frames":   sum(1 for f in frame_data if f.get("people_count",0)>1),
        "distinct_people_count": distinct_people,
        "dominant_scene_type":   mode_key("scene_type"), "dominant_activity":mode_key("primary_activity"),
        "indoor_percentage":     round(indoor_ratio*100,1), "outdoor_percentage":round(outdoor_ratio*100,1),
        "nature_percentage":     round(nature_ratio*100,1),
        "avg_aesthetic_score":   round(avg_aesthetic,1),  "object_diversity":len(object_categories),
        "top_objects_detected":  [obj for obj,_ in top_objects],
        "object_category_counts":object_categories,
        "has_people_in_video":   any(f.get("has_people") for f in frame_data),
        "has_pets_in_video":     any(f.get("has_pets") for f in frame_data),
        "has_technology_in_video":any(f.get("has_technology") for f in frame_data),
        "style": ("talking_head" if face_ratio>0.6 and mode_key("primary_activity")=="talking/presenting"
                  else "text_educational" if text_ratio>0.5
                  else "vlog_lifestyle" if "outdoor" in mode_key("scene_type").lower() or nature_ratio>0.3
                  else "tutorial" if mode_key("primary_activity") in ["working/studying","cooking"]
                  else "entertainment" if mode_key("primary_activity") in ["eating/drinking","concert/event"]
                  else "broll_cinematic" if avg("motion_score")<2
                  else "action_dynamic" if avg("motion_score")>8 else "mixed"),
    }

# ══════════════════════════════════════════════════════════
# MODULE 7 — CONTENT CATEGORIZER (v3.0: +viral_meme, +audio_events)
# ══════════════════════════════════════════════════════════
def categorize_content(visual_summary: dict, audio_data: dict,
                       transcript_data: dict, ollama_result: dict = None) -> dict:
    face_ratio  = visual_summary["face_presence_ratio"]
    text_ratio  = visual_summary["text_overlay_ratio"]
    eye_ratio   = visual_summary["eye_contact_ratio"]
    avg_motion  = visual_summary["avg_motion"]
    pacing      = visual_summary["pacing"]
    duration    = visual_summary["duration_sec"]
    audio_mood  = audio_data.get("audio_mood","unknown")
    audio_type  = audio_data.get("audio_type","unknown")
    tempo       = audio_data.get("tempo_bpm",0)
    has_voice   = audio_data.get("has_speech",False)
    audio_events= audio_data.get("audio_events",[])
    # New audio fields
    emotion = audio_data.get("emotion", {}).get("emotion", "unknown")
    num_speakers = audio_data.get("num_speakers", 0)

    transcript_quality  = transcript_data.get("quality",{})
    transcript_reliable = transcript_quality.get("reliable",True)
    transcript = transcript_data.get("transcript","") if transcript_reliable else ""

    combined_text = (visual_summary.get("all_ocr_text","") + " " + transcript).lower().strip()
    for ev in audio_events:
        combined_text += " " + ev["class"].lower()

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
        "meme_format":         ["ghop","m kha laadle","drinking sound","gulp","slurp","take this l"],
    }
    detected_hooks = [k for k,phrases in hook_patterns.items()
                      if any(p in combined_text for p in phrases)]

    niche_keywords = {
        "viral_meme":          ["ghop","m kha","laadle","drinking sound","gulp","slurp","take this l"],
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
                                "किस्मत","fate","rashi","astro","planet","nakshatra","साप","सर्प"],
    }
    niche_scores = {}
    for niche, keywords in niche_keywords.items():
        score = sum(1 for kw in keywords if kw in combined_text)
        if score > 0: niche_scores[niche] = score
    detected_niche = max(niche_scores, key=niche_scores.get) if niche_scores else "general"

    if (not transcript_reliable and ollama_result and
            ollama_result.get("available") and not ollama_result.get("error")):
        ollama_niche = ollama_result.get("topic_niche","")
        ollama_conf  = ollama_result.get("confidence",0)
        if ollama_niche and ollama_conf >= 60:
            niche_map = {"astrology":"astrology_horoscope","horoscope":"astrology_horoscope",
                         "spiritual":"mental_health","comedy":"humor_relatable",
                         "finance":"finance","fitness":"fitness","food":"food",
                         "tech":"tech_ai","motivation":"motivation",
                         "beauty":"beauty_fashion","travel":"travel",
                         "education":"education","relationship":"relationship"}
            for k,v in niche_map.items():
                if k in ollama_niche.lower(): detected_niche=v; break
            else: detected_niche=ollama_niche.replace(" ","_").lower()

    if face_ratio>0.6 and has_voice and eye_ratio>0.4:  content_format="direct_to_camera"
    elif face_ratio>0.6 and has_voice:                   content_format="talking_head"
    elif face_ratio>0.6 and not has_voice:               content_format="silent_presenter"
    elif text_ratio>0.7:                                 content_format="text_only"
    elif text_ratio>0.5 and face_ratio>0.3:              content_format="text_plus_face"
    elif avg_motion>8:                                   content_format="action_broll"
    elif avg_motion<2:                                   content_format="static_broll"
    else:                                                content_format="mixed_broll"

    wpm = transcript_data.get("speaking_rate_wpm",0)
    if wpm>160:   speaking_style="fast_energetic"
    elif wpm>120: speaking_style="conversational"
    elif wpm>80:  speaking_style="deliberate"
    elif wpm>0:   speaking_style="slow_dramatic"
    else:         speaking_style="no_speech"

    score=0
    if pacing=="fast":     score+=20
    elif pacing=="medium": score+=10
    if audio_mood in ["hype_energetic","upbeat_positive"]: score+=12
    if detected_hooks:              score+=15
    if text_ratio>0.5:              score+=8
    if 100<=tempo<=140:             score+=8
    if "relatable_situation" in detected_hooks: score+=10
    if "call_to_action" in detected_hooks:      score+=5
    if has_voice and transcript_reliable:       score+=10
    if detected_niche!="general":              score+=8
    if eye_ratio>0.4:               score+=4
    if audio_data.get("silence_ratio",0)>0.2:  score-=5
    if pacing=="slow":              score-=5
    if not detected_hooks:          score-=5
    # New factors
    if emotion in ["happy", "excited", "surprised"]: score+=5
    if num_speakers > 1: score+=5  # multiple speakers adds dynamism
    score=max(0,min(score,100))

    if duration<8:    duration_tag="ultra_short"
    elif duration<15: duration_tag="short"
    elif duration<30: duration_tag="medium"
    elif duration<60: duration_tag="long"
    else:             duration_tag="extended"

    recommendations=[]
    if pacing=="slow":             recommendations.append("⚡ Speed up pacing — aim for a cut every 2-3 seconds")
    if not detected_hooks:         recommendations.append("🪝 Add a stronger hook in the first 1-2 seconds")
    if text_ratio<0.3:             recommendations.append("📝 Add text overlays — 85% of reels watched on mute")
    if tempo<100:                  recommendations.append("🎵 Use faster trending audio (100-130 BPM performs best)")
    if face_ratio==0:              recommendations.append("👤 Add a face — videos with faces get 38% more engagement")
    if eye_ratio<0.3 and face_ratio>0.3: recommendations.append("👁️ Improve eye contact with camera for trust signal")
    if audio_data.get("silence_ratio",0)>0.15: recommendations.append("🔇 Reduce silence gaps — viewers drop off in quiet sections")
    if "call_to_action" not in detected_hooks: recommendations.append("📣 Add a CTA (save, follow, comment) before the end")
    if not transcript_reliable:    recommendations.append("🎙️ Audio quality may be unclear — consider adding subtitle text overlays")
    if not recommendations:        recommendations.append("✅ Strong format — replicate this structure exactly")

    result = {"detected_niche":detected_niche,"niche_scores":niche_scores,
            "content_format":content_format,"hook_formats":detected_hooks,
            "speaking_style":speaking_style,"audio_mood":audio_mood,"audio_type":audio_type,
            "duration_category":duration_tag,"viral_score":score,
            "recommendations":recommendations,"transcript_reliable":transcript_reliable,
            "replication_template":{"format":content_format,
                                    "hook_style":detected_hooks[0] if detected_hooks else "unknown",
                                    "audio_mood":audio_mood,"pacing":pacing,
                                    "use_face":face_ratio>0.3,"use_eye_contact":eye_ratio>0.3,
                                    "use_text":text_ratio>0.3,"target_bpm":round(tempo),
                                    "niche":detected_niche,"speaking_style":speaking_style}}
    
    # Add weak supervision labels if enabled
    if Config.USE_WEAK_SUPERVISION:
        try:
            labeler = WeakSupervisionLabeler()
            # Create a minimal report dict for labeler
            report_dict = {
                'visual_summary': visual_summary,
                'speech_transcript': transcript_data
            }
            labels = labeler.suggest_labels(report_dict, top_k=3)
            result['weak_supervision_labels'] = labels
        except Exception as e:
            print(f"⚠️  Weak supervision labeling error: {e}")
            result['weak_supervision_labels'] = []
    
    return result

# ══════════════════════════════════════════════════════════
# MODULE 4-FRAMES — FRAME ANALYSIS (with new OCR and SAM)
# ══════════════════════════════════════════════════════════
def analyze_frames_with_ml(path: str) -> tuple:
    """
    Returns (frame_data, keyframes, clip_meme_results).

    Enhanced with:
      - Multi-engine OCR (Paddle, EasyOCR, Surya, DocTR)
      - Optional SAM segmentation on keyframes
      - Scene cut detection via scenedetect
    """
    cached = load_cache(path, "frames_v6_ml")
    if cached:
        print("  ♻️  Frames (with ML) — using cache")
        cap=cv2.VideoCapture(path); fps=cap.get(cv2.CAP_PROP_FPS)
        interval=max(1,int(fps*Config.SAMPLE_EVERY_SEC)); keyframes=[]; frame_idx=0
        while cap.isOpened():
            ret,frame=cap.read()
            if not ret: break
            if frame_idx%interval==0: keyframes.append(frame.copy())
            frame_idx+=1
        cap.release()
        return cached, keyframes, []

    classifier=ImageClassifier(); classifier.load_models()
    if Config.USE_CLIP: init_clip()
    if Config.USE_SAM: 
        sam_predictor = SAMPredictor()
        sam_predictor.load_model()
    else:
        sam_predictor = None

    # Precompute scene cuts for more accurate scene_change flags
    cut_frames = get_scene_cuts(path) if Config.USE_SCENEDETECT else []

    cap=cv2.VideoCapture(path); fps=cap.get(cv2.CAP_PROP_FPS)
    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval=max(1,int(fps*Config.SAMPLE_EVERY_SEC))
    frame_data=[]; keyframes=[]; prev_gray=None; frame_idx=0; clip_meme_results=[]

    with tqdm(total=total, desc="  🖼️  Frames (v3.1 ML+OCR+SAM)", unit="fr") as pbar:
        while cap.isOpened():
            ret,frame=cap.read()
            if not ret: break
            if frame_idx%interval==0:
                sec=round(frame_idx/fps,2)
                gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
                lab=cv2.cvtColor(frame,cv2.COLOR_BGR2Lab)
                motion,camera_move=0.0,"static"
                if prev_gray is not None:
                    flow=cv2.calcOpticalFlowFarneback(prev_gray,gray,None,0.5,3,15,3,5,1.2,0)
                    motion=float(np.mean(np.sqrt(flow[...,0]**2+flow[...,1]**2)))
                    mx,my=float(np.mean(flow[...,0])),float(np.mean(flow[...,1]))
                    if abs(mx)>abs(my):
                        camera_move="pan_right" if mx>1 else "pan_left" if mx<-1 else "static"
                    else:
                        camera_move="tilt_down" if my>1 else "tilt_up" if my<-1 else "static"
                sharpness=float(cv2.Laplacian(gray,cv2.CV_64F).var())
                edge_density=float(cv2.Canny(gray,100,200).mean())
                perceptual_lightness=float(lab[:,:,0].mean())
                color_temp=("warm" if np.mean(frame[:,:,2])>np.mean(frame[:,:,0])*1.1
                            else "cool" if np.mean(frame[:,:,0])>np.mean(frame[:,:,2])*1.1 else "neutral")

                face_data  = analyze_face_pose(frame)
                scene_data = detect_scene_background(frame, face_data.get("faces",[]))
                body_data  = count_bodies_hog(frame)

                # ── OCR (multi-engine) ──────────────────
                ocr_text, has_text, used_engine = extract_text_enhanced(frame)

                # If Paddle or Easy succeeded, we have simple text. For storyboard we might want word boxes.
                # We'll still try to get full EasyOCR detail if Paddle failed, but if Paddle succeeded we can skip.
                if used_engine == "paddle" and EASYOCR_AVAILABLE:
                    # Paddle gave us text, but we want word boxes? We could run EasyOCR just for boxes, but that's extra.
                    # For now, we'll just build a minimal dict.
                    easyocr_detail = _empty_easyocr_detail(ocr_text, has_text)
                elif used_engine in ["easyocr", "surya", "doctr"]:
                    # For these, we might have word-level info? Surya and DocTR don't give easy bbox format.
                    # We'll build minimal.
                    easyocr_detail = _empty_easyocr_detail(ocr_text, has_text)
                else:
                    # No OCR, or only paddle but we want boxes? Let's run EasyOCR for boxes if available.
                    if EASYOCR_AVAILABLE and has_text:
                        easyocr_detail = extract_text_easyocr(frame)
                        if not easyocr_detail["text"]:
                            easyocr_detail = _empty_easyocr_detail(ocr_text, has_text)
                    else:
                        easyocr_detail = _empty_easyocr_detail(ocr_text, has_text)

                ml_analysis=classifier.analyze_image(frame,scene_data,face_data)

                clip_meme=None
                clip_embedding=None
                if Config.USE_CLIP and frame_idx%(interval*10)==0:
                    clip_meme=classify_with_clip(frame)
                    if clip_meme: clip_meme_results.append(clip_meme)
                # Extract CLIP embedding for anomaly detection (if enabled)
                # Only extract on every 10th sampled frame to reduce computation
                clip_embedding = None
                if Config.USE_CLIP and Config.USE_ANOMALY_DETECTION and CLIP_MODEL is not None and frame_idx % (interval * 10) == 0:
                    try:
                        image = CLIP_PREPROCESS(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))).unsqueeze(0)
                        if TORCH_AVAILABLE:
                            with torch.no_grad():
                                image_features = CLIP_MODEL.encode_image(image)
                                clip_embedding = image_features.cpu().numpy().flatten()
                    except Exception as e:
                        clip_embedding = None

                # Optional SAM segmentation on key objects (e.g., people)
                sam_masks = None
                if sam_predictor and frame_idx % (interval * 5) == 0:  # every 5th sampled frame
                    # Get person boxes from YOLO
                    person_boxes = [obj["bbox"] for obj in ml_analysis["object_detection"].get("objects", []) if obj["class"] == "person"]
                    if person_boxes:
                        sam_masks, _ = sam_predictor.segment_objects(frame, boxes=person_boxes)

                # Scene change flag: either from scenedetect or motion
                scene_change = frame_idx in cut_frames or motion > 20

                frame_data.append({
                    "frame":frame_idx,"sec":sec,
                    "face_count":face_data["face_count"],"face_positions":face_data["faces"],
                    "gaze_direction":face_data.get("gaze_direction","unknown"),
                    "eye_contact":face_data.get("eye_contact",False),
                    "body_detected":face_data["body_detected"],
                    "hands_raised":face_data.get("hands_raised",False),
                    "people_count":max(face_data["face_count"],body_data["body_count_hog"]),
                    "body_count_hog":body_data["body_count_hog"],
                    "body_positions":body_data["body_positions"],
                    "indoor_outdoor":scene_data["indoor_outdoor"],"setting_type":scene_data["setting_type"],
                    "room_type":scene_data["room_type"],"wall_color":scene_data["wall_color"],
                    "is_flat_bg":scene_data["is_flat_bg"],"bg_texture_std":scene_data["bg_texture_std"],
                    "sky_ratio":scene_data["sky_ratio"],"green_ratio":scene_data["green_ratio"],
                    "lighting":scene_data["lighting"],"bg_mean_rgb":scene_data["bg_mean_rgb"],
                    "motion_score":round(motion,3),"camera_move":camera_move,"scene_change":scene_change,
                    "sharpness":round(sharpness,1),"edge_density":round(edge_density,3),
                    "perceptual_lightness":round(perceptual_lightness,1),"color_temp":color_temp,
                    "hue":round(float(hsv[:,:,0].mean()),1),"saturation":round(float(hsv[:,:,1].mean()),1),
                    "brightness":round(float(hsv[:,:,2].mean()),1),
                    "ocr_text":ocr_text,"ocr_word_count":len(ocr_text.split()) if ocr_text else 0,
                    "has_text_overlay":has_text,
                    "text_readability":easyocr_detail.get("readability_score",0),
                    "hook_text":easyocr_detail.get("hook_text",ocr_text[:150] if ocr_text else ""),
                    "ocr_words":easyocr_detail.get("words",[]),
                    "scripts_detected":easyocr_detail.get("scripts_detected",[]),
                    "hindi_text":easyocr_detail.get("hindi_text",""),
                    "english_text":easyocr_detail.get("english_text",""),
                    "ocr_engine_used":used_engine,
                    # ML fields
                    "scene_type":ml_analysis.get("scene_classification",{}).get("scene_type","unknown"),
                    "scene_confidence":ml_analysis.get("scene_classification",{}).get("confidence",0),
                    "scene_category":ml_analysis.get("scene_classification",{}).get("scene_category","unknown"),
                    "objects_detected":ml_analysis.get("object_detection",{}).get("objects",[]),
                    "object_count":ml_analysis.get("object_detection",{}).get("object_count",0),
                    "object_categories":ml_analysis.get("object_detection",{}).get("object_categories",{}),
                    "has_people":ml_analysis.get("object_detection",{}).get("has_people",False),
                    "has_pets":ml_analysis.get("object_detection",{}).get("has_pets",False),
                    "has_technology":ml_analysis.get("object_detection",{}).get("has_technology",False),
                    "primary_activity":ml_analysis.get("activity_recognition",{}).get("primary_activity","unknown"),
                    "activity_confidence":ml_analysis.get("activity_recognition",{}).get("confidence",0),
                    "aesthetic_score":ml_analysis.get("aesthetic_quality",{}).get("aesthetic_score",50),
                    "aesthetic_factors":ml_analysis.get("aesthetic_quality",{}).get("factors",[]),
                    # SAM masks (store as list of mask arrays? too heavy; just note presence)
                    "has_sam_segmentation": sam_masks is not None,
                    # CLIP embedding for anomaly detection (if available)
                    "clip_embedding": clip_embedding.tolist() if clip_embedding is not None else None,
                })
                keyframes.append(frame.copy())
                prev_gray=gray
            frame_idx+=1; pbar.update(1)
    cap.release()
    
    # Apply temporal analysis if enabled
    if Config.USE_TEMPORAL_ANALYSIS and frame_data:
        temporal = TemporalAnalyzer(window=5)
        activities = [f.get('primary_activity', 'unknown') for f in frame_data]
        smoothed_activities = temporal.smooth_predictions(activities)
        activity_changes = temporal.detect_activity_changes(smoothed_activities)
        for i, f in enumerate(frame_data):
            f['primary_activity_smoothed'] = smoothed_activities[i]
        # Store activity change points
        if activity_changes:
            for idx in activity_changes:
                if idx < len(frame_data):
                    frame_data[idx]['is_activity_change'] = True
    
    # Apply anomaly detection if enabled and CLIP embeddings available
    if Config.USE_ANOMALY_DETECTION and frame_data:
        embeddings = [f.get('clip_embedding') for f in frame_data if f.get('clip_embedding') is not None]
        if embeddings:
            anomaly_detector = VideoAnomalyDetector(threshold=0.7)
            anomalies = anomaly_detector.detect_style_breaks(embeddings)
            # Map anomaly indices back to frame_data indices
            embedding_idx = 0
            for i, f in enumerate(frame_data):
                if f.get('clip_embedding') is not None:
                    if embedding_idx in anomalies:
                        f['is_anomaly'] = True
                    embedding_idx += 1
    
    save_cache(path,"frames_v6_ml",frame_data)
    return frame_data, keyframes, clip_meme_results

# ══════════════════════════════════════════════════════════
# MODULE 8 — AI BRIEF (CLAUDE) — unchanged
# ══════════════════════════════════════════════════════════
def generate_ai_brief(full_report: dict) -> dict:
    if not ANTHROPIC_AVAILABLE or not Config.ANTHROPIC_API_KEY or not Config.ENABLE_AI_BRIEF:
        return {"replication_script":"(disabled)","hook_variations":[],"caption_pack":[],"posting_strategy":""}
    client=anthropic.Anthropic(api_key=Config.ANTHROPIC_API_KEY)
    cat=full_report["content_category"]; audio=full_report["audio_analysis"]
    visual=full_report["visual_summary"]; speech=full_report.get("speech_transcript",{})
    ollama=full_report.get("ollama_analysis",{})
    transcript = speech.get("transcript","") if speech.get("quality",{}).get("reliable") else ""
    ocr_text   = visual.get("all_ocr_text","")
    ollama_brief = ollama.get("content_summary","") if ollama.get("available") else ""
    compressed={
        "duration_sec":full_report["container"]["duration_sec"],"niche":cat["detected_niche"],
        "content_format":cat["content_format"],"hook_formats":cat["hook_formats"],
        "viral_score":cat["viral_score"],"pacing":visual["pacing"],
        "face_ratio":visual["face_presence_ratio"],"text_overlay_ratio":visual["text_overlay_ratio"],
        "audio_mood":audio.get("audio_mood","unknown"),"tempo_bpm":audio.get("tempo_bpm",0),
        "transcript":transcript[:400] if transcript else "(unreliable/none)",
        "ocr_text":ocr_text[:300] if ocr_text else "(none)",
        "ollama_content_brief":ollama_brief[:300] if ollama_brief else "(not available)",
        "indoor_outdoor":visual.get("indoor_outdoor","?"),"setting":visual.get("dominant_setting","?"),
        "avg_people":visual.get("avg_people_in_frame",1),"recommendations":cat["recommendations"],
        "audio_events":[ev["class"] for ev in audio.get("audio_events",[])],
        "clip_meme":full_report.get("clip_meme",""),
    }
    prompt=f"""You are a viral short-form video strategist for Instagram Reels/TikTok.
Analysis of a reel (viral score {compressed['viral_score']}/100):
{json.dumps(compressed, indent=2)}

Return ONLY valid JSON (no markdown) with:
1. "replication_script": 6-10 shots. Each: {{"shot":1,"time":"0-2s","visual":"...","audio":"...","text_overlay":"...","note":"..."}}
2. "hook_variations": 5 hooks. Each: {{"hook":"...","type":"..."}}
3. "caption_pack": 3 captions. Each: {{"caption":"...","hashtags":[...],"cta":"..."}}
4. "posting_strategy": one paragraph
5. "content_improvement": 3 improvements"""
    try:
        response=client.messages.create(model=Config.ANTHROPIC_MODEL,max_tokens=2000,
                                         messages=[{"role":"user","content":prompt}])
        raw=response.content[0].text.strip()
        if raw.startswith("```"): raw="\n".join(raw.split("\n")[1:-1])
        return json.loads(raw)
    except Exception as e:
        return {"error":str(e),"replication_script":[],"hook_variations":[],"caption_pack":[],"posting_strategy":"API call failed"}

# ══════════════════════════════════════════════════════════
# MODULE 9 — HTML REPORT (updated with new sections)
# ══════════════════════════════════════════════════════════
def generate_html_report(full_report: dict, ai_brief: dict, output_path: str):
    cat=full_report["content_category"]; audio=full_report["audio_analysis"]
    visual=full_report["visual_summary"]; container=full_report["container"]
    speech=full_report.get("speech_transcript",{}); ollama=full_report.get("ollama_analysis",{})
    score=cat["viral_score"]
    score_color="#22c55e" if score>=70 else "#f59e0b" if score>=40 else "#ef4444"
    quality=speech.get("quality",{}); q_score=quality.get("score",0)
    q_reliable=quality.get("reliable",False); q_issues=quality.get("issues",[])
    q_color="#22c55e" if q_score>=70 else "#f59e0b" if q_score>=40 else "#ef4444"
    q_script=quality.get("detected_script","unknown")
    transcript_text=speech.get("transcript","") or "(no spoken audio detected)"
    if not q_reliable:
        transcript_display=(f'<div style="background:#2d1a0e;border-left:4px solid #f59e0b;'
                            f'padding:10px;border-radius:6px;margin-bottom:10px;color:#fbbf24;">'
                            f'⚠️ <b>Low confidence transcript</b> — Quality: {q_score}/100 · '
                            f'Script: {q_script} · Issues: {", ".join(q_issues)}<br>'
                            f'<small>Treat this text with caution.</small></div>'
                            f'<div class="transcript-box">{transcript_text}</div>')
    else:
        transcript_display=(f'<div style="background:#0e2d1a;border-left:4px solid #22c55e;'
                            f'padding:10px;border-radius:6px;margin-bottom:10px;color:#86efac;">'
                            f'✅ Transcript quality: {q_score}/100 · Script: {q_script}'
                            f'</div><div class="transcript-box">{transcript_text}</div>')

    audio_events = audio.get("audio_events", [])
    audio_events_html = ""
    if audio_events:
        events_str = ", ".join(f"{ev['class']} ({int(ev['confidence']*100)}%)" for ev in audio_events)
        audio_events_html = f'<div class="stat"><span class="stat-label">🎧 Sound Events</span><span class="stat-value" style="font-size:.8em;">{events_str}</span></div>'

    clip_meme = full_report.get("clip_meme","")
    clip_meme_html = f'<div class="stat"><span class="stat-label">🎭 CLIP Meme Match</span><span class="stat-value">{clip_meme}</span></div>' if clip_meme else ""

    # New audio features
    emotion = audio.get("emotion", {}).get("emotion", "unknown")
    emotion_conf = audio.get("emotion", {}).get("emotion_confidence", 0)
    emotion_html = f'<div class="stat"><span class="stat-label">😊 Emotion</span><span class="stat-value">{emotion} ({int(emotion_conf*100)}%)</span></div>' if emotion != "unknown" else ""

    num_speakers = audio.get("num_speakers", 0)
    speakers_html = f'<div class="stat"><span class="stat-label">🗣️ Speakers</span><span class="stat-value">{num_speakers}</span></div>' if num_speakers > 0 else ""

    if ollama.get("available") and not ollama.get("error"):
        key_phrases_html="".join(f'<span class="tag">{p}</span>' for p in ollama.get("key_phrases",[]))
        layers_html=f"<br>Audio layers: {', '.join(audio.get('layers_detected',[]))}" if audio.get('layers_detected') else ""
        ollama_html=f"""<div class="card" style="border-left:4px solid #06b6d4;">
          <h2>🤖 Ollama Content Brief ({ollama.get("model_used","llama3")})</h2>
          <div class="stat"><span class="stat-label">Summary</span><span class="stat-value" style="max-width:70%;text-align:right;font-size:.9em;font-weight:400;">{ollama.get("content_summary","—")}</span></div>
          <div class="stat"><span class="stat-label">Topic / Niche</span><span class="stat-value">{ollama.get("topic_niche","?")} → {ollama.get("sub_topic","?")}</span></div>
          <div class="stat"><span class="stat-label">Language</span><span class="stat-value">{ollama.get("content_language","?")} {ollama.get("is_multilingual",False) and '🌐' or ''}</span></div>
          <div class="stat"><span class="stat-label">Emotional Tone</span><span class="stat-value">{ollama.get("emotional_tone","?")}</span></div>
          <div class="stat"><span class="stat-label">Target Audience</span><span class="stat-value">{ollama.get("target_audience","?")}</span></div>
          <div class="stat"><span class="stat-label">Horoscope / Advice</span><span class="stat-value">{'🔮 Yes' if ollama.get("has_horoscope_elements") else '—'}{' · 💡 Advice' if ollama.get("has_advice_elements") else ''}</span></div>
          <div class="stat"><span class="stat-label">Confidence</span><span class="stat-value">{ollama.get("confidence","?")}%</span></div>
          <div style="margin-top:10px;">{key_phrases_html}</div>
          <div style="margin-top:5px;font-size:.8em;color:#94a3b8;">{layers_html}</div>
        </div>"""
    elif not ollama.get("available"):
        ollama_html=(f'<div class="card" style="opacity:.5;"><h2>🤖 Ollama Content Brief</h2>'
                     f'<p style="color:#64748b;">Not available: {ollama.get("reason","")}. '
                     f'Run: <code>ollama pull {Config.OLLAMA_MODEL}</code></p></div>')
    else:
        ollama_html=(f'<div class="card" style="opacity:.5;"><h2>🤖 Ollama Content Brief</h2>'
                     f'<p style="color:#ef4444;">Error: {ollama.get("error","")}</p></div>')

    ml_stats=f"""<div class="card"><h2>🧠 ML Analysis</h2>
      <div class="stat"><span class="stat-label">Distinct People</span><span class="stat-value">{visual.get("distinct_people_count",0)}</span></div>
      <div class="stat"><span class="stat-label">Dominant Scene</span><span class="stat-value">{visual.get("dominant_scene_type","unknown")}</span></div>
      <div class="stat"><span class="stat-label">Primary Activity</span><span class="stat-value">{visual.get("dominant_activity","unknown")}</span></div>
      <div class="stat"><span class="stat-label">Aesthetic Score</span><span class="stat-value">{visual.get("avg_aesthetic_score",0)}/100</span></div>
      <div class="stat"><span class="stat-label">Top Objects</span><span class="stat-value">{", ".join(visual.get("top_objects_detected",[])[:3])}</span></div>
      <div class="stat"><span class="stat-label">Indoor/Outdoor/Nature</span><span class="stat-value">{visual.get("indoor_percentage",0)}% / {visual.get("outdoor_percentage",0)}% / {visual.get("nature_percentage",0)}%</span></div>
      {audio_events_html}{clip_meme_html}{emotion_html}{speakers_html}
    </div>"""

    shots_html="".join(f"""<div class="shot-card"><div class="shot-num">Shot {s.get("shot","?")} <span>{s.get("time","")}</span></div>
      <div class="shot-row"><b>🎬</b> {s.get("visual","")}</div><div class="shot-row"><b>🎵</b> {s.get("audio","")}</div>
      <div class="shot-row"><b>📝</b> {s.get("text_overlay","")}</div><div class="shot-note">{s.get("note","")}</div></div>"""
                       for s in ai_brief.get("replication_script",[]) if isinstance(s,dict))
    hooks_html="".join(f'<div class="hook-pill"><span class="hook-type">{h.get("type","")}</span>{h.get("hook","")}</div>'
                       for h in ai_brief.get("hook_variations",[]) if isinstance(h,dict))
    captions_html="".join(f'<div class="caption-card"><p>{c.get("caption","")}</p>'
                          f'<div class="hashtags">{" ".join("#"+t.lstrip("#") for t in c.get("hashtags",[]))}</div>'
                          f'<div class="cta">📣 {c.get("cta","")}</div></div>'
                          for c in ai_brief.get("caption_pack",[]) if isinstance(c,dict))
    improvements_html="".join(f"<li>{i}</li>" for i in ai_brief.get("content_improvement",[]))
    rec_html="".join(f"<li>{r}</li>" for r in cat.get("recommendations",[]))
    scripts_str=", ".join(visual.get("scripts_detected",[])) or "latin"
    energy_data=json.dumps([e["energy"] for e in audio.get("energy_over_time",[])[:60]])
    energy_times=json.dumps([e["sec"] for e in audio.get("energy_over_time",[])[:60]])
    
    # Weak supervision labels
    weak_labels_html = ""
    if Config.USE_WEAK_SUPERVISION and cat.get("weak_supervision_labels"):
        labels = cat.get("weak_supervision_labels", [])
        weak_labels_html = '<div class="card" style="border-left:4px solid #7c3aed;"><h2>🏷️ Weak Supervision Labels</h2>'
        for label in labels:
            conf_color = "#22c55e" if label.get("confidence", 0) > 0.6 else "#f59e0b" if label.get("confidence", 0) > 0.3 else "#64748b"
            evidence = ", ".join(label.get("evidence", []))[:100]
            weak_labels_html += f'<div class="stat"><span class="stat-label">{label.get("label", "unknown").title()}</span><span class="stat-value" style="color:{conf_color};">{label.get("confidence", 0):.2f}</span></div>'
            if evidence:
                weak_labels_html += f'<div style="font-size:.8em;color:#94a3b8;margin-left:10px;margin-bottom:8px;">Keywords: {evidence}</div>'
        weak_labels_html += '</div>'
    
    # Content repurposing and clip generation
    repurposing_html = ""
    clips_info = []
    video_path = full_report.get("file", "")
    if Config.USE_CONTENT_REPURPOSING or Config.USE_AUTO_CLIPS:
        try:
            frame_data = full_report.get("frame_analysis", [])
            if frame_data and video_path and os.path.exists(video_path):
                # Generate clip packages
                if Config.USE_AUTO_CLIPS and FFMPEG_PYTHON_AVAILABLE:
                    clip_gen = AutoClipGenerator(video_path)
                    clips = clip_gen.identify_clips(frame_data, audio)
                    clips_dir = os.path.join(Config.OUTPUT_DIR, "clips", os.path.basename(video_path).replace(".", "_"))
                    extracted = clip_gen.extract_clips(clips, clips_dir)
                    clips_info = clips
                    if extracted:
                        print(f"  ✅ Extracted {len(extracted)} clips to {clips_dir}")
                
                # Generate repurposing packages
                if Config.USE_CONTENT_REPURPOSING:
                    packages = ContentRepurposingEngine.generate_packages(full_report, clips_info)
                    repurposing_html = '<div class="card" style="border-left:4px solid #06b6d4;"><h2>📱 Multi-Platform Repurposing</h2>'
                    for platform, pkg in packages.items():
                        if platform == "schedule":
                            continue
                        repurposing_html += f'<div style="margin-bottom:15px;padding:10px;background:#1e1e2e;border-radius:8px;">'
                        repurposing_html += f'<h3 style="color:#06b6d4;font-size:1em;margin-bottom:8px;">{platform.replace("_", " ").title()}</h3>'
                        repurposing_html += f'<div class="stat"><span class="stat-label">Duration</span><span class="stat-value">{pkg.get("duration", [0, 0])[0]}-{pkg.get("duration", [0, 0])[1]}s</span></div>'
                        repurposing_html += f'<div class="stat"><span class="stat-label">Aspect</span><span class="stat-value">{pkg.get("aspect", "?")}</span></div>'
                        hashtags_str = " ".join(pkg.get("hashtags", []))
                        if hashtags_str:
                            repurposing_html += f'<div style="margin-top:8px;color:#7c3aed;font-size:.85em;">{hashtags_str}</div>'
                        repurposing_html += '</div>'
                    if "schedule" in packages:
                        sched = packages["schedule"]
                        repurposing_html += f'<div style="margin-top:15px;padding:10px;background:#1e1e2e;border-radius:8px;">'
                        repurposing_html += f'<h3 style="color:#06b6d4;font-size:1em;margin-bottom:8px;">📅 Recommended Schedule</h3>'
                        repurposing_html += f'<div class="stat"><span class="stat-label">Days</span><span class="stat-value">{", ".join(sched.get("days", []))}</span></div>'
                        repurposing_html += f'<div class="stat"><span class="stat-label">Times</span><span class="stat-value">{", ".join(sched.get("times", []))}</span></div>'
                        repurposing_html += '</div>'
                    repurposing_html += '</div>'
        except Exception as e:
            print(f"⚠️  Content repurposing error: {e}")

    html=f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Reel Analysis Report v3.1</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0;}}
body{{font-family:'Segoe UI',system-ui,sans-serif;background:#0f0f17;color:#e2e8f0;}}
.header{{background:linear-gradient(135deg,#1a1a2e,#16213e,#0f3460);padding:40px;text-align:center;}}
.header h1{{font-size:2.2em;font-weight:800;color:#fff;}}
.header p{{color:#94a3b8;margin-top:6px;}}
.score-badge{{display:inline-block;background:{score_color}22;border:3px solid {score_color};color:{score_color};font-size:2.8em;font-weight:900;padding:14px 28px;border-radius:14px;margin:16px 0;}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:18px;padding:28px;max-width:1400px;margin:0 auto;}}
.card{{background:#1e1e2e;border-radius:14px;padding:22px;border:1px solid #2d2d44;}}
.card h2{{font-size:.9em;color:#94a3b8;text-transform:uppercase;letter-spacing:.08em;margin-bottom:14px;}}
.stat{{display:flex;justify-content:space-between;padding:7px 0;border-bottom:1px solid #2d2d44;}}
.stat:last-child{{border-bottom:none;}}
.stat-label{{color:#94a3b8;font-size:.88em;}}
.stat-value{{font-weight:700;color:#e2e8f0;font-size:.92em;}}
.tag{{display:inline-block;background:#3b1f5e;color:#c084fc;padding:3px 10px;border-radius:18px;font-size:.78em;margin:2px;font-weight:600;}}
.section{{max-width:1400px;margin:0 auto;padding:0 28px 28px;}}
.section-title{{font-size:1.3em;font-weight:700;margin:28px 0 14px;color:#fff;border-left:4px solid #7c3aed;padding-left:11px;}}
.shot-card{{background:#1e1e2e;border-radius:10px;padding:18px;border-left:4px solid #7c3aed;margin-bottom:10px;}}
.shot-num{{font-size:1em;font-weight:800;color:#a78bfa;margin-bottom:8px;}}
.shot-num span{{font-size:.78em;color:#64748b;font-weight:400;margin-left:6px;}}
.shot-row{{font-size:.88em;margin:3px 0;color:#cbd5e1;}}
.shot-note{{margin-top:7px;font-size:.83em;color:#7c3aed;font-style:italic;}}
.hook-pill{{background:#1e1e2e;border:1px solid #2d2d44;border-radius:10px;padding:12px 16px;margin-bottom:9px;font-size:.93em;}}
.hook-type{{display:inline-block;background:#7c3aed22;color:#a78bfa;font-size:.74em;padding:2px 7px;border-radius:9px;margin-right:7px;font-weight:700;}}
.caption-card{{background:#1e1e2e;border-radius:10px;padding:18px;border:1px solid #2d2d44;margin-bottom:10px;}}
.caption-card p{{color:#e2e8f0;line-height:1.6;font-size:.93em;}}
.hashtags{{color:#7c3aed;font-size:.83em;margin-top:9px;}}
.cta{{color:#22c55e;font-size:.83em;margin-top:5px;}}
.rec-list,.improve-list{{list-style:none;}}
.rec-list li{{padding:9px 13px;background:#1e1e2e;border-radius:7px;margin-bottom:7px;border-left:3px solid #f59e0b;font-size:.93em;}}
.improve-list li{{padding:9px 13px;background:#1e1e2e;border-radius:7px;margin-bottom:7px;border-left:3px solid #22c55e;font-size:.93em;}}
.transcript-box{{background:#1e1e2e;border-radius:10px;padding:18px;border:1px solid #2d2d44;line-height:1.8;color:#cbd5e1;font-size:.93em;font-family:'Noto Sans Devanagari','Segoe UI',sans-serif;direction:auto;}}
.strategy-box{{background:#1e1e2e;border-radius:10px;padding:18px;border-left:4px solid #22c55e;color:#cbd5e1;line-height:1.7;}}
canvas{{max-height:200px;}} code{{background:#2d2d44;padding:2px 6px;border-radius:4px;font-size:.85em;}}
.footer{{text-align:center;padding:28px;color:#475569;font-size:.83em;}}
</style></head><body>
<div class="header">
  <h1>🎬 Reel Intelligence Report <span style="font-size:.5em;color:#7c3aed;">v3.1 enhanced</span></h1>
  <p>Generated {datetime.now().strftime("%B %d, %Y at %H:%M")} · Whisper ({Config.WHISPER_MODEL}, {Config.WHISPER_LANG or "auto"}) · Multi-OCR · MediaPipe · YOLO · Places365 · YAMNet · Demucs · SpeechBrain · Pyannote · SAM · Ollama ({Config.OLLAMA_MODEL})</p>
  <div class="score-badge">{score}/100 Viral Score</div>
</div>
<div class="grid">
  <div class="card"><h2>📦 Container</h2>
    <div class="stat"><span class="stat-label">Duration</span><span class="stat-value">{container["duration_sec"]}s</span></div>
    <div class="stat"><span class="stat-label">Resolution</span><span class="stat-value">{container["resolution"][0]}×{container["resolution"][1]}</span></div>
    <div class="stat"><span class="stat-label">FPS</span><span class="stat-value">{container["fps"]:.1f}</span></div>
    <div class="stat"><span class="stat-label">Format</span><span class="stat-value">{container["format_hint"]}</span></div>
    <div class="stat"><span class="stat-label">Has Audio</span><span class="stat-value">{'✅' if container["has_audio"] else '❌'}</span></div>
  </div>
  <div class="card"><h2>🎯 Content Identity</h2>
    <div class="stat"><span class="stat-label">Niche</span><span class="stat-value">{cat["detected_niche"].replace("_"," ").title()}</span></div>
    <div class="stat"><span class="stat-label">Format</span><span class="stat-value">{cat["content_format"].replace("_"," ").title()}</span></div>
    <div class="stat"><span class="stat-label">Duration Type</span><span class="stat-value">{cat["duration_category"].replace("_"," ").title()}</span></div>
    <div class="stat"><span class="stat-label">Speaking Style</span><span class="stat-value">{cat["speaking_style"].replace("_"," ").title()}</span></div>
    <div class="stat"><span class="stat-label">Transcript Reliable</span><span class="stat-value">{'✅' if cat.get("transcript_reliable") else '⚠️ No'}</span></div>
    <div style="margin-top:11px;">{''.join(f'<span class="tag">{h.replace("_"," ")}</span>' for h in cat.get("hook_formats",[]))}</div>
  </div>
  <div class="card"><h2>🎵 Audio Profile</h2>
    <div class="stat"><span class="stat-label">Tempo</span><span class="stat-value">{audio.get("tempo_bpm",0)} BPM</span></div>
    <div class="stat"><span class="stat-label">Key</span><span class="stat-value">{audio.get("detected_key","unknown")}</span></div>
    <div class="stat"><span class="stat-label">Mood</span><span class="stat-value">{audio.get("audio_mood","unknown").replace("_"," ").title()}</span></div>
    <div class="stat"><span class="stat-label">Layers</span><span class="stat-value">{", ".join(audio.get("layers_detected",["unknown"]))}</span></div>
    <div class="stat"><span class="stat-label">Voice/Music %</span><span class="stat-value">{audio.get("voice_confidence",0)}% / {audio.get("music_confidence",0)}%</span></div>
    {audio_events_html}
    {emotion_html}
    {speakers_html}
  </div>
  <div class="card"><h2>🖼️ Visual Metrics</h2>
    <div class="stat"><span class="stat-label">Avg Brightness</span><span class="stat-value">{visual["avg_brightness"]:.1f}</span></div>
    <div class="stat"><span class="stat-label">Avg Saturation</span><span class="stat-value">{visual["avg_saturation"]:.1f}</span></div>
    <div class="stat"><span class="stat-label">Face Presence</span><span class="stat-value">{int(visual["face_presence_ratio"]*100)}%</span></div>
    <div class="stat"><span class="stat-label">Eye Contact</span><span class="stat-value">{int(visual["eye_contact_ratio"]*100)}%</span></div>
    <div class="stat"><span class="stat-label">Text Overlay</span><span class="stat-value">{int(visual["text_overlay_ratio"]*100)}%</span></div>
    <div class="stat"><span class="stat-label">Cuts / Pacing</span><span class="stat-value">{visual["estimated_cuts"]} / {visual["pacing"]}</span></div>
    <div class="stat"><span class="stat-label">Scripts Detected</span><span class="stat-value">{scripts_str}</span></div>
  </div>
  {ml_stats}
  <div class="card"><h2>🏠 Scene & Background</h2>
    <div class="stat"><span class="stat-label">Indoor / Outdoor</span><span class="stat-value">{visual.get("indoor_outdoor","?").title()}</span></div>
    <div class="stat"><span class="stat-label">Setting</span><span class="stat-value">{visual.get("dominant_setting","?").replace("_"," ").title()}</span></div>
    <div class="stat"><span class="stat-label">Lighting</span><span class="stat-value">{visual.get("dominant_lighting","?").replace("_"," ").title()}</span></div>
    <div class="stat"><span class="stat-label">BG Color</span><span class="stat-value">{visual.get("dominant_wall_color","?").replace("_"," ").title()}</span></div>
  </div>
  <div class="card"><h2>👥 People in Frame</h2>
    <div class="stat"><span class="stat-label">Avg People / Frame</span><span class="stat-value">{visual.get("avg_people_in_frame",0)}</span></div>
    <div class="stat"><span class="stat-label">Max People</span><span class="stat-value">{visual.get("max_people_in_frame",0)}</span></div>
    <div class="stat"><span class="stat-label">Solo Frames</span><span class="stat-value">{visual.get("solo_presenter_frames",0)}</span></div>
    <div class="stat"><span class="stat-label">Multi-Person</span><span class="stat-value">{visual.get("multi_person_frames",0)}</span></div>
    <div class="stat"><span class="stat-label">Distinct People</span><span class="stat-value">{visual.get("distinct_people_count",0)}</span></div>
  </div>
  <div class="card"><h2>🎙️ Transcript Quality</h2>
    <div class="stat"><span class="stat-label">Quality Score</span><span class="stat-value" style="color:{q_color};">{q_score}/100</span></div>
    <div class="stat"><span class="stat-label">Reliable</span><span class="stat-value">{'✅ Yes' if q_reliable else '⚠️ No'}</span></div>
    <div class="stat"><span class="stat-label">Script</span><span class="stat-value">{q_script}</span></div>
    <div class="stat"><span class="stat-label">Avg Log Prob</span><span class="stat-value">{quality.get("avg_logprob","?")}</span></div>
    <div class="stat"><span class="stat-label">Issues</span><span class="stat-value" style="font-size:.8em;color:#f59e0b;">{", ".join(q_issues) or "none"}</span></div>
  </div>
  {ollama_html}
  {weak_labels_html}
  {repurposing_html}
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
  <div class="strategy-box">{ai_brief.get("posting_strategy","Set ANTHROPIC_API_KEY to generate strategy.")}</div>
  <div class="section-title">⚡ Content Improvements</div>
  <ul class="improve-list">{improvements_html or "<li>Set ANTHROPIC_API_KEY to generate improvements.</li>"}</ul>
</div>
<div class="footer">Reel Analyzer v3.1 Enhanced · Whisper · Multi-OCR · MediaPipe · YOLO · Places365 · YAMNet · Demucs · SpeechBrain · Pyannote · SAM · Ollama · Claude</div>
<script>
const ctx=document.getElementById('energyChart').getContext('2d');
new Chart(ctx,{{type:'line',data:{{labels:{energy_times},datasets:[{{label:'Audio Energy',data:{energy_data},borderColor:'#7c3aed',backgroundColor:'#7c3aed22',fill:true,tension:0.4,pointRadius:0}}]}},
options:{{responsive:true,plugins:{{legend:{{labels:{{color:'#94a3b8'}}}}}},scales:{{x:{{ticks:{{color:'#64748b'}},grid:{{color:'#1e1e2e'}}}},y:{{ticks:{{color:'#64748b'}},grid:{{color:'#2d2d44'}}}}}}}}}}
);
</script></body></html>"""
    with open(output_path,"w",encoding="utf-8") as f: f.write(html)
    print(f"📄 HTML report saved: {output_path}")

# ══════════════════════════════════════════════════════════
# MODULE 10 — STORYBOARD (unchanged)
# ══════════════════════════════════════════════════════════
def generate_storyboard(keyframes, frame_data, audio_data, category, output_path, cols=4):
    if not keyframes: return
    thumb_w,thumb_h=360,640; label_h=150
    rows=math.ceil(len(keyframes)/cols)
    board=Image.new("RGB",(cols*thumb_w,rows*(thumb_h+label_h)+90),(10,10,14))
    draw=ImageDraw.Draw(board)
    try:
        font=ImageFont.truetype("arial.ttf",14); font_small=ImageFont.truetype("arial.ttf",11)
        font_bold=ImageFont.truetype("arialbd.ttf",15); font_title=ImageFont.truetype("arialbd.ttf",18)
    except: font=font_small=font_bold=font_title=ImageFont.load_default()
    draw.rectangle([0,0,cols*thumb_w,85],fill=(20,10,40))
    draw.text((10,8),f"🎯 {category['detected_niche'].upper()}  |  {category['content_format']}  |  VIRAL: {category['viral_score']}/100",font=font_title,fill=(255,220,50))
    draw.text((10,35),f"🎵 {audio_data.get('tempo_bpm',0)} BPM · {audio_data.get('detected_key','?')} · {audio_data.get('audio_mood','?')}",font=font_bold,fill=(160,210,255))
    draw.text((10,58),f"🪝 {', '.join(category['hook_formats']) or 'no hooks detected'}",font=font,fill=(180,255,180))
    if audio_data.get("audio_events"):
        evt_str=", ".join(ev["class"] for ev in audio_data["audio_events"][:4])
        draw.text((10,75),f"🎧 {evt_str}",font=font_small,fill=(255,200,100))
    for i,(frame,fd) in enumerate(zip(keyframes,frame_data)):
        col=i%cols; row=i//cols
        x=col*thumb_w; y=row*(thumb_h+label_h)+90
        img=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)).resize((thumb_w,thumb_h))
        img_draw=ImageDraw.Draw(img)
        for fp in fd.get("face_positions",[]):
            fx=int(fp["cx"]*thumb_w); fy=int(fp["cy"]*thumb_h)
            sz=int(math.sqrt(max(fp.get("size_ratio",0.05),0.01))*thumb_w)
            c=(0,255,80) if fd.get("eye_contact") else (255,160,0)
            img_draw.ellipse([fx-sz//2,fy-sz//2,fx+sz//2,fy+sz//2],outline=c,width=3)
        for bp in fd.get("body_positions",[]):
            bx=int(bp["cx"]*thumb_w); by=int(bp["cy"]*thumb_h)
            img_draw.rectangle([bx-30,by-60,bx+30,by+60],outline=(80,160,255),width=2)
        for wrd in fd.get("ocr_words",[])[:8]:
            if "bbox" in wrd:
                bx1=int(wrd["bbox"][0]*thumb_w/frame.shape[1]); by1=int(wrd["bbox"][1]*thumb_h/frame.shape[0])
                bx2=int(wrd["bbox"][2]*thumb_w/frame.shape[1]); by2=int(wrd["bbox"][3]*thumb_h/frame.shape[0])
                c=(255,100,100) if wrd.get("script")=="devanagari" else (255,220,0)
                img_draw.rectangle([bx1,by1,bx2,by2],outline=c,width=1)
        beat_near=any(abs(bt-fd["sec"])<0.5 for bt in audio_data.get("beat_times",[]))
        if beat_near:
            img_draw.rectangle([0,thumb_h-8,thumb_w,thumb_h],fill=(200,40,40))
            img_draw.text((4,thumb_h-20),"♪ BEAT",font=font_small,fill=(255,255,255))
        if fd.get("scene_change"): img_draw.rectangle([0,0,thumb_w,6],fill=(255,80,0))
        io_col=(80,200,120) if fd.get("indoor_outdoor")=="outdoor" else (100,140,255)
        img_draw.rectangle([thumb_w-45,4,thumb_w-4,22],fill=io_col)
        img_draw.text((thumb_w-42,6),fd.get("indoor_outdoor","?")[:3].upper(),font=font_small,fill=(0,0,0))
        board.paste(img,(x,y))
        ly=y+thumb_h
        draw.rectangle([x,ly,x+thumb_w,ly+label_h],fill=(14,14,22))
        ec="👁️" if fd.get("eye_contact") else ""
        draw.text((x+5,ly+4),f"⏱ {fd['sec']}s  📷 {fd['camera_move']}  {ec}",font=font,fill=(160,210,255))
        draw.text((x+5,ly+22),f"👤 {fd['face_count']} faces  👥 {fd.get('people_count',0)} people  💨 motion:{fd['motion_score']:.1f}",font=font,fill=(200,200,200))
        draw.text((x+5,ly+40),f"🏠 {fd.get('setting_type','?')[:22]}  💡 {fd.get('lighting','?')[:10]}",font=font_small,fill=(255,200,100))
        draw.text((x+5,ly+56),f"🎨 bg:{fd.get('wall_color','?')[:14]}  📝 {'✅' if fd['has_text_overlay'] else '❌'}",font=font_small,fill=(255,220,80))
        hook_text=fd.get("hook_text",fd.get("ocr_text",""))
        draw.text((x+5,ly+72),f"{hook_text[:42]}{'…' if len(hook_text)>42 else ''}",font=font_small,fill=(200,200,200))
        draw.text((x+5,ly+88),f"scripts:{','.join(fd.get('scripts_detected',[]) or ['?'])[:20]}",font=font_small,fill=(140,220,140))
        draw.text((x+5,ly+104),f"gaze:{fd.get('gaze_direction','?')[:16]}  ♪{'BEAT' if beat_near else '—'}",font=font_small,fill=(255,100,100) if beat_near else (80,80,100))
        draw.text((x+5,ly+120),f"readability:{fd.get('text_readability',0):.0f}%  {'✂️CUT' if fd.get('scene_change') else ''}",font=font_small,fill=(100,180,100))
        draw.text((x+5,ly+136),f"{'HINDI:'+fd.get('hindi_text','')[:20] if fd.get('hindi_text') else ''}",font=font_small,fill=(255,160,80))
        draw.rectangle([x,y,x+42,y+22],fill=(120,30,200))
        draw.text((x+5,y+4),f"#{i+1}",font=font_bold,fill="white")
    board.save(output_path,quality=95)
    print(f"🎬 Storyboard saved: {output_path}")

# ══════════════════════════════════════════════════════════
# MODULE 11 — CSV LOGGING (updated with new fields)
# ══════════════════════════════════════════════════════════
def append_to_csv(full_report: dict, video_path: str):
    cat=full_report["content_category"]; audio=full_report["audio_analysis"]
    visual=full_report["visual_summary"]; speech=full_report.get("speech_transcript",{})
    ollama=full_report.get("ollama_analysis",{})
    audio_events_str="|".join(ev["class"] for ev in audio.get("audio_events",[]))
    emotion = audio.get("emotion", {}).get("emotion", "")
    emotion_conf = audio.get("emotion", {}).get("emotion_confidence", 0)
    num_speakers = audio.get("num_speakers", 0)
    row={
        "filename":os.path.basename(video_path),"analyzed_at":datetime.now().isoformat(),
        "duration_sec":full_report["container"]["duration_sec"],"viral_score":cat["viral_score"],
        "niche":cat["detected_niche"],"content_format":cat["content_format"],
        "pacing":visual["pacing"],"cuts":visual["estimated_cuts"],
        "hook_formats":"|".join(cat.get("hook_formats",[])),
        "tempo_bpm":audio.get("tempo_bpm",0),"detected_key":audio.get("detected_key","unknown"),
        "audio_mood":audio.get("audio_mood","unknown"),"audio_type":audio.get("audio_type","unknown"),
        "face_ratio":visual["face_presence_ratio"],"eye_contact_ratio":visual["eye_contact_ratio"],
        "text_ratio":visual["text_overlay_ratio"],"avg_brightness":visual["avg_brightness"],
        "avg_saturation":visual["avg_saturation"],"avg_motion":visual["avg_motion"],
        "harmony_pct":audio.get("harmony_pct",0),"speaking_style":cat.get("speaking_style",""),
        "transcript_words":speech.get("word_count",0),"speaking_rate_wpm":speech.get("speaking_rate_wpm",0),
        "silence_ratio":audio.get("silence_ratio",0),
        "transcript_quality":speech.get("quality",{}).get("score",0),
        "transcript_reliable":speech.get("quality",{}).get("reliable",False),
        "transcript_script":speech.get("detected_script",""),
        "indoor_outdoor":visual.get("indoor_outdoor",""),"setting_type":visual.get("dominant_setting",""),
        "wall_color":visual.get("dominant_wall_color",""),"lighting":visual.get("dominant_lighting",""),
        "avg_people":visual.get("avg_people_in_frame",0),"max_people":visual.get("max_people_in_frame",0),
        "scripts_detected":"|".join(visual.get("scripts_detected",[])),
        "voice_confidence":audio.get("voice_confidence",0),"music_confidence":audio.get("music_confidence",0),
        "audio_layers":"|".join(audio.get("layers_detected",[])),
        "ollama_topic":ollama.get("topic_niche","") if ollama.get("available") else "",
        "ollama_tone":ollama.get("emotional_tone","") if ollama.get("available") else "",
        "ollama_confidence":ollama.get("confidence",0) if ollama.get("available") else 0,
        "ollama_language":ollama.get("content_language","") if ollama.get("available") else "",
        "distinct_people":visual.get("distinct_people_count",0),
        "dominant_scene":visual.get("dominant_scene_type",""),
        "dominant_activity":visual.get("dominant_activity",""),
        "aesthetic_score":visual.get("avg_aesthetic_score",0),
        "top_objects":"|".join(visual.get("top_objects_detected",[])),
        "indoor_pct":visual.get("indoor_percentage",0),"outdoor_pct":visual.get("outdoor_percentage",0),
        "nature_pct":visual.get("nature_percentage",0),
        "audio_events":audio_events_str,
        "clip_meme":full_report.get("clip_meme",""),
        "emotion":emotion,
        "emotion_confidence":emotion_conf,
        "num_speakers":num_speakers,
    }
    file_exists=os.path.exists(Config.CSV_LOG)
    with open(Config.CSV_LOG,"a",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=row.keys())
        if not file_exists: writer.writeheader()
        writer.writerow(row)
    print(f"📊 CSV log updated: {Config.CSV_LOG}")

# ══════════════════════════════════════════════════════════
# MODULE 12 — ORGANIZE BY CATEGORY (unchanged)
# ══════════════════════════════════════════════════════════
def organize_by_category(report_path, storyboard_path, niche, viral_score):
    import shutil
    tier="top_tier" if viral_score>=70 else "mid_tier" if viral_score>=40 else "low_tier"
    folder=os.path.join(Config.OUTPUT_DIR,"by_category",niche,tier)
    os.makedirs(folder,exist_ok=True)
    for p in [report_path,storyboard_path]:
        if p and os.path.exists(p):
            shutil.copy(p,os.path.join(folder,os.path.basename(p)))
    print(f"📁 Filed under: by_category/{niche}/{tier}/")

# ══════════════════════════════════════════════════════════
# MODULE 13 — BATCH ANALYZE (unchanged)
# ══════════════════════════════════════════════════════════
def batch_analyze(folder: str):
    exts={".mp4",".mov",".avi",".mkv",".webm",".m4v"}
    videos=[str(p) for p in Path(folder).rglob("*") if p.suffix.lower() in exts]
    print(f"\n📂 Found {len(videos)} video(s) in {folder}\n")
    for i,vp in enumerate(videos,1):
        print(f"[{i}/{len(videos)}] {os.path.basename(vp)}")
        try: analyze_video(vp)
        except Exception as e: print(f"  ❌ Failed: {e}")

# ══════════════════════════════════════════════════════════
# MAIN PIPELINE (v3.1 with new features)
# ══════════════════════════════════════════════════════════
def analyze_video(video_path: str) -> dict:
    print(f"\n{'═'*62}")
    print(f"  🎬 Analyzing (v3.1 ULTRA ENHANCED): {os.path.basename(video_path)}")
    print(f"{'═'*62}\n")
    if Config.OLLAMA_ENABLED:
        print(f"  💡 Tip: Pre-warm Ollama:  ollama run {Config.OLLAMA_MODEL} \"ready\"\n")
    slug=Path(video_path).stem[:30].replace(" ","_"); ts=datetime.now().strftime("%Y%m%d_%H%M%S")

    print("📦 [1/7] Container metadata...")
    moviepy_data=analyze_container(video_path)
    print(f"    {moviepy_data['duration_sec']}s  {moviepy_data['fps']:.1f}fps  "
          f"{moviepy_data['resolution'][0]}×{moviepy_data['resolution'][1]}  {moviepy_data['format_hint']}")

    print(f"\n🎙️  [2/7] Speech-to-text (Whisper {Config.WHISPER_MODEL}, lang={Config.WHISPER_LANG or 'auto'})...")
    speech_data=transcribe_speech(video_path,moviepy_data["duration_sec"])
    q=speech_data.get("quality",{})
    if speech_data.get("transcript"):
        print(f"    Lang:{speech_data['language']}  Words:{speech_data['word_count']}  "
              f"WPM:{speech_data['speaking_rate_wpm']}  Quality:{q.get('score',0)}/100 ({'✅' if q.get('reliable') else '⚠️'})")

    print("\n🎵 [3/7] Enhanced audio analysis (Librosa + YAMNet + Demucs/SpeechBrain/Pyannote)...")
    audio_data=analyze_audio(video_path)
    if audio_data.get("is_speech_with_music"):   print("    ✅ Speech with background music")
    elif audio_data.get("is_speech_only"):        print("    ✅ Clean speech detected")
    elif audio_data.get("is_music_only"):         print("    🎵 Music only (no speech)")
    print(f"    Tempo:{audio_data['tempo_bpm']} BPM  Key:{audio_data['detected_key']}  "
          f"Mood:{audio_data.get('audio_mood','?')}  Type:{audio_data.get('audio_type','?')}")
    print(f"    Layers: {', '.join(audio_data.get('layers_detected',['unknown']))}")
    if audio_data.get("audio_events"):
        print(f"    🎧 Sound events: {', '.join(ev['class'] for ev in audio_data['audio_events'])}")
    if audio_data.get("emotion"):
        print(f"    😊 Emotion: {audio_data['emotion'].get('emotion','?')} ({audio_data['emotion'].get('emotion_confidence',0)*100:.0f}%)")
    if audio_data.get("num_speakers"):
        print(f"    🗣️ Speakers: {audio_data['num_speakers']}")

    print("\n🖼️  [4/7] Enhanced frame analysis (Multi-OCR + ML + SAM)...")
    frame_data,keyframes,clip_results=analyze_frames_with_ml(video_path)

    print("\n🎨 [5/7] Color palette (PIL)...")
    pil_data=analyze_color_palette(video_path)

    print("\n📊 [6/7] Aggregating with ML insights...")
    visual_summary=aggregate_visual(frame_data,pil_data,moviepy_data)
    print(f"    🏷️  Dominant Scene:   {visual_summary['dominant_scene_type']}")
    print(f"    🎯 Primary Activity: {visual_summary['dominant_activity']}")
    print(f"    🏠 Indoor:{visual_summary['indoor_percentage']}% / Outdoor:{visual_summary['outdoor_percentage']}%")
    print(f"    🎨 Aesthetic Score:  {visual_summary['avg_aesthetic_score']}/100")
    print(f"    📦 Top Objects:      {', '.join(visual_summary['top_objects_detected'][:3])}")
    print(f"    👥 Distinct People:  {visual_summary['distinct_people_count']}")

    print("\n🤖 [6b] Ollama content analysis...")
    ollama_result=ollama_analyze_content(
        speech_data.get("transcript",""), visual_summary.get("all_ocr_text",""),
        speech_data.get("quality",{}), audio_data.get("layers_detected",[]),
        moviepy_data["duration_sec"])
    if ollama_result.get("available") and not ollama_result.get("error"):
        print(f"    ✅ Ollama: {ollama_result.get('topic_niche','?')} · "
              f"{ollama_result.get('emotional_tone','?')} · confidence:{ollama_result.get('confidence','?')}%")

    print("\n🎯 [7/7] Categorizing with v3.1 features...")
    category=categorize_content(visual_summary,audio_data,speech_data,ollama_result)
    print(f"    Niche:{category['detected_niche']}  Format:{category['content_format']}  Viral:{category['viral_score']}/100")

    full_report={
        "file":video_path,"analyzed_at":datetime.now().isoformat(),
        "container":moviepy_data,"speech_transcript":speech_data,
        "audio_analysis":audio_data,"visual_summary":visual_summary,
        "content_category":category,"frame_analysis":frame_data,
        "color_analysis":pil_data,"ollama_analysis":ollama_result,
    }
    if clip_results:
        full_report["clip_meme"]=Counter(clip_results).most_common(1)[0][0]

    print("\n🤖 [7b] Generating AI brief (Claude)...")
    ai_brief={}
    if Config.ENABLE_AI_BRIEF and Config.ANTHROPIC_API_KEY:
        ai_brief=generate_ai_brief(full_report)
        full_report["ai_brief"]=ai_brief
    else:
        print("  ⏭️  Claude AI brief skipped (set ANTHROPIC_API_KEY)")

    print("\n📝 Generating outputs...")
    json_path=os.path.join(Config.REPORTS_DIR,f"{slug}_{ts}_v3.json")
    with open(json_path,"w",encoding="utf-8") as f:
        json.dump(full_report,f,indent=2,ensure_ascii=False)
    print(f"  ✅ JSON: {json_path}")

    sb_path=os.path.join(Config.REPORTS_DIR,f"{slug}_{ts}_storyboard_v3.jpg")
    generate_storyboard(keyframes,frame_data,audio_data,category,sb_path)

    html_path=os.path.join(Config.REPORTS_DIR,f"{slug}_{ts}_report_v3.html")
    generate_html_report(full_report,ai_brief,html_path)

    append_to_csv(full_report,video_path)
    organize_by_category(html_path,sb_path,category["detected_niche"],category["viral_score"])
    log_trends(full_report)

    print(f"\n{'═'*62}  ✅ ANALYSIS COMPLETE")
    print(f"{'─'*62}")
    print(f"  🎯 Niche:          {category['detected_niche'].replace('_',' ').title()}")
    print(f"  🏆 Viral Score:    {category['viral_score']}/100")
    print(f"  🏷️  Top Scene:       {visual_summary['dominant_scene_type']}")
    print(f"  🎯 Activity:        {visual_summary['dominant_activity']}")
    print(f"  🎨 Aesthetic:       {visual_summary['avg_aesthetic_score']}/100")
    print(f"  📦 Objects:         {', '.join(visual_summary['top_objects_detected'][:3])}")
    print(f"  👥 Distinct People: {visual_summary['distinct_people_count']}")
    if audio_data.get("audio_events"):
        print(f"  🎧 Audio Events:    {', '.join(ev['class'] for ev in audio_data['audio_events'])}")
    if full_report.get("clip_meme"):
        print(f"  🎭 CLIP Meme:       {full_report['clip_meme']}")
    if audio_data.get("emotion"):
        print(f"  😊 Emotion:         {audio_data['emotion'].get('emotion','?')}")
    if audio_data.get("num_speakers"):
        print(f"  🗣️ Speakers:         {audio_data['num_speakers']}")
    print(f"\n  📄 HTML  → {html_path}")
    print(f"  🎬 Board → {sb_path}")
    print(f"  📈 Trends → {Config.TRENDS_LOG}")
    print(f"{'═'*62}\n")
    return full_report

# ══════════════════════════════════════════════════════════
# CLI ENTRY POINT (updated with new arguments)
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser=argparse.ArgumentParser(description="Reel Analyzer v3.1 ULTRA ENHANCED")
    parser.add_argument("path",nargs="?",help="Video file or folder")
    parser.add_argument("--video",type=str,help="Single video file")
    parser.add_argument("--folder",type=str,help="Folder of videos")
    parser.add_argument("--analyze-trends",action="store_true",help="Show trend analysis from existing logs")
    parser.add_argument("--no-ai",action="store_true",help="Skip Claude AI brief")
    parser.add_argument("--no-ollama",action="store_true",help="Skip Ollama analysis")
    parser.add_argument("--whisper-model",default="base",choices=["tiny","base","small","medium","large"])
    parser.add_argument("--lang",default="hi",help="Whisper language (default: hi). Use 'auto' for auto-detect.")
    parser.add_argument("--ollama-model",default=Config.OLLAMA_MODEL)
    parser.add_argument("--ollama-timeout",type=int,default=Config.OLLAMA_TIMEOUT)
    parser.add_argument("--no-ml",action="store_true",help="Disable ML image classification")
    parser.add_argument("--use-clip",action="store_true",help="Enable CLIP meme detection (heavy)")
    parser.add_argument("--no-paddle",action="store_true",help="Disable PaddleOCR, use EasyOCR only")
    parser.add_argument("--no-yamnet",action="store_true",help="Disable YAMNet audio event detection")
    # New v3.1 options
    parser.add_argument("--use-surya",action="store_true",help="Enable Surya OCR (heavy)")
    parser.add_argument("--use-doctr",action="store_true",help="Enable DocTR OCR")
    parser.add_argument("--use-speechbrain",action="store_true",help="Enable SpeechBrain emotion")
    parser.add_argument("--use-demucs",action="store_true",help="Enable Demucs source separation (heavy)")
    parser.add_argument("--use-pyannote",action="store_true",help="Enable Pyannote speaker diarization (requires HF token)")
    parser.add_argument("--no-scenedetect",action="store_true",help="Disable SceneDetect (use motion-based cuts)")
    parser.add_argument("--use-sam",action="store_true",help="Enable Segment Anything (SAM) segmentation (very heavy)")

    args=parser.parse_args()

    if args.analyze_trends: analyze_trends(); sys.exit(0)
    if args.use_clip:    Config.USE_CLIP=True
    if args.no_paddle:   Config.USE_PADDLE=False
    if args.no_yamnet:   Config.USE_YAMNET=False
    if args.no_ai:       Config.ENABLE_AI_BRIEF=False
    if args.no_ollama:   Config.OLLAMA_ENABLED=False
    Config.WHISPER_MODEL  = args.whisper_model
    Config.WHISPER_LANG   = None if args.lang=="auto" else args.lang
    Config.OLLAMA_MODEL   = args.ollama_model
    Config.OLLAMA_TIMEOUT = args.ollama_timeout
    # New flags
    if args.use_surya:     Config.USE_SURYA = True
    if args.use_doctr:     Config.USE_DOCTR = True
    if args.use_speechbrain: Config.USE_SPEECHBRAIN = True
    if args.use_demucs:    Config.USE_DEMUCS = True
    if args.use_pyannote:  Config.USE_PYANNOTE = True
    if args.no_scenedetect: Config.USE_SCENEDETECT = False
    if args.use_sam:       Config.USE_SAM = True

    target=args.video or args.path
    if args.folder:    batch_analyze(args.folder)
    elif target:
        if os.path.isdir(target): batch_analyze(target)
        else:                     analyze_video(target)
    else:
        print("Please specify a video file or folder.")
        parser.print_help() 