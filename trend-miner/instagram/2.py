import cv2
import numpy as np
from PIL import Image, ImageStat, ImageDraw, ImageFont
from moviepy import VideoFileClip
import pytesseract
import json
import os
import math

# ─────────────────────────────────────────
# INIT
# ─────────────────────────────────────────
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

VIDEO_PATH = r"C:\Users\mailt\Downloads\reel_33000likes_AQP1mge7e9VA5c-Ed_1b.mp4"
OUTPUT_DIR = "analysis_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Quick tesseract check
try:
    ver = pytesseract.get_tesseract_version()
    print(f"✅ Tesseract {ver} ready")
except Exception as e:
    print(f"❌ Tesseract not found: {e}")
    print("   Install from: https://github.com/UB-Mannheim/tesseract/wiki")
    exit(1)

# ─────────────────────────────────────────
# 1. MOVIEPY — container + audio
# ─────────────────────────────────────────
def analyze_with_moviepy(path):
    clip  = VideoFileClip(path)
    audio = clip.audio
    data  = {
        "duration_sec":  round(clip.duration, 2),
        "fps":           clip.fps,
        "resolution":    clip.size,
        "aspect_ratio":  round(clip.size[0] / clip.size[1], 3),
        "total_frames":  int(clip.duration * clip.fps),
        "has_audio":     audio is not None,
    }
    if audio:
        samples = audio.to_soundarray(fps=22050)
        mono    = samples.mean(axis=1) if samples.ndim > 1 else samples
        data["audio_channels"]      = samples.shape[1] if samples.ndim > 1 else 1
        data["audio_rms"]           = round(float(np.sqrt(np.mean(mono**2))), 4)
        data["audio_peak"]          = round(float(np.max(np.abs(mono))), 4)
        data["audio_energy_per_sec"]= round(float(np.mean(np.abs(mono))), 4)

        # Beat detection — energy spikes above 1.5x mean
        chunk    = 2205
        energies = [np.sqrt(np.mean(mono[i:i+chunk]**2)) for i in range(0, len(mono)-chunk, chunk)]
        mean_e   = np.mean(energies)
        beats    = [round(i * 0.1, 2) for i, e in enumerate(energies) if e > mean_e * 1.5]
        data["audio_beat_timestamps"] = beats[:20]
        data["estimated_bpm"]         = round(len(beats) / clip.duration * 60, 1) if clip.duration > 0 else 0

    clip.close()
    return data

# ─────────────────────────────────────────
# 2. TESSERACT — extract on-screen text
# ─────────────────────────────────────────
def extract_text_from_frame(frame):
    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_rgb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Three passes: raw, high-contrast, inverted (white text on dark bg)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    inverted  = cv2.bitwise_not(thresh)

    try:
        t1 = pytesseract.image_to_string(img_rgb,                   config="--psm 11").strip()
        t2 = pytesseract.image_to_string(Image.fromarray(thresh),   config="--psm 11").strip()
        t3 = pytesseract.image_to_string(Image.fromarray(inverted), config="--psm 11").strip()
    except Exception as e:
        print(f"  ⚠️ OCR error: {e}")
        return {"text": "", "word_count": 0, "words_with_position": [], "has_caption": False, "hook_text": ""}

    # Word-level bounding boxes with confidence
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
                    "position_v": "top"    if cy < 0.33 else "bottom" if cy > 0.66 else "middle",
                    "position_h": "left"   if cx < 0.33 else "right"  if cx > 0.66 else "center",
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

# ─────────────────────────────────────────
# 3. OPENCV — per-frame visual analysis
# ─────────────────────────────────────────
def analyze_with_opencv(path, sample_every_n_sec=1):
    cap      = cv2.VideoCapture(path)
    fps      = cap.get(cv2.CAP_PROP_FPS)
    total    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, int(fps * sample_every_n_sec))

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eye_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")

    frame_data = []
    keyframes  = []
    prev_gray  = None
    frame_idx  = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % interval == 0:
            sec = round(frame_idx / fps, 2)
            print(f"  🖼️  Frame {frame_idx}/{total} @ {sec}s")

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lab  = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)

            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
            eyes  = eye_cascade.detectMultiScale(gray, 1.1, 5)
            body  = body_cascade.detectMultiScale(gray, 1.1, 3)

            # Motion + camera direction
            motion      = 0.0
            camera_move = "static"
            if prev_gray is not None:
                flow   = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                motion = float(np.mean(np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)))
                mx     = float(np.mean(flow[..., 0]))
                my     = float(np.mean(flow[..., 1]))
                if abs(mx) > abs(my):
                    camera_move = "pan_right" if mx > 1 else "pan_left" if mx < -1 else "static"
                else:
                    camera_move = "tilt_down" if my > 1 else "tilt_up"  if my < -1 else "static"

            # Visual metrics
            edges       = cv2.Canny(gray, 100, 200)
            edge_dens   = float(edges.mean())
            sharpness   = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            hist        = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist_spread = float(np.std(hist))

            # Color temperature
            b_mean     = float(frame[:, :, 0].mean())
            r_mean     = float(frame[:, :, 2].mean())
            color_temp = "warm" if r_mean > b_mean * 1.1 else "cool" if b_mean > r_mean * 1.1 else "neutral"

            # Rule of thirds brightness map
            h, w   = frame.shape[:2]
            thirds = {
                "top_left":   float(frame[:h//3,        :w//3].mean()),
                "top_center": float(frame[:h//3,        w//3:2*w//3].mean()),
                "top_right":  float(frame[:h//3,        2*w//3:].mean()),
                "mid_left":   float(frame[h//3:2*h//3,  :w//3].mean()),
                "center":     float(frame[h//3:2*h//3,  w//3:2*w//3].mean()),
                "mid_right":  float(frame[h//3:2*h//3,  2*w//3:].mean()),
                "bot_left":   float(frame[2*h//3:,      :w//3].mean()),
                "bot_center": float(frame[2*h//3:,      w//3:2*w//3].mean()),
                "bot_right":  float(frame[2*h//3:,      2*w//3:].mean()),
            }
            brightest_region = max(thirds, key=thirds.get)

            face_positions = [
                {"cx": round((x + fw/2) / w, 2), "cy": round((y + fh/2) / h, 2)}
                for (x, y, fw, fh) in faces
            ]

            # OCR
            text_data = extract_text_from_frame(frame)

            frame_data.append({
                "frame":               frame_idx,
                "sec":                 sec,
                "faces":               len(faces),
                "eyes_detected":       len(eyes),
                "body_detected":       len(body),
                "face_positions":      face_positions,
                "motion_score":        round(motion, 3),
                "camera_move":         camera_move,
                "sharpness":           round(sharpness, 1),
                "edge_density":        round(edge_dens, 3),
                "hue":                 round(float(hsv[:, :, 0].mean()), 1),
                "saturation":          round(float(hsv[:, :, 1].mean()), 1),
                "brightness":          round(float(hsv[:, :, 2].mean()), 1),
                "perceptual_lightness":round(float(lab[:, :, 0].mean()), 1),
                "color_temp":          color_temp,
                "hist_spread":         round(hist_spread, 1),
                "brightest_region":    brightest_region,
                "ocr_text":            text_data["text"],
                "ocr_word_count":      text_data["word_count"],
                "ocr_words":           text_data["words_with_position"],
                "has_text_overlay":    text_data["has_caption"],
                "hook_text":           text_data["hook_text"],
            })
            keyframes.append(frame.copy())
            prev_gray = gray

        frame_idx += 1

    cap.release()
    return frame_data, keyframes

# ─────────────────────────────────────────
# 4. PIL — color palette + mood per frame
# ─────────────────────────────────────────
def analyze_with_pil(path, sample_every_n_sec=3):
    cap       = cv2.VideoCapture(path)
    fps       = cap.get(cv2.CAP_PROP_FPS)
    interval  = max(1, int(fps * sample_every_n_sec))
    frame_idx = 0
    pil_data  = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % interval == 0:
            img  = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            stat = ImageStat.Stat(img)
            small     = img.resize((50, 50))
            quantized = small.quantize(colors=6).convert("RGB")
            palette   = sorted(quantized.getcolors(maxcolors=2500) or [], key=lambda x: -x[0])
            dominant  = [list(p[1]) for p in palette[:5]]

            r, g, b = stat.mean[:3]
            if   r > 150 and g < 100 and b < 100: mood = "energetic_red"
            elif b > 150 and r < 120:              mood = "calm_cool"
            elif g > 140 and r < 130:              mood = "natural_green"
            elif r > 160 and g > 140 and b < 100: mood = "warm_golden"
            elif stat.mean[0] < 60:                mood = "dark_moody"
            elif stat.mean[0] > 200:               mood = "bright_clean"
            else:                                  mood = "neutral"

            pil_data.append({
                "sec":                round(frame_idx / fps, 2),
                "mean_rgb":           [round(x, 1) for x in stat.mean[:3]],
                "stddev_rgb":         [round(x, 1) for x in stat.stddev[:3]],
                "dominant_colors":    dominant,
                "perceived_brightness": round(0.299*r + 0.587*g + 0.114*b, 1),
                "color_mood":         mood,
                "contrast_score":     round(float(np.mean(stat.stddev[:3])), 1),
            })

        frame_idx += 1

    cap.release()
    return pil_data

# ─────────────────────────────────────────
# 5. STORYBOARD generator
# ─────────────────────────────────────────
def generate_storyboard(keyframes, frame_data, output_path, cols=4):
    if not keyframes:
        print("⚠️ No keyframes to generate storyboard")
        return

    thumb_w, thumb_h = 360, 640
    label_h = 90
    rows    = math.ceil(len(keyframes) / cols)
    board   = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + label_h)), (10, 10, 10))
    draw    = ImageDraw.Draw(board)

    try:
        font       = ImageFont.truetype("arial.ttf", 14)
        font_small = ImageFont.truetype("arial.ttf", 11)
        font_bold  = ImageFont.truetype("arialbd.ttf", 15)
    except:
        font = font_small = font_bold = ImageFont.load_default()

    for i, (frame, fd) in enumerate(zip(keyframes, frame_data)):
        col = i % cols
        row = i // cols
        x   = col * thumb_w
        y   = row * (thumb_h + label_h)

        img      = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img      = img.resize((thumb_w, thumb_h))
        img_draw = ImageDraw.Draw(img)

        # Draw face circles
        for fp in fd.get("face_positions", []):
            fx = int(fp["cx"] * thumb_w)
            fy = int(fp["cy"] * thumb_h)
            img_draw.ellipse([fx-12, fy-12, fx+12, fy+12], outline=(0, 255, 80), width=3)

        # Draw OCR word boxes
        for word in fd.get("ocr_words", [])[:5]:
            wx = int(word["x"] * thumb_w / frame.shape[1])
            wy = int(word["y"] * thumb_h / frame.shape[0])
            img_draw.rectangle([wx, wy, wx+60, wy+16], outline=(255, 220, 0), width=1)

        board.paste(img, (x, y))

        # Label panel
        ly       = y + thumb_h
        sec      = fd["sec"]
        motion   = fd["motion_score"]
        faces    = fd["faces"]
        cam      = fd["camera_move"]
        temp     = fd["color_temp"]
        bright   = fd["brightness"]
        hook     = fd["hook_text"][:38] + "…" if len(fd["hook_text"]) > 38 else fd["hook_text"]
        region   = fd["brightest_region"]
        has_text = "✅" if fd["has_text_overlay"] else "❌"

        draw.rectangle([x, ly, x + thumb_w, ly + label_h], fill=(18, 18, 25))
        draw.text((x+5, ly+4),  f"⏱ {sec}s   📷 {cam}   🌡 {temp}",                   font=font,       fill=(160, 210, 255))
        draw.text((x+5, ly+22), f"👤 {faces} face   💨 motion {motion:.1f}   💡 {bright:.0f}", font=font, fill=(200, 200, 200))
        draw.text((x+5, ly+40), f"📝 {has_text}  {hook}",                               font=font_small, fill=(255, 220, 80))
        draw.text((x+5, ly+58), f"🎨 focus: {region}",                                 font=font_small, fill=(140, 140, 160))

        # Frame number badge
        draw.rectangle([x, y, x+42, y+22], fill=(220, 50, 50))
        draw.text((x+5, y+4), f"#{i+1}", font=font_bold, fill="white")

    board.save(output_path, quality=95)
    print(f"🎬 Storyboard saved: {output_path}  ({len(keyframes)} frames, {cols} cols)")

# ─────────────────────────────────────────
# 6. AGGREGATE — summary statistics
# ─────────────────────────────────────────
def aggregate(frame_data, pil_data):
    if not frame_data:
        return {}

    def avg(key):
        vals = [f[key] for f in frame_data if isinstance(f.get(key), (int, float))]
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
            "talking_head"      if face_ratio > 0.6         else
            "text_educational"  if text_ratio > 0.5         else
            "broll_cinematic"   if avg("motion_score") < 2  else
            "action_dynamic"    if avg("motion_score") > 8  else
            "mixed"
        ),
    }

# ─────────────────────────────────────────
# RUN ALL
# ─────────────────────────────────────────
print(f"\n🎬 Analyzing: {VIDEO_PATH}\n")

print("📦 Step 1/4 — MoviePy (container + audio)...")
moviepy_data = analyze_with_moviepy(VIDEO_PATH)

print("\n🖼️  Step 2/4 — OpenCV + Tesseract (frame analysis)...")
opencv_frames, kframes = analyze_with_opencv(VIDEO_PATH)

print("\n🎨 Step 3/4 — PIL (color palette)...")
pil_frames = analyze_with_pil(VIDEO_PATH)

print("\n📊 Step 4/4 — Aggregating...")
summary = aggregate(opencv_frames, pil_frames)

storyboard_path = os.path.join(OUTPUT_DIR, "storyboard.jpg")
generate_storyboard(kframes, opencv_frames, storyboard_path)

report = {
    "file":           VIDEO_PATH,
    "container":      moviepy_data,
    "visual_summary": summary,
    "frame_analysis": opencv_frames,
    "color_analysis": pil_frames,
}
report_path = os.path.join(OUTPUT_DIR, "video_report.json")
with open(report_path, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print("\n" + "="*60)
print("📦 CONTAINER (MoviePy)")
print("="*60)
print(json.dumps(moviepy_data, indent=2))

print("\n" + "="*60)
print("🧠 VISUAL SUMMARY")
print("="*60)
print(json.dumps(summary, indent=2))

print("\n" + "="*60)
print("📝 ALL OCR TEXT FOUND")
print("="*60)
print(summary.get("all_ocr_text") or "(no text detected)")

print(f"\n✅ Storyboard  → {storyboard_path}")
print(f"✅ Full report → {report_path}")