import cv2
import numpy as np
from PIL import Image, ImageStat
from moviepy import VideoFileClip
import json
import os

VIDEO_PATH = r"C:\Users\mailt\Downloads\reel_33000likes_AQP1mge7e9VA5c-Ed_1b.mp4"  # point to any downloaded reel

# ─────────────────────────────────────────
# 1. MOVIEPY — container-level metadata
# ─────────────────────────────────────────
def analyze_with_moviepy(path):
    clip = VideoFileClip(path)
    audio = clip.audio

    data = {
        "duration_sec": round(clip.duration, 2),
        "fps": clip.fps,
        "resolution": clip.size,          # (width, height)
        "aspect_ratio": round(clip.size[0] / clip.size[1], 3),
        "total_frames": int(clip.duration * clip.fps),
        "has_audio": audio is not None,
    }

    if audio:
        # Sample audio waveform energy
        samples = audio.to_soundarray(fps=22050)
        data["audio_channels"] = samples.shape[1] if samples.ndim > 1 else 1
        data["audio_rms"] = round(float(np.sqrt(np.mean(samples**2))), 4)
        data["audio_peak"] = round(float(np.max(np.abs(samples))), 4)
        data["audio_energy_per_sec"] = round(float(np.mean(np.abs(samples))), 4)

    clip.close()
    return data

# ─────────────────────────────────────────
# 2. OPENCV — frame-level visual analysis
# ─────────────────────────────────────────
def analyze_with_opencv(path, sample_every_n_sec=1):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, int(fps * sample_every_n_sec))

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    
    frame_data = []
    prev_gray = None
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lab  = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)

            # Faces
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)

            # Motion score (optical flow between frames)
            motion = 0.0
            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                       0.5, 3, 15, 3, 5, 1.2, 0)
                motion = float(np.mean(np.sqrt(flow[...,0]**2 + flow[...,1]**2)))

            # Edge density (how busy / text-heavy the frame is)
            edges = cv2.Canny(gray, 100, 200)
            edge_density = float(edges.mean())

            # Sharpness (Laplacian variance — low = blurry)
            sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

            # Color palette dominance (HSV channels)
            h_mean = float(hsv[:,:,0].mean())
            s_mean = float(hsv[:,:,1].mean())   # saturation
            v_mean = float(hsv[:,:,2].mean())   # brightness

            # Lab for perceptual lightness
            l_mean = float(lab[:,:,0].mean())

            # Rule of thirds — where are the faces?
            face_positions = []
            for (x, y, w, h) in faces:
                cx = (x + w/2) / frame.shape[1]  # normalized 0-1
                cy = (y + h/2) / frame.shape[0]
                face_positions.append({"cx": round(cx, 2), "cy": round(cy, 2)})

            frame_data.append({
                "frame": frame_idx,
                "sec": round(frame_idx / fps, 2),
                "faces": len(faces),
                "face_positions": face_positions,
                "motion_score": round(motion, 3),
                "edge_density": round(edge_density, 3),
                "sharpness": round(sharpness, 1),
                "hue": round(h_mean, 1),
                "saturation": round(s_mean, 1),
                "brightness": round(v_mean, 1),
                "perceptual_lightness": round(l_mean, 1),
            })

            prev_gray = gray

        frame_idx += 1

    cap.release()
    return frame_data

# ─────────────────────────────────────────
# 3. PIL / PILLOW — color + histogram
# ─────────────────────────────────────────
def analyze_with_pil(path, sample_every_n_sec=3):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = max(1, int(fps * sample_every_n_sec))
    frame_idx = 0
    pil_data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % interval == 0:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            stat = ImageStat.Stat(img)

            # Dominant color via quantization
            small = img.resize((50, 50))
            quantized = small.quantize(colors=5).convert("RGB")
            palette = quantized.getcolors(maxcolors=2500)
            palette_sorted = sorted(palette, key=lambda x: -x[0]) if palette else []
            dominant = [p[1] for p in palette_sorted[:3]]  # top 3 RGB tuples

            pil_data.append({
                "sec": round(frame_idx / fps, 2),
                "mean_rgb": [round(x, 1) for x in stat.mean],
                "stddev_rgb": [round(x, 1) for x in stat.stddev],
                "median_rgb": [round(x, 1) for x in stat.median],
                "dominant_colors": dominant,
                "perceived_brightness": round(
                    0.299 * stat.mean[0] + 0.587 * stat.mean[1] + 0.114 * stat.mean[2], 1
                )
            })

        frame_idx += 1

    cap.release()
    return pil_data

# ─────────────────────────────────────────
# 4. AGGREGATE — summary statistics
# ─────────────────────────────────────────
def aggregate(frame_data):
    if not frame_data:
        return {}

    def avg(key): return round(np.mean([f[key] for f in frame_data]), 3)
    def mx(key):  return round(np.max([f[key] for f in frame_data]), 3)

    total_frames_with_face = sum(1 for f in frame_data if f["faces"] > 0)
    face_ratio = round(total_frames_with_face / len(frame_data), 3)
    cut_frames = [f for f in frame_data if f["motion_score"] > 15]  # hard cuts

    return {
        "avg_brightness": avg("brightness"),
        "avg_saturation": avg("saturation"),
        "avg_sharpness": avg("sharpness"),
        "avg_motion": avg("motion_score"),
        "peak_motion": mx("motion_score"),
        "avg_edge_density": avg("edge_density"),
        "face_presence_ratio": face_ratio,       # 0 = no faces, 1 = face every frame
        "estimated_cuts": len(cut_frames),        # high motion spikes = scene cuts
        "pacing": "fast" if len(cut_frames) > 8 else "medium" if len(cut_frames) > 3 else "slow",
        "style": (
            "talking_head" if face_ratio > 0.6 else
            "broll_cinematic" if avg("motion_score") < 2 else
            "action_dynamic" if avg("motion_score") > 8 else
            "mixed"
        )
    }

# ─────────────────────────────────────────
# RUN EVERYTHING
# ─────────────────────────────────────────
print("🎬 Analyzing:", VIDEO_PATH)

moviepy_data = analyze_with_moviepy(VIDEO_PATH)
opencv_frames = analyze_with_opencv(VIDEO_PATH)
pil_frames    = analyze_with_pil(VIDEO_PATH)
summary       = aggregate(opencv_frames)

report = {
    "file": VIDEO_PATH,
    "container": moviepy_data,
    "visual_summary": summary,
    "frame_analysis": opencv_frames,
    "color_analysis": pil_frames,
}

with open("video_report.json", "w") as f:
    json.dump(report, f, indent=2)

print("\n📦 CONTAINER (MoviePy)")
print(json.dumps(moviepy_data, indent=2))

print("\n🧠 VISUAL SUMMARY (OpenCV)")
print(json.dumps(summary, indent=2))

print("\n🎨 COLOR SAMPLE @ 0s (PIL)")
print(json.dumps(pil_frames[0] if pil_frames else {}, indent=2))

print("\n✅ Full report saved to video_report.json")