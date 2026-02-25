"""
utils/config.py — Central configuration for BlindAssist
"""

CONFIG = {
    # ── Model ──────────────────────────────────────────────────────────────
    # Options: yolov8n.pt (fastest) | yolov8s.pt | yolov8m.pt | yolov8l.pt (most accurate)
    "model_path": "yolov8n.pt",

    # ── Camera ─────────────────────────────────────────────────────────────
    "camera_source": 0,          # 0 = default webcam | 1 = external | "video.mp4" for file
    "frame_width":  1280,
    "frame_height":  720,

    # ── Detection ──────────────────────────────────────────────────────────
    "confidence_threshold": 0.45,   # lower = more detections, higher = fewer false positives

    # ── Audio / TTS ────────────────────────────────────────────────────────
    "tts_rate":     160,             # words per minute (default ~200)
    "tts_volume":   1.0,             # 0.0 – 1.0
    "announcement_cooldown": 4.0,   # seconds between repeating the same object label

    # ── Distance thresholds (fraction of frame height) ─────────────────────
    # Larger bounding box → closer object
    "dist_near_ratio":   0.40,      # bbox_h / frame_h > this  → "very close"
    "dist_medium_ratio": 0.20,      # bbox_h / frame_h > this  → "nearby"
                                    # else                      → "far away"

    # ── Danger zone (horizontal) ───────────────────────────────────────────
    # Objects whose center falls within [danger_left .. danger_right] (0-1 of width)
    # are flagged as directly in the path
    "danger_left":  0.30,
    "danger_right": 0.70,

    # ── Priority objects — announced more aggressively ──────────────────────
    "priority_labels": [
        "person", "car", "truck", "bus", "motorcycle", "bicycle",
        "dog", "cat", "stairs", "chair", "door", "traffic light",
        "fire hydrant", "stop sign",
    ],

    # ── Display ─────────────────────────────────────────────────────────────
    "show_fps":         True,
    "show_distance":    True,
    "show_direction":   True,
    "show_danger_zone": True,
    "font_scale":       0.6,
}
