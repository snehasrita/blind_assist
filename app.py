"""
BlindAssist Web Server
Flask backend for web interface
"""

from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import time
import threading
import queue
import json
import base64
from collections import deque, Counter
import sys
import os

# Import existing modules
try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] Run: pip install ultralytics")
    sys.exit(1)

# Configuration
MODEL_PATH = "yolov8n.pt"
CONFIDENCE = 0.45
ANNOUNCE_COOLDOWN = 4.0
DANGER_COOLDOWN = 2.0

DANGEROUS = {
    "car","truck","bus","motorcycle","bicycle",
    "dog","cat","knife","scissors","train","horse","fire hydrant"
}

# Colors (BGR)
C = {
    "bg": (15, 17, 26),
    "panel": (22, 26, 40),
    "card": (32, 38, 56),
    "border": (55, 62, 88),
    "danger": (45, 45, 220),
    "warn": (30, 140, 255),
    "safe": (80, 200, 80),
    "blue": (220, 160, 50),
    "white": (235, 235, 240),
    "gray": (140, 145, 165),
    "dim": (75, 80, 100),
    "flash": (30, 30, 180),
}

app = Flask(__name__)
app.config['SECRET_KEY'] = 'blind_assist_secret'
socketio = SocketIO(app, cors_allowed_origins="*")

class BlindAssistWeb:
    def __init__(self):
        self.conf = CONFIDENCE
        self.mode = "detect"
        self.last_said = {}
        self.vlog = deque(maxlen=30)
        self.frame_queue = queue.Queue(maxsize=1)
        self.running = False
        self.camera_thread = None
        
        # Load YOLO model
        print("[1/2] Loading YOLO model …")
        if not os.path.exists(MODEL_PATH):
            print(f"      Downloading {MODEL_PATH} (first run only) …")
        self.model = YOLO(MODEL_PATH)
        print(f"      Model ready: {MODEL_PATH}")
        
        # Initialize camera
        print("[2/2] Opening camera …")
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("[ERR] Camera not found!")
            return
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        print("      Camera ready")

    def _direction(self, cx, w):
        r = cx / w
        if r < 0.20: return "far left"
        if r < 0.40: return "to your left"
        if r < 0.60: return "straight ahead"
        if r < 0.80: return "to your right"
        return "far right"

    def _distance(self, bh, label):
        ref = {"person":170,"car":150,"truck":250,"bus":280,
               "motorcycle":110,"bicycle":100,"dog":60,"cat":30}.get(label,100)
        if bh == 0: return "far away"
        d = (ref * 700) / bh
        if d < 80:  return "very close"
        if d < 200: return "nearby"
        if d < 450: return "ahead"
        return "far away"

    def _zone(self, dist):
        return {"very close":"danger","nearby":"warn","ahead":"safe","far away":"safe"}.get(dist,"safe")

    def _can_say(self, label, dangerous):
        cd = DANGER_COOLDOWN if dangerous else ANNOUNCE_COOLDOWN
        now = time.time()
        if now - self.last_said.get(label,0) > cd:
            self.last_said[label] = now
            return True
        return False

    def _process_frame(self, frame):
        h, w = frame.shape[:2]
        dets = []
        results = self.model(frame, conf=self.conf, verbose=False)
        
        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                if conf < self.conf: continue
                cls = int(box.cls[0])
                name = self.model.names[cls]
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                bh = y2-y1
                cx = (x1+x2)//2

                dist = self._distance(bh, name)
                zone = self._zone(dist)
                direc = self._direction(cx, w)
                dang = name in DANGEROUS

                dets.append({
                    "label": name,
                    "conf": conf,
                    "box": [x1, y1, x2, y2],
                    "center": [cx, (y1+y2)//2],
                    "distance": dist,
                    "zone": zone,
                    "direction": direc,
                    "dangerous": dang
                })

                # Emit detection to web clients
                if self._can_say(name, dang):
                    pfx = "Warning! " if dang else ""
                    message = f"{pfx}{name}, {direc}, {dist}"
                    self.vlog.append((time.time(), message))
                    socketio.emit('detection', {
                        'message': message,
                        'dangerous': dang,
                        'object': name,
                        'direction': direc,
                        'distance': dist,
                        'confidence': conf
                    })

        dets.sort(key=lambda d:(0 if d["dangerous"] else 1,
                                ["very close","nearby","ahead","far away"].index(d["distance"])))
        
        return frame, dets, results

    def _draw_detections(self, frame, dets):
        for d in dets:
            x1,y1,x2,y2 = d["box"]
            col = C["danger"] if d["zone"] == "danger" else (C["warn"] if d["zone"] == "warn" else C["safe"])
            thick = 3 if d["dangerous"] else 2

            cv2.rectangle(frame, (x1,y1), (x2,y2), col, thick)
            cv2.circle(frame, tuple(d["center"]), 5, col, -1)

            # Label
            lbl = f"{d['label'].upper()}  {d['conf']*100:.0f}%"
            (tw,th),_ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            ly = max(y1-8, th+12)
            cv2.rectangle(frame, (x1, ly-th-6), (x1+tw+12, ly+4), C["panel"], -1)
            cv2.rectangle(frame, (x1, ly-th-6), (x1+tw+12, ly+4), col, 1)
            cv2.putText(frame, lbl, (x1+6, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)

            # Distance badge
            bc = C["danger"] if d["zone"]=="danger" else (C["warn"] if d["zone"]=="warn" else C["safe"])
            cv2.putText(frame, f"{d['distance']}  {d['direction']}", (x1, y2+17), cv2.FONT_HERSHEY_SIMPLEX, 0.4, bc, 1, cv2.LINE_AA)

        return frame

    def camera_loop(self):
        """Background thread for camera processing"""
        fc, fps, t0 = 0, 0.0, time.time()
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            fc += 1
            if fc % 30 == 0:
                e = time.time() - t0
                fps = 30/e if e > 0 else fps
                t0 = time.time()

            frame, dets, results = self._process_frame(frame)
            frame = self._draw_detections(frame, dets)

            # Encode frame for web streaming
            _, buffer = cv2.imencode('.jpg', frame)
            frame_data = base64.b64encode(buffer).decode('utf-8')

            # Send frame to web clients
            socketio.emit('frame', {
                'image': frame_data,
                'fps': fps,
                'detections': dets,
                'mode': self.mode,
                'confidence': self.conf
            })

            # Auto scene narrate
            if self.mode == "scene" and fc % 90 == 0:
                self._describe_scene(results)

            time.sleep(0.033)  # ~30 FPS

    def _describe_scene(self, results):
        counts = Counter()
        for r in results:
            for box in r.boxes:
                if float(box.conf[0]) >= self.conf:
                    counts[self.model.names[int(box.cls[0])]] += 1
        if not counts:
            message = "Scene is clear, no objects detected."
        else:
            parts = [f"{n} {l}s" if n>1 else f"a {l}" for l,n in counts.most_common(5)]
            if len(parts)==1:   s=parts[0]
            elif len(parts)==2: s=f"{parts[0]} and {parts[1]}"
            else:               s=", ".join(parts[:-1])+f", and {parts[-1]}"
            message = f"I can see {s}."
        
        self.vlog.append((time.time(), message))
        socketio.emit('scene_description', {'message': message})

    def start(self):
        """Start the camera thread"""
        self.running = True
        self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
        self.camera_thread.start()

    def stop(self):
        """Stop the camera thread"""
        self.running = False
        if self.camera_thread:
            self.camera_thread.join()

# Global instance
blind_assist = BlindAssistWeb()

@app.route('/')
def index():
    return render_template('impressive.html')

@app.route('/classic')
def classic():
    return render_template('index.html')

@app.route('/api/status')
def status():
    return jsonify({
        'running': blind_assist.running,
        'mode': blind_assist.mode,
        'confidence': blind_assist.conf,
        'camera_active': blind_assist.cap.isOpened() if blind_assist.cap else False
    })

@app.route('/api/settings', methods=['POST'])
def update_settings():
    data = request.json
    if 'confidence' in data:
        blind_assist.conf = max(0.10, min(0.95, float(data['confidence'])))
    if 'mode' in data:
        blind_assist.mode = data['mode']
        socketio.emit('mode_change', {'mode': blind_assist.mode})
    return jsonify({'success': True})

@app.route('/api/describe_scene', methods=['POST'])
def describe_scene():
    # Trigger scene description
    if blind_assist.cap and blind_assist.cap.isOpened():
        ret, frame = blind_assist.cap.read()
        if ret:
            _, _, results = blind_assist._process_frame(frame)
            blind_assist._describe_scene(results)
    return jsonify({'success': True})

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('status', {
        'running': blind_assist.running,
        'mode': blind_assist.mode,
        'confidence': blind_assist.conf
    })

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

if __name__ == '__main__':
    blind_assist.start()
    try:
        socketio.run(app, host='0.0.0.0', port=5000, debug=False)
    finally:
        blind_assist.stop()
