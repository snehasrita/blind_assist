"""
BlindAssist - AI Object Detection System for Visually Impaired
==============================================================
Run:  python main.py
Keys: Q=Quit  S=Scene mode  SPACE=Describe scene  +/-=Confidence
"""

import cv2
import numpy as np
import time
import threading
import queue
import sys
import os
import subprocess
from collections import deque, Counter

# ── YOLOv8 ────────────────────────────────────────────────────────────────
try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] Run: pip install ultralytics")
    sys.exit(1)

# ══════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════
MODEL_PATH        = "yolov8n.pt"   # yolov8s.pt = better accuracy
CAMERA_SOURCE     = 0
CONFIDENCE        = 0.45
ANNOUNCE_COOLDOWN = 4.0            # seconds between same-object announcements
DANGER_COOLDOWN   = 2.0
PANEL_W           = 300            # right sidebar width
VOICE_LOG_SIZE    = 7

DANGEROUS = {
    "car","truck","bus","motorcycle","bicycle",
    "dog","cat","knife","scissors","train","horse","fire hydrant"
}

# ── Colours (BGR) ─────────────────────────────────────────────────────────
C = {
    "bg":      ( 15,  17,  26),
    "panel":   ( 22,  26,  40),
    "card":    ( 32,  38,  56),
    "border":  ( 55,  62,  88),
    "danger":  ( 45,  45, 220),
    "warn":    ( 30, 140, 255),
    "safe":    ( 80, 200,  80),
    "blue":    (220, 160,  50),
    "white":   (235, 235, 240),
    "gray":    (140, 145, 165),
    "dim":     ( 75,  80, 100),
    "flash":   ( 30,  30, 180),
}
ZCOL = {"danger": C["danger"], "warn": C["warn"], "safe": C["safe"]}
FONT  = cv2.FONT_HERSHEY_SIMPLEX
FONTD = cv2.FONT_HERSHEY_DUPLEX


# ══════════════════════════════════════════════════════════════════════════
# VOICE ENGINE
# ══════════════════════════════════════════════════════════════════════════
class VoiceEngine:
    """
    Background TTS thread.
    - Windows: uses built-in PowerShell SAPI (no install needed)
    - Other OS: uses pyttsx3
    """
    def __init__(self):
        self._q = queue.Queue()
        self._method = self._pick_method()
        print(f"[VOICE] Engine: {self._method}")
        threading.Thread(target=self._worker, daemon=True, name="TTS").start()

    def _pick_method(self):
        if sys.platform == "win32":
            return "sapi"
        try:
            import pyttsx3
            return "pyttsx3"
        except ImportError:
            return "none"

    def _worker(self):
        eng = None
        if self._method == "pyttsx3":
            import pyttsx3
            eng = pyttsx3.init()
            eng.setProperty("rate", 155)
            eng.setProperty("volume", 1.0)

        while True:
            text = self._q.get()
            if text is None:
                break
            print(f"  >> VOICE: {text}")
            try:
                if self._method == "sapi":
                    safe = text.replace('"', "").replace("'", "")
                    ps = (
                        "Add-Type -AssemblyName System.Speech;"
                        "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer;"
                        "$s.Rate=0;$s.Volume=100;"
                        f'$s.Speak("{safe}");'
                    )
                    subprocess.run(
                        ["powershell", "-WindowStyle", "Hidden", "-Command", ps],
                        timeout=15, capture_output=True
                    )
                elif self._method == "pyttsx3" and eng:
                    eng.say(text)
                    eng.runAndWait()
            except Exception as e:
                print(f"  [VOICE ERR] {e}")

    def say(self, text: str):
        if self._q.qsize() < 2:
            self._q.put(text)

    def stop(self):
        self._q.put(None)


# ══════════════════════════════════════════════════════════════════════════
# UI HELPERS
# ══════════════════════════════════════════════════════════════════════════
def _fill(img, x1, y1, x2, y2, col, alpha=1.0):
    if alpha < 1.0:
        ov = img.copy()
        cv2.rectangle(ov, (x1,y1),(x2,y2), col, -1)
        cv2.addWeighted(ov, alpha, img, 1-alpha, 0, img)
    else:
        cv2.rectangle(img, (x1,y1),(x2,y2), col, -1)

def _txt(img, t, x, y, col=None, sc=0.5, th=1, font=FONT):
    cv2.putText(img, t, (x,y), font, sc, col or C["white"], th, cv2.LINE_AA)

def _badge(img, t, x, y, bg, fg=None):
    fg = fg or C["white"]
    (tw,th),_ = cv2.getTextSize(t, FONT, 0.42, 1)
    cv2.rectangle(img, (x, y-th-4),(x+tw+8, y+4), bg, -1)
    cv2.putText(img, t, (x+4,y), FONT, 0.42, fg, 1, cv2.LINE_AA)

def _corner_box(img, x1,y1,x2,y2, col, thick=2):
    """Draw bounding box with stylish corner markers."""
    cv2.rectangle(img,(x1,y1),(x2,y2), col, thick)
    L = 16
    pts = [(x1,y1,x1+L,y1),(x2-L,y1,x2,y1),
           (x1,y2-L,x1,y2),(x2,y2-L,x2,y2),   # wait, fix below
           ]
    for sx,sy,ex,ey in [(x1,y1,x1+L,y1),(x2-L,y1,x2,y1),
                         (x1,y2,x1+L,y2),(x2-L,y2,x2,y2)]:
        cv2.line(img,(sx,sy),(ex,ey),col,3)
    for sx,sy,ex,ey in [(x1,y1,x1,y1+L),(x2,y1,x2,y1+L),
                         (x1,y2,x1,y2-L),(x2,y2,x2,y2-L)]:
        cv2.line(img,(sx,sy),(ex,ey),col,3)


# ══════════════════════════════════════════════════════════════════════════
# DRAW DETECTIONS ON CAMERA FRAME
# ══════════════════════════════════════════════════════════════════════════
def draw_detections(frame, dets, flash):
    for d in dets:
        x1,y1,x2,y2 = d["box"]
        col  = ZCOL.get(d["zone"], C["safe"])
        thick = 3 if d["dangerous"] else 2

        if d["zone"] == "danger" and flash:
            _fill(frame, x1,y1,x2,y2, C["flash"], 0.22)

        _corner_box(frame, x1,y1,x2,y2, col, thick)

        cx,cy = d["center"]
        cv2.circle(frame,(cx,cy), 5, col, -1)

        # Name + conf label
        lbl = f"{d['label'].upper()}  {d['conf']*100:.0f}%"
        (tw,th),_ = cv2.getTextSize(lbl, FONT, 0.5, 1)
        ly = max(y1-8, th+12)
        _fill(frame, x1, ly-th-6, x1+tw+12, ly+4, C["panel"], 0.85)
        cv2.rectangle(frame,(x1,ly-th-6),(x1+tw+12,ly+4), col, 1)
        _txt(frame, lbl, x1+6, ly, col, 0.5, 1)

        # Distance badge below box
        bc = C["danger"] if d["zone"]=="danger" else (
             C["warn"]   if d["zone"]=="warn"   else C["safe"])
        _badge(frame, f"{d['distance']}  {d['direction']}", x1, y2+17, bc)


# ══════════════════════════════════════════════════════════════════════════
# BUILD FULL UI CANVAS (camera + right panel)
# ══════════════════════════════════════════════════════════════════════════
def build_ui(cam, dets, fps, mode, conf, vlog, flash):
    h, w = cam.shape[:2]
    tw   = w + PANEL_W
    canvas = np.full((h, tw, 3), C["bg"], dtype=np.uint8)
    canvas[:, :w] = cam

    # Path guide lines
    lx,rx = int(w*0.30), int(w*0.70)
    cv2.line(canvas,(lx,50),(lx,h), C["border"],1)
    cv2.line(canvas,(rx,50),(rx,h), C["border"],1)
    _txt(canvas,"SAFE PATH",lx+8,h-12,C["dim"],0.4)

    # ── TOP BAR ────────────────────────────────────────────────────────────
    _fill(canvas,0,0,tw,50, C["panel"])
    cv2.line(canvas,(0,50),(tw,50),C["border"],1)

    cv2.putText(canvas,"BLIND", (12,33), FONTD, 0.75, C["blue"],  2, cv2.LINE_AA)
    cv2.putText(canvas,"ASSIST",(80,33), FONTD, 0.75, C["white"], 1, cv2.LINE_AA)
    _txt(canvas,f"FPS {fps:5.1f}", w-310, 33, C["gray"], 0.5)
    _txt(canvas,f"CONF {conf:.2f}", w-205, 33, C["gray"], 0.5)
    mc = C["blue"] if mode=="detect" else C["warn"]
    _txt(canvas,f"MODE {mode.upper()}", w-115, 33, mc, 0.5)

    # ── RIGHT PANEL ────────────────────────────────────────────────────────
    px = w
    _fill(canvas,px,0,tw,h, C["panel"])
    cv2.line(canvas,(px,0),(px,h), C["border"],1)

    y = 62

    # STATUS CARD
    danger_dets = [d for d in dets if d["zone"]=="danger" or d["dangerous"]]
    warn_dets   = [d for d in dets if d["zone"]=="warn" and not d["dangerous"]]

    if danger_dets and flash:
        sc, st, ss = C["danger"], "!! DANGER !!", ", ".join(set(d["label"] for d in danger_dets[:2]))
    elif danger_dets:
        sc, st, ss = C["danger"], "DANGER",       ", ".join(set(d["label"] for d in danger_dets[:2]))
    elif warn_dets:
        sc, st, ss = C["warn"],   "CAUTION",      ", ".join(set(d["label"] for d in warn_dets[:2]))
    elif dets:
        sc, st, ss = C["safe"],   "ALL CLEAR",    "Path looks safe"
    else:
        sc, st, ss = C["dim"],    "SCANNING ...", "No objects found"

    _fill(canvas,px+10,y,tw-10,y+72, C["card"])
    cv2.rectangle(canvas,(px+10,y),(tw-10,y+72),sc,2)
    _fill(canvas,px+10,y,px+16,y+72,sc)    # accent bar
    cv2.putText(canvas,st,(px+22,y+30),FONTD,0.65,sc,2,cv2.LINE_AA)
    _txt(canvas,ss[:28],px+22,y+54,C["gray"],0.43)
    y += 84

    # COUNTS
    _txt(canvas,"DETECTED",px+12,y,C["dim"],0.4)
    y += 18
    nd,nw,ns = len(danger_dets), len(warn_dets), len(dets)-len(danger_dets)-len(warn_dets)
    for i,(lbl,val,col) in enumerate([("DANGER",nd,C["danger"]),
                                       ("NEARBY",nw,C["warn"]),
                                       ("SAFE",  ns,C["safe"])]):
        bx = px+12 + i*86
        _fill(canvas,bx,y,bx+80,y+44,C["card"])
        cv2.rectangle(canvas,(bx,y),(bx+80,y+44),col,1)
        cv2.putText(canvas,str(val),(bx+12,y+30),FONTD,0.85,col,2,cv2.LINE_AA)
        _txt(canvas,lbl,bx+10,y+42,C["dim"],0.32)
    y += 58

    # OBJECT LIST
    _txt(canvas,"OBJECT LIST",px+12,y,C["dim"],0.4)
    y += 16
    cv2.line(canvas,(px+10,y),(tw-10,y),C["border"],1)
    y += 8

    for d in (dets[:7] if dets else []):
        col = ZCOL.get(d["zone"],C["safe"])
        _fill(canvas,px+10,y-2,tw-10,y+26,C["card"],0.5)
        cv2.circle(canvas,(px+20,y+12),5,col,-1)
        _txt(canvas,d["label"].capitalize()[:13],px+32,y+17,C["white"],0.5)
        bc = C["danger"] if d["zone"]=="danger" else (C["warn"] if d["zone"]=="warn" else C["safe"])
        _badge(canvas,d["distance"],px+148,y+17,bc)
        _txt(canvas,d["direction"][:11],px+210,y+17,C["gray"],0.38)
        y += 30

    if not dets:
        _txt(canvas,"  Nothing detected yet",px+12,y+16,C["dim"],0.43)
        y += 30

    y += 8
    cv2.line(canvas,(px+10,y),(tw-10,y),C["border"],1)
    y += 12

    # VOICE LOG
    _txt(canvas,"VOICE LOG",px+12,y,C["dim"],0.4)
    y += 18
    for ts,msg in list(vlog)[-VOICE_LOG_SIZE:]:
        age = time.time()-ts
        fc  = C["white"] if age<3 else C["gray"] if age<9 else C["dim"]
        _txt(canvas,("  "+msg)[:34],px+10,y,fc,0.41)
        y += 18

    y += 6
    cv2.line(canvas,(px+10,y),(tw-10,y),C["border"],1)
    y += 12

    # CONTROLS
    _txt(canvas,"CONTROLS",px+12,y,C["dim"],0.4)
    y += 16
    for k,desc in [("Q","Quit"),("S","Toggle scene"),
                   ("SPC","Describe now"),("+/-","Confidence")]:
        _fill(canvas,px+12,y,px+50,y+18,C["card"])
        cv2.rectangle(canvas,(px+12,y),(px+50,y+18),C["border"],1)
        _txt(canvas,k,  px+16,y+14,C["blue"],0.38,1)
        _txt(canvas,desc,px+56,y+14,C["gray"],0.4,1)
        y += 22

    # CONFIDENCE BAR (bottom)
    by = h-28
    _txt(canvas,f"CONFIDENCE  {conf:.0%}",px+12,by,C["dim"],0.4)
    by += 10
    bw = tw-px-24
    _fill(canvas,px+12,by,px+12+bw,by+8,C["card"])
    _fill(canvas,px+12,by,px+12+int(bw*conf),by+8,C["blue"])
    cv2.rectangle(canvas,(px+12,by),(px+12+bw,by+8),C["border"],1)

    return canvas


# ══════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════
class BlindAssistApp:

    def __init__(self):
        print("\n" + "="*52)
        print("  BlindAssist — AI Object Detection for the Blind")
        print("="*52)
        self.conf      = CONFIDENCE
        self.mode      = "detect"
        self.last_said = {}
        self.vlog      = deque(maxlen=30)

        print("[1/3] Starting voice engine …")
        self.voice = VoiceEngine()

        print("[2/3] Loading YOLO model …")
        if not os.path.exists(MODEL_PATH):
            print(f"      Downloading {MODEL_PATH} (first run only) …")
        self.model = YOLO(MODEL_PATH)
        print(f"      Model ready: {MODEL_PATH}")

        print("[3/3] Opening camera …")
        self.cap = cv2.VideoCapture(CAMERA_SOURCE)
        if not self.cap.isOpened():
            print("[ERR] Camera not found! Try changing CAMERA_SOURCE in code.")
            sys.exit(1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
        print("      Camera ready\n")
        print("  Q=Quit  S=Scene mode  SPACE=Describe  +/-=Confidence\n")

    # ── Helpers ────────────────────────────────────────────────────────────
    def _direction(self, cx, w):
        r = cx/w
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

    def _say(self, text):
        self.voice.say(text)
        self.vlog.append((time.time(), text))

    # ── Process frame ──────────────────────────────────────────────────────
    def _process(self, frame, results):
        h, w = frame.shape[:2]
        dets = []
        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                if conf < self.conf: continue
                cls  = int(box.cls[0])
                name = self.model.names[cls]
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                bh   = y2-y1
                cx   = (x1+x2)//2

                dist = self._distance(bh, name)
                zone = self._zone(dist)
                direc= self._direction(cx, w)
                dang = name in DANGEROUS

                dets.append({"label":name,"conf":conf,"box":(x1,y1,x2,y2),
                             "center":(cx,(y1+y2)//2),"distance":dist,
                             "zone":zone,"direction":direc,"dangerous":dang})

                if self._can_say(name, dang):
                    pfx = "Warning! " if dang else ""
                    self._say(f"{pfx}{name}, {direc}, {dist}")

        dets.sort(key=lambda d:(0 if d["dangerous"] else 1,
                                ["very close","nearby","ahead","far away"].index(d["distance"])))
        return dets

    def _describe_scene(self, results):
        counts = Counter()
        for r in results:
            for box in r.boxes:
                if float(box.conf[0]) >= self.conf:
                    counts[self.model.names[int(box.cls[0])]] += 1
        if not counts:
            self._say("Scene is clear, no objects detected.")
            return
        parts = [f"{n} {l}s" if n>1 else f"a {l}" for l,n in counts.most_common(5)]
        if len(parts)==1:   s=parts[0]
        elif len(parts)==2: s=f"{parts[0]} and {parts[1]}"
        else:               s=", ".join(parts[:-1])+f", and {parts[-1]}"
        self._say(f"I can see {s}.")

    # ── Run ────────────────────────────────────────────────────────────────
    def run(self):
        self._say("Blind assist system active. Scanning ahead.")

        fc,fps,t0 = 0,0.0,time.time()
        results = []

        while True:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.05); continue

            fc += 1
            if fc%30==0:
                e=time.time()-t0; fps=30/e if e>0 else fps; t0=time.time()

            results = self.model(frame, conf=self.conf, verbose=False)
            dets    = self._process(frame, results)

            # Flash on danger
            flash = any(d["zone"]=="danger" or d["dangerous"] for d in dets) \
                    and (int(time.time()*3)%2==0)

            # Auto scene narrate
            if self.mode=="scene" and fc%90==0:
                self._describe_scene(results)

            draw_detections(frame, dets, flash)
            ui = build_ui(frame, dets, fps, self.mode, self.conf, self.vlog, flash)

            cv2.imshow("BlindAssist — AI Vision System", ui)

            key = cv2.waitKey(1) & 0xFF
            if   key==ord("q"): break
            elif key==ord("s"):
                self.mode="scene" if self.mode=="detect" else "detect"
                self._say(f"Switched to {self.mode} mode")
            elif key in(ord("+"),ord("=")):
                self.conf=min(0.95,self.conf+0.05)
            elif key==ord("-"):
                self.conf=max(0.10,self.conf-0.05)
            elif key==ord(" "):
                self._describe_scene(results)

        self._say("Goodbye.")
        time.sleep(1.5)
        self.cap.release()
        cv2.destroyAllWindows()
        self.voice.stop()


if __name__ == "__main__":
    BlindAssistApp().run()
