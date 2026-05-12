# pyrefly: ignore [missing-import]
import cv2
import math
import time
import asyncio
import os
import sqlite3
from collections import defaultdict
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import shutil
from ultralytics import YOLO
import uvicorn
import numpy as np
from datetime import datetime

app = FastAPI(title="UTT SAFETY COMMAND CENTER")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── CẤU HÌNH AI ────────────────────────────────────────────────────────────
PPE_MODEL_PATH  = '../best.pt'
CONE_MODEL_PATH = '../cone_sign.pt'
FALL_MODEL_PATH = '../tuthenga.pt'
VIDEO_PATH      = '../video7.mp4'

TEMP_DIR = "temp_videos"
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

print("[AI] Đang nạp hệ thống 3 Model...")
model_ppe  = YOLO(PPE_MODEL_PATH)
model_cone = YOLO(CONE_MODEL_PATH)
model_fall = YOLO(FALL_MODEL_PATH)
print("[AI] Sẵn sàng! (PPE + Cone/Sign + Fall Detection)")

# ─── SQLite Database ─────────────────────────────────────────────────────────
DB_PATH = "safety_logs.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS logs (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            track_id  INTEGER,
            type      TEXT,
            detail    TEXT,
            camera    TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# ─── In-memory log (50 bản ghi gần nhất) ────────────────────────────────────
system_logs  = []
person_states = defaultdict(lambda: {"last_alert": 0})

def add_log(track_id: int, violation_type: str, detail: str = "", camera: str = "CAM 01"):
    timestamp = datetime.now().strftime('%H:%M:%S')

    # Lưu vào SQLite
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "INSERT INTO logs (timestamp, track_id, type, detail, camera) VALUES (?,?,?,?,?)",
            (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), track_id, violation_type, detail, camera)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[DB Error] {e}")

    # In-memory cache
    system_logs.insert(0, {
        "time": timestamp, "id": track_id,
        "type": violation_type, "detail": detail, "camera": camera
    })
    if len(system_logs) > 50:
        system_logs.pop()

# ─── IoU Helper ──────────────────────────────────────────────────────────────
def get_iou(boxA, boxB):
    ix1 = max(boxA[0], boxB[0]); iy1 = max(boxA[1], boxB[1])
    ix2 = min(boxA[2], boxB[2]); iy2 = min(boxA[3], boxB[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0
    inter = (ix2 - ix1) * (iy2 - iy1)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    return inter / float(areaA) if areaA > 0 else 0

# ─── Frame Generator (FIX: bọc try/except CancelledError để không crash) ─────
async def generate_frames(cam_id: str, active_filters: list):
    if cam_id == "1":
        source = VIDEO_PATH
    elif cam_id == "2":
        source = 0
    else:
        source = os.path.join(TEMP_DIR, cam_id)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[CAM] Không mở được nguồn: {source}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or math.isnan(fps):
        fps = 30.0
    interval = 1.0 / fps

    show_helmet = 'helmet' in active_filters
    show_vest   = 'vest'   in active_filters
    show_sign   = 'sign'   in active_filters
    show_fall   = 'pose'   in active_filters

    try:
        while True:
            t0 = time.time()
            ret, frame = cap.read()

            # Video file kết thúc → quay vòng; Webcam mất → dừng
            if not ret:
                if cam_id == "2":
                    break
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # ── 1. PPE & TRACKING ────────────────────────────────────────────
            if show_helmet or show_vest:
                try:
                    res_p = model_ppe.track(frame, conf=0.35, persist=True, verbose=False)
                    persons, helmets, vests = [], [], []
                    if res_p[0].boxes is not None:
                        for b in res_p[0].boxes:
                            cls = model_ppe.names[int(b.cls[0])].lower()
                            box = b.xyxy[0].tolist()
                            if 'person' in cls:
                                tid = int(b.id[0]) if b.id is not None else 0
                                persons.append({'box': box, 'id': tid})
                            elif 'helmet' in cls and show_helmet:
                                helmets.append(box)
                                cv2.rectangle(frame,
                                    (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                                    (0, 255, 0), 1)
                            elif 'vest' in cls and show_vest:
                                vests.append(box)
                                cv2.rectangle(frame,
                                    (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                                    (255, 230, 0), 1)

                    for p in persons:
                        box_p, tid = p['box'], p['id']
                        has_h = any(get_iou(h, box_p) > 0.05 for h in helmets) if show_helmet else True
                        has_v = any(get_iou(v, box_p) > 0.10 for v in vests)   if show_vest   else True

                        missing = []
                        if show_helmet and not has_h: missing.append("Helmet")
                        if show_vest   and not has_v: missing.append("Vest")

                        color = (0, 0, 255) if missing else (0, 255, 0)
                        lw    = 3 if missing else 2
                        cv2.rectangle(frame,
                            (int(box_p[0]), int(box_p[1])), (int(box_p[2]), int(box_p[3])),
                            color, lw)

                        if missing:
                            label = f"!! NO {' & '.join(missing)} !!"
                            cv2.putText(frame, label,
                                (int(box_p[0]), int(box_p[1]) - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                            # ID nhỏ bên dưới label
                            cv2.putText(frame, f"id:{tid}",
                                (int(box_p[0]), int(box_p[1]) - 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

                            if time.time() - person_states[tid]["last_alert"] > 10:
                                add_log(tid, "PPE",
                                    f"ID:{tid} No {' & '.join(missing)}",
                                    camera=f"CAM {cam_id}")
                                person_states[tid]["last_alert"] = time.time()
                        else:
                            cv2.putText(frame, f"ID:{tid} AN TOAN",
                                (int(box_p[0]), int(box_p[1]) - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                except Exception as e:
                    print(f"[PPE Error] {e}")

            # ── 2. SIGN/CONE DETECTION ───────────────────────────────────────
            if show_sign:
                try:
                    res_s = model_cone(frame, conf=0.5, verbose=False)
                    if res_s[0].boxes is not None:
                        for b in res_s[0].boxes:
                            x1, y1, x2, y2 = map(int, b.xyxy[0])
                            cls_name = model_cone.names[int(b.cls[0])]
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 220), 2)
                            cv2.putText(frame, cls_name.upper(), (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 220), 2)
                except Exception as e:
                    print(f"[CONE Error] {e}")

            # ── 3. FALL DETECTION ────────────────────────────────────────────
            if show_fall:
                try:
                    res_f = model_fall(frame, conf=0.35, verbose=False)
                    if res_f[0].boxes is not None:
                        for b in res_f[0].boxes:
                            x1, y1, x2, y2 = map(int, b.xyxy[0])
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 80, 255), 3)
                            cv2.putText(frame, "!!! FALL !!!", (x1, y1 - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 80, 255), 2)
                            if time.time() - person_states[0]["last_alert"] > 5:
                                add_log(0, "FALL", "Phát hiện người ngã",
                                    camera=f"CAM {cam_id}")
                                person_states[0]["last_alert"] = time.time()
                except Exception as e:
                    print(f"[FALL Error] {e}")

            # ── Encode & Stream ──────────────────────────────────────────────
            ret_enc, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret_enc:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                       + buffer.tobytes() + b'\r\n')

            elapsed = time.time() - t0
            await asyncio.sleep(max(0.001, interval - elapsed))

    except asyncio.CancelledError:
        # Browser đóng kết nối → dừng generator sạch, KHÔNG crash server
        print(f"[CAM {cam_id}] Client disconnected — stream stopped cleanly.")
    except Exception as e:
        print(f"[CAM {cam_id}] Stream error: {e}")
    finally:
        cap.release()

# ─── API Endpoints ────────────────────────────────────────────────────────────
@app.get("/video_feed/{cam_id}")
async def video_feed(cam_id: str, filters: str = Query("helmet,vest,sign,pose")):
    active_filters = [f.strip() for f in filters.split(",")]
    return StreamingResponse(
        generate_frames(cam_id, active_filters),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.post("/api/upload_video")
async def upload_video(file: UploadFile = File(...)):
    try:
        file_path = os.path.join(TEMP_DIR, file.filename)
        with open(file_path, "wb") as buf:
            shutil.copyfileobj(file.file, buf)
        return {"filename": file.filename, "status": "success"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})

@app.get("/api/cameras")
async def get_cameras():
    return [
        {"id": "1", "name": "CAM 01 - Server (Video)", "status": "online"},
        {"id": "2", "name": "CAM 02 - Webcam Trực tiếp", "status": "available"}
    ]

@app.get("/api/logs")
async def get_logs():
    return {"logs": system_logs}

@app.get("/api/logs/history")
async def get_log_history(limit: int = 100):
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM logs ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})

# ─── Static Files ─────────────────────────────────────────────────────────────
current_dir = os.path.dirname(os.path.abspath(__file__))
app.mount("/", StaticFiles(directory=current_dir, html=True), name="static")

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=5001, reload=False, log_level="warning")
