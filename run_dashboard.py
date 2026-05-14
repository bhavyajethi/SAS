from fastapi import FastAPI, Query, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import csv, os, json, uvicorn, subprocess, base64, sys
import cv2, numpy as np, face_recognition, pickle
from datetime import date, datetime

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

B_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_F = os.path.join(B_DIR, "data", "daily_attendance.csv")
F_DIR = os.path.join(B_DIR, "data", "registered_faces")
ENC_F = os.path.join(B_DIR, "data", "encodings.pkl")

def safe_serve(filepath):
    """Prevents 500 errors by checking if the HTML file actually exists"""
    if not os.path.exists(filepath):
        return {"error": f"FILE MISSING: The file {filepath} does not exist."}
    return FileResponse(filepath)

@app.get("/")
def home(): 
    return safe_serve(os.path.join(B_DIR, "frontend", "home.html"))

@app.get("/student")
def serve_student_kiosk(): 
    return safe_serve(os.path.join(B_DIR, "frontend", "user.html"))

@app.get("/register")
def reg_ui(): 
    return safe_serve(os.path.join(B_DIR, "frontend", "register.html"))

@app.get("/admin")
def adm_ui(): 
    return safe_serve(os.path.join(B_DIR, "frontend", "index.html"))

@app.get("/api/start_camera")
def start_cam():
    subprocess.Popen([sys.executable, "run_attendance.py"], cwd=B_DIR)
    return {"status": "ok"}

@app.post("/api/register_live")
async def reg_live(name: str = Form(...), image: str = Form(...)):
    img_b64 = image.split(",")[1]
    arr = np.frombuffer(base64.b64decode(img_b64), np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    boxes = face_recognition.face_locations(rgb, model="hog")
    if not boxes: return {"msg": "No face detected. Adjust lighting.", "success": False}
    
    new_enc = face_recognition.face_encodings(rgb, boxes)[0]
    
    if os.path.exists(ENC_F):
        with open(ENC_F, "rb") as f:
            data = pickle.load(f)
        if len(data["encodings"]) > 0:
            dists = face_recognition.face_distance(data["encodings"], new_enc)
            best_idx = int(dists.argmin())
            if dists[best_idx] < 0.5:
                match = data["names"][best_idx]
                return {"msg": f"Face already registered as {match}.", "success": False}
                
    os.makedirs(F_DIR, exist_ok=True)
    safe_n = name.replace(" ", "_")
    cv2.imwrite(os.path.join(F_DIR, f"{safe_n}.jpg"), frame)
    subprocess.run([sys.executable, "encode_faces.py"], cwd=B_DIR)
    return {"msg": f"Success! {safe_n} registered.", "success": True}

@app.get("/api/data")
def get_data(subject: str = Query("All")):
    reg_u = {os.path.splitext(f)[0] for f in os.listdir(F_DIR) if f.endswith(('.jpg','.jpeg','.png'))} if os.path.exists(F_DIR) else set()
    recs, pres = [], set()
    if os.path.exists(CSV_F):
        with open(CSV_F, "r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if subject != "All" and r.get("Subject", "") != subject: continue
                recs.append(r)
                pres.add(r["Name"])
    absent = list(reg_u - pres)
    return {"present_logs": recs[::-1], "absent": absent, "stats": {"total": len(reg_u), "present": len(pres), "absent": len(absent)}}

@app.get("/api/download")
def dl_rep():
    if os.path.exists(CSV_F):
        return FileResponse(path=CSV_F, filename=f"Attendance_{date.today()}.csv", media_type="text/csv")
    return {"error": "No data"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)