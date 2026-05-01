from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import csv, os, uvicorn
from datetime import date  # <-- Add this new import

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(BASE_DIR, "data", "daily_attendance.csv")
FACES_DIR = os.path.join(BASE_DIR, "data", "registered_faces")

@app.get("/")
def serve_ui():
    return FileResponse(os.path.join(BASE_DIR, "frontend", "index.html"))

@app.get("/api/data")
def get_data():
    reg_users = set()
    if os.path.exists(FACES_DIR):
        reg_users = {os.path.splitext(f)[0] for f in os.listdir(FACES_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))}

    records, present_users = [], set()
    if os.path.exists(CSV_FILE):
        with open(CSV_FILE, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                records.append(row)
                present_users.add(row["Name"])
    
    absent_users = list(reg_users - present_users)
    
    return {
        "present_logs": records[::-1],
        "absent": absent_users,
        "stats": {"total": len(reg_users), "present": len(present_users), "absent": len(absent_users)}
    }

@app.get("/api/download")
def download_report():
    """Forces the browser to download the CSV as an Excel-compatible file."""
    if os.path.exists(CSV_FILE):
        today = date.today().strftime("%Y-%m-%d")
        return FileResponse(
            path=CSV_FILE, 
            filename=f"Attendance_Report_{today}.csv", 
            media_type="text/csv"
        )
    return {"error": "No attendance data found for today."}

@app.get("/")
def serve_user_view():
    # This will be the simple screen for students
    return FileResponse(os.path.join(BASE_DIR, "frontend", "user.html"))

@app.get("/admin")
def serve_admin_view():
    # This remains your professional dashboard for professors
    return FileResponse(os.path.join(BASE_DIR, "frontend", "index.html"))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)