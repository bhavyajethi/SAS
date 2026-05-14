import cv2, face_recognition, mediapipe as mp, pickle, time, os, csv, math, json
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENCODINGS_FILE = os.path.join(BASE_DIR, "data", "encodings.pkl")
ATTENDANCE_FILE = os.path.join(BASE_DIR, "data", "daily_attendance.csv")
TIMETABLE_FILE = os.path.join(BASE_DIR, "data", "timetable.json")

EAR_THRESH = 0.22
COOLDOWN_SEC = 3600

print("[INFO] Loading DB...")
with open(ENCODINGS_FILE, "rb") as f:
    data = pickle.load(f)
encodings = data["encodings"]
names = data["names"]

mp_fm = mp.solutions.face_mesh
mesh = mp_fm.FaceMesh(max_num_faces=3, refine_landmarks=True)

def load_tt():
    if not os.path.exists(TIMETABLE_FILE): return {}
    with open(TIMETABLE_FILE, "r") as f: return json.load(f)

tt_data = load_tt()

def get_cls(day, tm, tt):
    for s in tt.get(day, []):
        if s["start"] <= tm <= s["end"]: return s["subject"]
    return None

def load_st():
    st = {}
    if os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, 'r') as f:
            rdr = csv.reader(f)
            next(rdr, None)
            for r in rdr:
                if len(r) >= 4:
                    n, dt_s, tm_s, sub = r[0], r[1], r[2], r[3]
                    try:
                        dt = datetime.strptime(f"{dt_s} {tm_s}", "%Y-%m-%d %H:%M:%S")
                        st[f"{n}_{sub}"] = dt.timestamp()
                    except ValueError: pass
    return st

marked = load_st()

def get_ear(lm):
    v1 = math.hypot(lm[1].x - lm[5].x, lm[1].y - lm[5].y)
    v2 = math.hypot(lm[2].x - lm[4].x, lm[2].y - lm[4].y)
    h = math.hypot(lm[0].x - lm[3].x, lm[0].y - lm[3].y)
    return (v1 + v2) / (2.0 * h)

def log_att(n, ac):
    now_ts = time.time()
    k = f"{n}_{ac}"
    if k in marked and (now_ts - marked[k]) < COOLDOWN_SEC: return False
    
    fe = os.path.isfile(ATTENDANCE_FILE)
    ie = not fe or os.path.getsize(ATTENDANCE_FILE) == 0
    now = datetime.now()
    
    with open(ATTENDANCE_FILE, 'a', newline='') as f:
        w = csv.writer(f)
        if ie: w.writerow(["Name", "Date", "Time", "Subject", "Status", "Liveness"])
        w.writerow([n, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"), ac, "Present", "Verified"])
    
    marked[k] = now_ts
    return True

print("[INFO] Starting Stream...")
cap = cv2.VideoCapture(0)
pf = True
bxs, c_nms, c_cfs = [], [], []

while True:
    ret, frm = cap.read()
    if not ret: break

    now = datetime.now()
    c_day = now.strftime("%A")
    c_tm = now.strftime("%H:%M")
    
    ac = get_cls(c_day, c_tm, tt_data)

    cv2.rectangle(frm, (5, 5), (350, 85), (0, 0, 0), -1)
    cv2.putText(frm, f"Sys Time: {c_day} {c_tm}", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    if not ac:
        cv2.putText(frm, "STANDBY: No Active Class", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frm, "Check Sys Time matches Timetable", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.imshow('Liveness-Guard', frm)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        continue 

    frm = cv2.convertScaleAbs(frm, alpha=2.5, beta=80)
    s_frm = cv2.resize(frm, (0, 0), fx=0.25, fy=0.25)
    rgb_s = cv2.cvtColor(s_frm, cv2.COLOR_BGR2RGB)
    rgb_f = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
    
    res = mesh.process(rgb_f)
    blk = False
    
    if res.multi_face_landmarks:
        for fl in res.multi_face_landmarks:
            l_e = [fl.landmark[i] for i in [33, 160, 158, 133, 153, 144]]
            r_e = [fl.landmark[i] for i in [362, 385, 387, 263, 373, 380]]
            a_ear = (get_ear(l_e) + get_ear(r_e)) / 2.0
            if a_ear < EAR_THRESH:
                blk = True
                cv2.putText(frm, "BLINK DETECTED", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if pf:
        bxs = face_recognition.face_locations(rgb_s, model="hog")
        encs = face_recognition.face_encodings(rgb_s, bxs)
        c_nms, c_cfs = [], []
        
        for enc in encs:
            n, cf = "Unknown", 0.0
            dsts = face_recognition.face_distance(encodings, enc)
            if len(dsts) > 0:
                b_idx = int(dsts.argmin())
                if dsts[b_idx] < 0.5: 
                    n = names[b_idx]
                    cf = (1 - dsts[b_idx]) * 100
            c_nms.append(n)
            c_cfs.append(cf)
            
    pf = not pf

    for (t, r, b, l), n, cf in zip(bxs, c_nms, c_cfs):
        t *= 4; r *= 4; b *= 4; l *= 4
        clr = (0, 0, 255)
        
        if n == "Unknown":
            lbl = "Unknown Face"
        else:
            lbl = f"{n} - Checking..."
            k = f"{n}_{ac}"

            if blk:
                if log_att(n, ac):
                    clr, lbl = (0, 255, 0), f"{n} - VERIFIED"
                else:
                    clr, lbl = (255, 255, 0), f"{n} - ALREADY MARKED"
            elif k in marked and (time.time() - marked[k]) < COOLDOWN_SEC:
                clr, lbl = (255, 255, 0), f"{n} - ALREADY MARKED"
                
        cv2.rectangle(frm, (l, t), (r, b), clr, 2)
        cv2.putText(frm, lbl, (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, clr, 2)

    cv2.putText(frm, f"Active: {ac}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frm, f"System Active | HW: i3 CPU", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    cv2.imshow('Liveness-Guard', frm)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()