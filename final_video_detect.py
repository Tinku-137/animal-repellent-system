import cv2
import time
import os
import threading
from ultralytics import YOLO
from playsound import playsound

# ==============================
# LOAD PREVIOUS WORKING MODEL
# ==============================
MODEL_PATH = r"C:\Users\MSI\Smart Animal Repellent System\runs\detect\train3\weights\best.pt"
model = YOLO(MODEL_PATH)

# ==============================
# SOUND FILES
# ==============================
BASE = r"C:\Users\MSI\Smart Animal Repellent System\sound"

sound_files = {
    1: os.path.join(BASE, "high_threat.mp3"),
    2: os.path.join(BASE, "moderate_threat.mp3"),
    3: os.path.join(BASE, "low_threat.mp3")
}

# ==============================
# THREAT GROUPS
# ==============================
LEVEL_1 = ['sheep','goat','monkey','elephant','deer','cattle','bull','bear']
LEVEL_2 = ['snake','pig','rabbit','chicken']
LEVEL_3 = ['sparrow','parrot']

def get_level(name):
    if name in LEVEL_1: return 1
    if name in LEVEL_2: return 2
    if name in LEVEL_3: return 3
    return None

# ==============================
# NON-BLOCKING SOUND PLAYER
# ==============================
def play_sound_async(path):
    if os.path.exists(path):
        threading.Thread(target=playsound, args=(path,), daemon=True).start()

# ==============================
# VIDEO DETECTION (OPTIMIZED)
# ==============================
def detect_video(path):
    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        print("❌ Cannot open video")
        return

    last_sound_time = 0
    last_animal = None

    SOUND_DURATION = 3   # seconds
    GAP_DURATION = 2     # seconds
    TOTAL_COOLDOWN = SOUND_DURATION + GAP_DURATION  # 5 sec

    print("\n▶ Fast & Smooth Detection Started (Press Q to quit)\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)[0]

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            name = model.names[cls_id]

            if conf < 0.45:
                continue

            level = get_level(name)
            if level is None:
                continue

            # ---- DRAW IMMEDIATELY ----
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

            label = f"{name.upper()}  {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            # ---- SOUND LOGIC (NO DELAY) ----
            now = time.time()

            if (name != last_animal) or (now - last_sound_time >= TOTAL_COOLDOWN):
                print(f"🔊 {name} detected → Level {level}")
                play_sound_async(sound_files[level])
                last_sound_time = now
                last_animal = name

            break  # handle only 1 animal per frame

        cv2.imshow("SMART ANIMAL REPELLENT - FAST MODE", frame)

        # Smooth playback (~25 FPS)
        if cv2.waitKey(40) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ==============================
# RUN
# ==============================
video_path = input("Enter video path: ")
detect_video(video_path)
