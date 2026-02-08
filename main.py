import cv2
import time
import os
import threading
from ultralytics import YOLO
from playsound import playsound

# =====================================
# LOAD YOUR OLD MODEL (train3)
# =====================================
MODEL_PATH = r"C:\Users\MSI\Smart Animal Repellent System\runs\detect\train3\weights\best.pt"
model = YOLO(MODEL_PATH)

# =====================================
# SOUND FILES
# =====================================
BASE = r"C:\Users\MSI\Smart Animal Repellent System\sound"

sound_files = {
    1: os.path.join(BASE, "high_threat.mp3"),
    2: os.path.join(BASE, "moderate_threat.mp3"),
    3: os.path.join(BASE, "low_threat.mp3")
}

# =====================================
# THREAT GROUPS
# =====================================
LEVEL_1 = ['sheep','goat','monkey','elephant','deer','cattle','bull','bear']
LEVEL_2 = ['snake','pig','rabbit','chicken']
LEVEL_3 = ['sparrow','parrot']

def get_level(name):
    if name in LEVEL_1: return 1
    if name in LEVEL_2: return 2
    if name in LEVEL_3: return 3
    return None

# =====================================
# PLAY SOUND SAFELY
# =====================================
def play_sound(path):
    if os.path.exists(path):
        threading.Thread(target=playsound, args=(path,), daemon=True).start()

# =====================================
# VIDEO DETECTION FUNCTION
# =====================================
def detect_video(path):
    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        print("❌ Cannot open video")
        return

    last_play_time = 0
    last_name = None
    COOLDOWN = 5  # seconds

    print("\n▶ Starting Final Animal Detection... (Press Q to quit)\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        time.sleep(0.05)  # smooth video 20 FPS

        results = model(frame, verbose=False)[0]

        found = False

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            name = model.names[cls_id]

            if conf < 0.50:
                continue

            found = True
            level = get_level(name)

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

            label = f"{name} ({conf:.2f})"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            # SOUND LOGIC
            now = time.time()
            if level and (last_name != name or now - last_play_time >= COOLDOWN):
                print(f"\n🔊 DETECTED {name} | Level {level}")
                print("🎶 Playing sound for 3 seconds...")

                play_sound(sound_files[level])
                last_play_time = now
                last_name = name

                time.sleep(2)  # silence gap

            break

        cv2.imshow("SMART ANIMAL REPELLENT - FINAL VERSION", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# =====================================
# RUN
# =====================================
video_path = input("Enter video path: ")
detect_video(video_path)
