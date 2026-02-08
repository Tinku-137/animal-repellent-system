from ultralytics import YOLO
from playsound import playsound
import os
import cv2
import time

# Load the trained model
model = YOLO(r"C:\Users\MSI\Smart Animal Repellent System\runs\animal_train6\weights\best.pt")

# Define animal threat levels
high_threat = ['sheep', 'goat', 'monkey', 'elephant', 'deer', 'cattle', 'bull', 'bear']
moderate_threat = ['snake', 'pig', 'rabbit', 'chicken']
low_threat = ['sparrow', 'parrot']

# Sound paths
sound_folder = r"C:\Users\MSI\Smart Animal Repellent System\sound"
sound_files = {
    "high": os.path.join(sound_folder, "high_threat.mp3"),
    "moderate": os.path.join(sound_folder, "moderate_threat.mp3"),
    "low": os.path.join(sound_folder, "low_threat.mp3")
}

# Input source: you can give a video file or 0 for webcam
source = r"C:\Users\MSI\Smart Animal Repellent System\video\multi_animal_test.mp4"  # Or use 0 for webcam

# Open video
cap = cv2.VideoCapture(source)

# To avoid playing same sound continuously
last_played = {"high": 0, "moderate": 0, "low": 0}
cooldown = 5  # seconds between replays of same threat sound

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model.predict(frame, conf=0.25, verbose=False)
    current_threats = set()

    for r in results:
        boxes = r.boxes
        if len(boxes) > 0:
            for box in boxes:
                cls = int(box.cls[0])
                class_name = model.names[cls].lower()

                # Determine threat level
                if class_name in high_threat:
                    threat = "high"
                elif class_name in moderate_threat:
                    threat = "moderate"
                elif class_name in low_threat:
                    threat = "low"
                else:
                    threat = None

                if threat:
                    current_threats.add(threat)

    # Play sounds (one per level detected)
    for level in current_threats:
        if time.time() - last_played[level] > cooldown:
            print(f"⚠️ {level.upper()} threat detected! Playing sound...")
            playsound(sound_files[level])
            last_played[level] = time.time()

    # Display live detections
    annotated_frame = results[0].plot()
    cv2.imshow("Animal Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
