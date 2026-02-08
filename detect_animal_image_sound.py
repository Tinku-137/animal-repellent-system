from ultralytics import YOLO
from playsound import playsound
import os

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

# Path to test image
image_path = r"C:\Users\MSI\Smart Animal Repellent System\images\parrot.jpg"

# Run detection
results = model.predict(source=image_path, show=True, conf=0.25)

# Track which threat levels were triggered
played_levels = set()

for r in results:
    boxes = r.boxes
    if len(boxes) > 0:
        for box in boxes:
            cls = int(box.cls[0])
            class_name = model.names[cls].lower()
            print(f"Detected: {class_name}")

            # Check which sound to play
            if class_name in high_threat:
                threat = "high"
            elif class_name in moderate_threat:
                threat = "moderate"
            elif class_name in low_threat:
                threat = "low"
            else:
                threat = None

            # Avoid playing same sound multiple times in one image
            if threat and threat not in played_levels:
                sound_path = sound_files[threat]
                print(f"Playing {threat} threat sound: {sound_path}")
                playsound(sound_path)
                played_levels.add(threat)
    else:
        print("No animal detected.")
