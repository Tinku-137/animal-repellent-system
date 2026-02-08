import os
import cv2

# Paths to your label and image folders
folders = [
    r"C:\Users\MSI\Smart Animal Repellent System\train",
    r"C:\Users\MSI\Smart Animal Repellent System\test"
]

def normalize_label(label_path, image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Missing image: {image_path}")
            return
        h, w = img.shape[:2]

        with open(label_path, "r") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            cls_id = parts[0]
            coords = list(map(float, parts[1:]))

            # If coordinates are > 1, assume they’re in pixels → normalize
            if any(v > 1 for v in coords):
                x_c, y_c, bw, bh = coords
                x_c /= w
                y_c /= h
                bw /= w
                bh /= h
                new_lines.append(f"{cls_id} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}")
            else:
                new_lines.append(line.strip())

        with open(label_path, "w") as f:
            f.write("\n".join(new_lines))

    except Exception as e:
        print(f"⚠️ Error in {label_path}: {e}")

for folder in folders:
    label_dir = os.path.join(folder, "labels")
    image_dir = os.path.join(folder, "images")

    for file in os.listdir(label_dir):
        if file.endswith(".txt"):
            label_path = os.path.join(label_dir, file)
            image_path = os.path.join(image_dir, file.replace(".txt", ".jpg"))
            normalize_label(label_path, image_path)

print("✅ All label files have been normalized successfully!")
