import cv2  # type: ignore
import os
import numpy as np
from facenet_pytorch import MTCNN  # type: ignore

# Initialize face detector
aa = MTCNN()

# INPUT_DIR = r"D:\Security System\training"
# OUTPUT_DIR = r"D:\Security System\cropped_faces"


INPUT_DIR = r"D:\Security System\captured_faces"
OUTPUT_DIR = r"D:\Security System\cropped_captured"

os.makedirs(OUTPUT_DIR, exist_ok=True)
def augment_face(face, img_name, save_dir):
    face = cv2.resize(face, (160, 160))  # FIXED SIZE
    base = os.path.splitext(img_name)[0]

    def save(img, name):
        path = os.path.join(save_dir, name)
        success = cv2.imwrite(path, img)
        if not success:
            print("Failed to save:", path)

    # 1. Original
    save(face, f"{base}_orig.jpg")

    # 2. Flip
    flip = cv2.flip(face, 1)
    save(flip, f"{base}_flip.jpg")

    # 3. Rotate
    for angle in [10, -10]:
        M = cv2.getRotationMatrix2D((80, 80), angle, 1.0)
        rotated = cv2.warpAffine(face, M, (160, 160))
        save(rotated, f"{base}_rot{angle}.jpg")

    # 4. Brightness up
    bright = cv2.convertScaleAbs(face, alpha=1.3, beta=40)
    save(bright, f"{base}_bright.jpg")

    # 5. Brightness down
    dark = cv2.convertScaleAbs(face, alpha=0.7, beta=-40)
    save(dark, f"{base}_dark.jpg")

    # 6. Blur
    blur = cv2.GaussianBlur(face, (5, 5), 0)
    save(blur, f"{base}_blur.jpg")


# ---------- MAIN LOOP ----------
for class_name in os.listdir(INPUT_DIR):
    class_path = os.path.join(INPUT_DIR, class_name)
    if not os.path.isdir(class_path):
        continue

    output_class_path = os.path.join(OUTPUT_DIR, class_name)
    os.makedirs(output_class_path, exist_ok=True)

    for img_name in os.listdir(class_path):
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        img_path = os.path.join(class_path, img_name)
        image = cv2.imread(img_path)

        if image is None:
            print(f"Could not read {img_name}")
            continue

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes, _ = aa.detect(rgb)

        if boxes is None:
            print(f"No face detected in {img_name}")
            continue

        x1, y1, x2, y2 = map(int, boxes[0])
        h, w, _ = image.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        face_crop = image[y1:y2, x1:x2]

        if face_crop is not None and face_crop.size > 0:
            augment_face(face_crop, img_name, output_class_path)
            print(f"Augmented faces saved for {img_name}")
        else:
            print(f"Invalid crop for {img_name}")
