
from insightface.app import FaceAnalysis
from pathlib import Path
import cv2
import csv
import math

# ---------- PATHS ----------
DATASET_DIR = Path(r"C:\Mini\Dataset")
OUTPUT_DIR = Path(r"C:\Mini\Processed_Dataset")
REPORT_PATH = Path(r"C:\Mini\face_crop_report.csv")

# ---------- FACE DETECTOR ----------
app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

app.prepare(ctx_id=-1, det_size=(640, 640))

# Image formats we support
valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

if not DATASET_DIR.is_dir():
    raise FileNotFoundError(
        f"Dataset folder not found: {DATASET_DIR}"
    )

# Find all images, including images in nested folders
image_paths = [
    path for path in DATASET_DIR.rglob("*")
    if path.is_file() and path.suffix.lower() in valid_extensions
]

if not image_paths:
    raise RuntimeError(
        f"No supported images found inside {DATASET_DIR}"
    )

print(f"Images found: {len(image_paths)}")
print("Starting automatic face cropping...\n")

report = []

for index, image_path in enumerate(image_paths, start=1):
    relative_path = image_path.relative_to(DATASET_DIR)
    image = cv2.imread(str(image_path))

    if image is None:
        report.append([
            str(relative_path), 0, "Unreadable image"
        ])
        print(f"[{index}/{len(image_paths)}] Unreadable")
        continue

    faces = app.get(image)

    # Only process images containing exactly one detected face
    if len(faces) == 0:
        report.append([
            str(relative_path), 0, "No face detected"
        ])
        print(f"[{index}/{len(image_paths)}] No face")
        continue

    if len(faces) > 1:
        report.append([
            str(relative_path), len(faces), "Multiple faces"
        ])
        print(
            f"[{index}/{len(image_paths)}] "
            f"Multiple faces: {len(faces)}"
        )
        continue

    # Get the detected face's bounding box
    x1, y1, x2, y2 = faces[0].bbox

    # Convert coordinates to integers
    x1, y1 = math.floor(x1), math.floor(y1)
    x2, y2 = math.ceil(x2), math.ceil(y2)

    height, width = image.shape[:2]

    # Add a 15% margin around the face
    margin_x = int((x2 - x1) * 0.15)
    margin_y = int((y2 - y1) * 0.15)

    x1 = max(0, x1 - margin_x)
    y1 = max(0, y1 - margin_y)
    x2 = min(width, x2 + margin_x)
    y2 = min(height, y2 + margin_y)

    face_crop = image[y1:y2, x1:x2]

    if face_crop.size == 0:
        report.append([
            str(relative_path), 1, "Invalid crop"
        ])
        continue

    # Preserve the original student/view folder structure
    output_path = OUTPUT_DIR / relative_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    saved = cv2.imwrite(str(output_path), face_crop)

    if saved:
        report.append([
            str(relative_path), 1, "Cropped successfully"
        ])
        print(f"[{index}/{len(image_paths)}] Cropped")
    else:
        report.append([
            str(relative_path), 1, "Save failed"
        ])

# ---------- SAVE REPORT ----------
with open(REPORT_PATH, "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Image", "Faces detected", "Status"])
    writer.writerows(report)

successful = sum(
    row[2] == "Cropped successfully" for row in report
)
flagged = len(report) - successful

print("\n---------- SUMMARY ----------")
print(f"Total images: {len(image_paths)}")
print(f"Successfully cropped: {successful}")
print(f"Images needing review: {flagged}")
print(f"Processed images: {OUTPUT_DIR}")
print(f"Report saved to: {REPORT_PATH}")
