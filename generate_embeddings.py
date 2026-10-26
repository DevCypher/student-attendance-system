
from insightface.app import FaceAnalysis
from pathlib import Path
import numpy as np
import cv2
import csv
import json

# ==================================================
# 1. CONFIGURATION
# ==================================================

DATASET_DIR = Path(r"C:\Mini\Processed_Dataset2")

OUTPUT_DIR = Path(
    r"C:\Mini\Student_attendace_system\face_database"
)

REPORT_PATH = OUTPUT_DIR / "generation_report.json"
FAILURE_CSV = OUTPUT_DIR / "failed_images.csv"

VALID_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# 2. CHECK DATASET
# ==================================================

if not DATASET_DIR.is_dir():
    raise FileNotFoundError(
        f"Processed dataset not found: {DATASET_DIR}"
    )

student_folders = sorted(
    folder for folder in DATASET_DIR.iterdir()
    if folder.is_dir()
)

if not student_folders:
    raise RuntimeError(
        "No student folders found in the processed dataset."
    )

image_paths = [
    path
    for path in DATASET_DIR.rglob("*")
    if path.is_file()
    and path.suffix.lower() in VALID_EXTENSIONS
]

if not image_paths:
    raise RuntimeError("No supported images found.")


# ==================================================
# 3. LOAD INSIGHTFACE MODEL
# ==================================================

print("Loading InsightFace model...")

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

app.prepare(
    ctx_id=-1,
    det_size=(640, 640)
)

print("Model loaded successfully.")
print(f"Students found: {len(student_folders)}")
print(f"Images found: {len(image_paths)}")
print("\nStarting embedding generation...\n")


# ==================================================
# 4. PROCESS IMAGES
# ==================================================

embeddings = []
student_ids = []
image_paths_saved = []
failed_images = []

for index, image_path in enumerate(image_paths, start=1):

    relative_path = image_path.relative_to(DATASET_DIR)
    student_id = relative_path.parts[0]

    print(
        f"[{index}/{len(image_paths)}] "
        f"Student: {student_id} | "
        f"Image: {image_path.name}"
    )

    # Read the image
    image = cv2.imread(str(image_path))

    if image is None:
        failed_images.append({
            "path": str(relative_path),
            "reason": "Unreadable image"
        })
        print("    FAILED: Unreadable image")
        continue

    # Detect faces in the processed crop
    try:
        faces = app.get(image)
    except Exception as error:
        failed_images.append({
            "path": str(relative_path),
            "reason": f"Detection error: {type(error).__name__}: {error}"
        })
        print("    FAILED: Detection error")
        continue

    # No face found
    if len(faces) == 0:
        failed_images.append({
            "path": str(relative_path),
            "reason": "No face detected in processed crop"
        })
        print("    FAILED: No face detected")
        continue

    # More than one face found
    if len(faces) > 1:
        failed_images.append({
            "path": str(relative_path),
            "reason": f"Multiple faces detected: {len(faces)}"
        })
        print(f"    FAILED: {len(faces)} faces detected")
        continue

    # Extract normalized face embedding
    embedding = faces[0].normed_embedding

    if embedding is None or not np.all(np.isfinite(embedding)):
        failed_images.append({
            "path": str(relative_path),
            "reason": "Invalid embedding"
        })
        print("    FAILED: Invalid embedding")
        continue

    # Save embedding and its student identity
    embeddings.append(
        np.asarray(embedding, dtype=np.float32)
    )

    student_ids.append(student_id)
    image_paths_saved.append(str(relative_path))

    print("    SUCCESS: Embedding generated")


# ==================================================
# 5. SAVE EMBEDDING DATABASE
# ==================================================

if embeddings:

    database_path = OUTPUT_DIR / "student_embeddings.npz"

    np.savez_compressed(
        database_path,
        embeddings=np.stack(embeddings),
        student_ids=np.asarray(student_ids),
        image_paths=np.asarray(image_paths_saved)
    )

    print(f"\nDatabase saved: {database_path}")

else:
    print("\nNo embeddings were generated.")

    # Avoid leaving a previous database looking current.
    database_path = OUTPUT_DIR / "student_embeddings.npz"

    if database_path.exists():
        database_path.unlink()

    print("No embedding database was retained.")


# ==================================================
# 6. SAVE DETAILED JSON REPORT
# ==================================================

report = {
    "dataset_directory": str(DATASET_DIR),
    "total_students_found": len(student_folders),
    "total_images_found": len(image_paths),
    "successful_embeddings": len(embeddings),
    "students_with_embeddings": sorted(set(student_ids)),
    "failed_images_count": len(failed_images),
    "failed_images": failed_images
}

with open(REPORT_PATH, "w", encoding="utf-8") as file:
    json.dump(report, file, indent=2)

# ==================================================
# 7. SAVE FAILURE CSV
# ==================================================

with open(
    FAILURE_CSV,
    "w",
    newline="",
    encoding="utf-8-sig"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=["path", "reason"]
    )

    writer.writeheader()
    writer.writerows(failed_images)


# ==================================================
# 8. FINAL SUMMARY
# ==================================================

print("\n" + "=" * 45)
print("       EMBEDDING GENERATION SUMMARY")
print("=" * 45)

print(f"Students found:             {len(student_folders)}")
print(f"Total images:               {len(image_paths)}")
print(f"Embeddings generated:       {len(embeddings)}")
print(f"Images needing review:      {len(failed_images)}")
print(f"Students with embeddings:   {len(set(student_ids))}")

print("\nFailure breakdown:")

failure_counts = {}

for item in failed_images:
    reason = item["reason"]

    # Group detailed errors by their general category
    if reason.startswith("Detection error:"):
        category = "Detection errors"
    elif reason.startswith("Multiple faces detected:"):
        category = "Multiple faces detected"
    else:
        category = reason

    failure_counts[category] = (
        failure_counts.get(category, 0) + 1
    )

if failure_counts:
    for reason, count in sorted(failure_counts.items()):
        print(f"  {reason}: {count}")
else:
    print("  No failures!")

print(f"\nJSON report: {REPORT_PATH}")
print(f"Failure CSV: {FAILURE_CSV}")

print("\nProcessing completed.")
