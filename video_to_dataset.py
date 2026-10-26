
from insightface.app import FaceAnalysis
from pathlib import Path
import cv2
import numpy as np
import json
import re

# ==================================================
# CONFIGURATION
# ==================================================

OUTPUT_ROOT = Path(r"C:\Mini\dataset2")

# Sample every 2nd video frame to reduce processing time
FRAME_STEP = 2

# Target images for each view
VIEW_TARGETS = {
    "front": 40,
    "left": 30,
    "right": 30
}

# Minimum acceptable face crop dimensions
MIN_FACE_SIZE = 80

# Laplacian variance is used as a basic blur-quality measure
MIN_SHARPNESS = 40.0

# Similarity above this value means two embeddings are
# potentially too similar to keep both.
DUPLICATE_SIMILARITY = 0.995

VALID_VIDEO_EXTENSIONS = {
    ".mp4", ".avi", ".mov", ".mkv", ".webm"
}

# ==================================================
# 1. GET INPUTS
# ==================================================

video_path = Path(
    input("Enter full path of the student video: ")
    .strip().strip('"')
)

student_id = input("Enter student USN: ").strip()

if not re.fullmatch(r"[A-Za-z0-9_-]+", student_id):
    raise ValueError(
        "Invalid USN. Use only letters, numbers, hyphens or underscores."
    )

if not video_path.is_file():
    raise FileNotFoundError(f"Video not found: {video_path}")

if video_path.suffix.lower() not in VALID_VIDEO_EXTENSIONS:
    raise ValueError("Unsupported video format.")

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

student_output = OUTPUT_ROOT / student_id

# Never silently mix a new recording with existing student images
if student_output.exists():
    raise FileExistsError(
        f"The folder already exists: {student_output}\n"
        "Back up the existing data or choose a different USN folder "
        "before processing this video."
    )

# ==================================================
# 2. INITIALIZE INSIGHTFACE
# ==================================================

print("\nLoading InsightFace model...")

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

app.prepare(
    ctx_id=-1,
    det_size=(640, 640)
)

print("Model loaded.")

# ==================================================
# 3. OPEN VIDEO
# ==================================================

cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    raise RuntimeError("Could not open the video.")

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if fps <= 0 or total_frames <= 0:
    cap.release()
    raise RuntimeError("Could not determine video duration or FPS.")

duration = total_frames / fps

print(f"Video duration: {duration:.2f} seconds")
print(f"Video FPS: {fps:.2f}")
print(f"Total video frames: {total_frames}")

if duration < 8:
    print(
        "WARNING: This video is shorter than 8 seconds. "
        "The selected view periods may not contain enough samples."
    )

# The recording must follow this order:
# first 40% = front, next 30% = left, final 30% = right.
front_end = duration * 0.40
left_end = duration * 0.70

# ==================================================
# 4. READ FRAMES AND DETECT FACES
# ==================================================

candidates = {
    "front": [],
    "left": [],
    "right": []
}

failure_counts = {
    "no_face": 0,
    "multiple_faces": 0,
    "blurry": 0,
    "face_too_small": 0,
    "invalid_crop": 0,
    "duplicate": 0
}

frame_index = 0
sampled_frames = 0

print("\nScanning video and extracting face candidates...")

try:
    while True:
        success, frame = cap.read()

        if not success:
            break

        current_index = frame_index
        frame_index += 1

        if current_index % FRAME_STEP != 0:
            continue

        timestamp = current_index / fps

        # Assign a view label according to the recording schedule.
        if timestamp < front_end:
            view = "front"
        elif timestamp < left_end:
            view = "left"
        else:
            view = "right"

        sampled_frames += 1

        faces = app.get(frame)

        if len(faces) == 0:
            failure_counts["no_face"] += 1
            continue

        if len(faces) > 1:
            failure_counts["multiple_faces"] += 1
            continue

        face = faces[0]

        x1, y1, x2, y2 = face.bbox.astype(int)

        height, width = frame.shape[:2]

        face_width = x2 - x1
        face_height = y2 - y1

        if face_width < MIN_FACE_SIZE or face_height < MIN_FACE_SIZE:
            failure_counts["face_too_small"] += 1
            continue

        # Add a margin around the face
        margin_x = int(face_width * 0.15)
        margin_y = int(face_height * 0.15)

        x1 = max(0, x1 - margin_x)
        y1 = max(0, y1 - margin_y)
        x2 = min(width, x2 + margin_x)
        y2 = min(height, y2 + margin_y)

        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            failure_counts["invalid_crop"] += 1
            continue

        # Basic sharpness check
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        sharpness = float(
            cv2.Laplacian(gray, cv2.CV_64F).var()
        )

        if sharpness < MIN_SHARPNESS:
            failure_counts["blurry"] += 1
            continue

        embedding = face.normed_embedding

        if embedding is None or not np.all(np.isfinite(embedding)):
            failure_counts["invalid_crop"] += 1
            continue

        candidates[view].append({
            "crop": crop.copy(),
            "embedding": np.asarray(
                embedding, dtype=np.float32
            ),
            "sharpness": sharpness,
            "timestamp": timestamp
        })

        if sampled_frames % 30 == 0:
            print(
                f"Sampled {sampled_frames} frames | "
                f"Front: {len(candidates['front'])}, "
                f"Left: {len(candidates['left'])}, "
                f"Right: {len(candidates['right'])}"
            )

finally:
    cap.release()

# ==================================================
# 5. SELECT SHARP, DISTINCT IMAGES
# ==================================================

def select_distinct_images(items, target):
    # Prefer sharp images first.
    items = sorted(
        items,
        key=lambda item: item["sharpness"],
        reverse=True
    )

    selected = []

    for item in items:
        if len(selected) >= target:
            break

        embedding = item["embedding"]

        # Reject near-identical face representations.
        is_duplicate = False

        for existing in selected:
            similarity = float(
                np.dot(embedding, existing["embedding"])
            )

            if similarity >= DUPLICATE_SIMILARITY:
                is_duplicate = True
                break

        if is_duplicate:
            failure_counts["duplicate"] += 1
            continue

        selected.append(item)

    # Save images in chronological order.
    selected.sort(key=lambda item: item["timestamp"])

    return selected

selected_images = {}

for view, target in VIEW_TARGETS.items():
    selected_images[view] = select_distinct_images(
        candidates[view],
        target
    )

# ==================================================
# 6. SAVE SELECTED IMAGES
# ==================================================

student_output.mkdir(parents=True, exist_ok=False)

saved_counts = {}

for view, items in selected_images.items():
    saved_counts[view] = 0

    for number, item in enumerate(items, start=1):
        filename = f"{view}_{number:03d}.jpg"
        output_path = student_output / filename

        if cv2.imwrite(str(output_path), item["crop"]):
            saved_counts[view] += 1
        else:
            print(f"WARNING: Could not save {output_path}")

# ==================================================
# 7. SAVE PROCESSING REPORT
# ==================================================

report = {
    "student_id": student_id,
    "source_video": str(video_path),
    "duration_seconds": round(duration, 2),
    "video_fps": round(fps, 2),
    "sampled_frames": sampled_frames,
    "target_images": sum(VIEW_TARGETS.values()),
    "saved_images": sum(saved_counts.values()),
    "view_targets": VIEW_TARGETS,
    "saved_per_view": saved_counts,
    "candidate_frames_per_view": {
        view: len(items)
        for view, items in candidates.items()
    },
    "rejected_frames": failure_counts
}

report_path = student_output / "processing_report.json"

with open(report_path, "w", encoding="utf-8") as file:
    json.dump(report, file, indent=2)

# ==================================================
# 8. DISPLAY SUMMARY
# ==================================================

print("\n" + "=" * 45)
print("       VIDEO DATASET SUMMARY")
print("=" * 45)

print("Student USN:", student_id)

for view, target in VIEW_TARGETS.items():
    print(
        f"{view.capitalize():<8}: "
        f"{saved_counts[view]:>3}/{target} images"
    )

total_saved = sum(saved_counts.values())

print("-" * 45)
print(f"Total saved: {total_saved}/100 images")
print("Output folder:", student_output)
print("Report:", report_path)

if total_saved < 100:
    print(
        "\nNOTE: Fewer than 100 images passed the filters. "
        "Review the report and recording before collecting more data."
    )

print("\nRejected frame counts:")

for reason, count in failure_counts.items():
    print(f"  {reason}: {count}")

print("\nProcessing completed.")
