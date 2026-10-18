
from insightface.app import FaceAnalysis
from pathlib import Path
import numpy as np
import cv2

# ---------- PATHS ----------
DATABASE_PATH = Path(
    r"C:\Mini\Student_attendace_system\face_database\student_embeddings.npz"
)

# ---------- LOAD EMBEDDING DATABASE ----------
if not DATABASE_PATH.exists():
    raise FileNotFoundError(
        f"Embedding database not found: {DATABASE_PATH}"
    )

data = np.load(DATABASE_PATH, allow_pickle=False)

known_embeddings = data["embeddings"].astype(np.float32)
known_student_ids = data["student_ids"].astype(str)

if len(known_embeddings) == 0:
    raise RuntimeError("The embedding database is empty.")

print("Embedding database loaded.")
print("Stored embeddings:", len(known_embeddings))
print("Students:", sorted(set(known_student_ids)))

# ---------- LOAD RECOGNITION MODEL ----------
app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)
app.prepare(ctx_id=-1, det_size=(640, 640))

# ---------- SELECT TEST IMAGE ----------
print("\nEnter the FULL path to a test image.")
print("Prefer an original image not used to generate the database.")

image_path = Path(input("Image path: ").strip().strip('"'))

if not image_path.is_file():
    raise FileNotFoundError(f"Image not found: {image_path}")

image = cv2.imread(str(image_path))

if image is None:
    raise RuntimeError("OpenCV could not read this image.")

# ---------- DETECT FACE AND EXTRACT EMBEDDING ----------
faces = app.get(image)

if len(faces) == 0:
    print("No face detected in the test image.")
    raise SystemExit

if len(faces) > 1:
    print(f"Found {len(faces)} faces.")
    print("Use a test image containing only one student.")
    raise SystemExit

test_embedding = faces[0].normed_embedding.astype(np.float32)

# ---------- COMPARE EMBEDDINGS ----------
# Both stored and test embeddings are normalized.
# Their dot product gives cosine similarity.
similarities = known_embeddings @ test_embedding

# Calculate each student's mean of their top 3 image similarities.
# This reduces dependence on just one reference image.
student_scores = {}

for student_id in sorted(set(known_student_ids)):
    student_similarities = similarities[
        known_student_ids == student_id
    ]

    top_k = min(3, len(student_similarities))
    top_scores = np.sort(student_similarities)[-top_k:]

    student_scores[student_id] = float(np.mean(top_scores))

ranked = sorted(
    student_scores.items(),
    key=lambda item: item[1],
    reverse=True
)

# ---------- DISPLAY RESULTS ----------
print("\n---------- RECOGNITION RESULTS ----------")

for rank, (student_id, score) in enumerate(ranked, start=1):
    print(
        f"{rank}. Student: {student_id} | "
        f"Similarity: {score:.4f}"
    )

best_student, best_score = ranked[0]

print("\nBest candidate:", best_student)
print(f"Similarity score: {best_score:.4f}")

if len(ranked) > 1:
    margin = ranked[0][1] - ranked[1][1]
    print(f"Margin over second candidate: {margin:.4f}")

print(
    "\nNote: This is a preliminary match, not a confirmed identity. "
    "Scores alone do not establish a reliable acceptance threshold."
)
