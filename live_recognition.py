
from insightface.app import FaceAnalysis
from pathlib import Path
import numpy as np
import cv2

# ---------- DATABASE ----------
DATABASE_PATH = Path(
    r"C:\Mini\Student_attendace_system\face_database\student_embeddings.npz"
)

if not DATABASE_PATH.exists():
    raise FileNotFoundError(
        f"Embedding database not found: {DATABASE_PATH}"
    )

data = np.load(DATABASE_PATH, allow_pickle=False)

known_embeddings = data["embeddings"].astype(np.float32)
known_student_ids = data["student_ids"].astype(str)

if len(known_embeddings) == 0:
    raise RuntimeError("The embedding database is empty.")

# ---------- FACE MODEL ----------
app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)
app.prepare(ctx_id=-1, det_size=(640, 640))

# ---------- RECOGNITION SETTINGS ----------
# Initial experimental threshold, NOT a validated attendance threshold.
# Adjust only after testing known and unknown people.
SIMILARITY_THRESHOLD = 0.40

# ---------- WEBCAM ----------
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Could not open webcam.")

print("Live recognition started. Press Q to quit.")
print("Testing with", len(known_embeddings), "embeddings.")

try:
    while True:
        success, frame = camera.read()

        if not success:
            print("Could not read webcam frame.")
            break

        faces = app.get(frame)

        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int)

            # Keep bounding box within the frame
            height, width = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)

            query = face.normed_embedding.astype(np.float32)

            # Cosine similarities; embeddings are normalized
            scores = known_embeddings @ query

            # Find each student's best matching reference image
            student_scores = {}

            for student_id in np.unique(known_student_ids):
                scores_for_student = scores[
                    known_student_ids == student_id
                ]
                student_scores[student_id] = float(
                    np.max(scores_for_student)
                )

            best_student = max(
                student_scores,
                key=student_scores.get
            )
            best_score = student_scores[best_student]

            if best_score >= SIMILARITY_THRESHOLD:
                label = f"{best_student} ({best_score:.2f})"
                box_color = (0, 200, 0)
            else:
                label = f"Unknown ({best_score:.2f})"
                box_color = (0, 0, 255)

            cv2.rectangle(
                frame, (x1, y1), (x2, y2), box_color, 2
            )

            text_y = max(25, y1 - 10)

            cv2.putText(
                frame,
                label,
                (x1, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                box_color,
                2
            )

        cv2.imshow("Student Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()
