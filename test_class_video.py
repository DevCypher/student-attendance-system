
import cv2
import csv
import numpy as np

from pathlib import Path
from collections import Counter
from insightface.app import FaceAnalysis


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

VIDEO_PATH = Path(r"C:\Mini\testvid1.mp4")

DATABASE_PATH = Path(
    r"C:\Mini\Student_attendace_system\face_database_v2"
    r"\student_embeddings_v2.npz"
)

OUTPUT_DIR = Path(
    r"C:\Mini\Student_attendace_system\video_test_results"
)

# Process approximately one frame per second
FRAME_INTERVAL_SECONDS = 1.0

# Provisional thresholds; validate using your results
SIMILARITY_THRESHOLD = 0.40
MINIMUM_SCORE_MARGIN = 0.05

# Save annotated snapshots for visual review
SAVE_ANNOTATED_FRAMES = True


# --------------------------------------------------
# 2. LOAD DATABASE
# --------------------------------------------------

def main():
    if not VIDEO_PATH.is_file():
        raise FileNotFoundError(
            f"Video not found: {VIDEO_PATH}"
        )

    if not DATABASE_PATH.is_file():
        raise FileNotFoundError(
            f"Embedding database not found: {DATABASE_PATH}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    frames_dir = OUTPUT_DIR / "sampled_frames"
    if SAVE_ANNOTATED_FRAMES:
        frames_dir.mkdir(parents=True, exist_ok=True)

    database = np.load(DATABASE_PATH, allow_pickle=False)

    known_embeddings = np.asarray(
        database["embeddings"], dtype=np.float32
    )
    known_ids = database["student_ids"].astype(str)

    if len(known_embeddings) != len(known_ids):
        raise ValueError(
            "Embedding count and student ID count do not match."
        )

    # Normalize defensively before cosine similarity
    norms = np.linalg.norm(
        known_embeddings, axis=1, keepdims=True
    )
    known_embeddings = known_embeddings / np.maximum(
        norms, 1e-12
    )

    student_ids = sorted(set(known_ids.tolist()))

    print(f"Students in database: {len(student_ids)}")
    print(f"Stored embeddings: {len(known_embeddings)}")
    print(f"Video: {VIDEO_PATH}")
    print("\nLoading InsightFace buffalo_l...")

    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"]
    )
    app.prepare(ctx_id=-1, det_size=(640, 640))

    recognition_model = app.models.get("recognition")

    if recognition_model is None:
        raise RuntimeError(
            "InsightFace recognition model was not loaded."
        )

    # Find the embedding indices belonging to each student
    student_indices = {
        sid: np.flatnonzero(known_ids == sid)
        for sid in student_ids
    }


    # --------------------------------------------------
    # 3. OPEN VIDEO
    # --------------------------------------------------

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {VIDEO_PATH}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if fps <= 0:
        cap.release()
        raise RuntimeError("Could not determine video FPS.")

    duration = total_frames / fps
    frame_interval = max(
        1, int(round(fps * FRAME_INTERVAL_SECONDS))
    )

    print(f"Video FPS: {fps:.2f}")
    print(f"Duration: {duration:.1f} seconds")
    print(
        f"Sampling approximately every "
        f"{FRAME_INTERVAL_SECONDS:g} second(s)"
    )


    # --------------------------------------------------
    # 4. PROCESS SAMPLED FRAMES
    # --------------------------------------------------

    results = []
    prediction_counts = Counter()
    frame_number = 0
    sampled_number = 0

    while True:
        success, frame = cap.read()

        if not success:
            break

        if frame_number % frame_interval != 0:
            frame_number += 1
            continue

        sampled_number += 1
        timestamp = frame_number / fps

        faces = app.get(frame)

        annotated = frame.copy()
        frame_predictions = []

        for face_number, face in enumerate(faces, start=1):
            x1, y1, x2, y2 = face.bbox.astype(int)

            # Keep bounding coordinates inside the frame
            height, width = frame.shape[:2]
            x1 = max(0, min(x1, width - 1))
            x2 = max(0, min(x2, width))
            y1 = max(0, min(y1, height - 1))
            y2 = max(0, min(y2, height))

            face_width = x2 - x1
            face_height = y2 - y1

            if face_width < 40 or face_height < 40:
                results.append({
                    "timestamp_seconds": round(timestamp, 2),
                    "frame_number": frame_number,
                    "face_number": face_number,
                    "predicted_id": "Unknown",
                    "best_similarity": "",
                    "score_margin": "",
                    "status": "Face too small"
                })
                continue

            # Add a 10% margin to the detected face crop
            margin_x = int(face_width * 0.10)
            margin_y = int(face_height * 0.10)

            cx1 = max(0, x1 - margin_x)
            cx2 = min(width, x2 + margin_x)
            cy1 = max(0, y1 - margin_y)
            cy2 = min(height, y2 + margin_y)

            face_crop = frame[cy1:cy2, cx1:cx2]

            if face_crop.size == 0:
                continue

            try:
                # Match the database-generation method:
                # extract features directly from the cropped face.
                embedding = recognition_model.get_feat(
                    face_crop
                )
                embedding = np.asarray(
                    embedding, dtype=np.float32
                ).reshape(-1)

                if not np.all(np.isfinite(embedding)):
                    raise ValueError(
                        "Invalid embedding values"
                    )

                embedding_norm = np.linalg.norm(embedding)

                if embedding_norm < 1e-10:
                    raise ValueError("Zero embedding")

                embedding /= embedding_norm

                # Cosine similarities against all stored images
                similarities = known_embeddings @ embedding

                # Rank students using their top 3 image scores
                student_scores = {}

                for sid, indices in student_indices.items():
                    scores = similarities[indices]
                    top_scores = np.sort(scores)[-3:]
                    student_scores[sid] = float(
                        np.mean(top_scores)
                    )

                ranked = sorted(
                    student_scores.items(),
                    key=lambda item: item[1],
                    reverse=True
                )

                best_id, best_score = ranked[0]

                second_score = (
                    ranked[1][1] if len(ranked) > 1 else -1.0
                )
                margin = best_score - second_score

                if (
                    best_score >= SIMILARITY_THRESHOLD
                    and margin >= MINIMUM_SCORE_MARGIN
                ):
                    prediction = best_id
                    status = "Accepted provisional match"
                else:
                    prediction = "Unknown"
                    status = "Below threshold or ambiguous"

                prediction_counts[prediction] += 1

                results.append({
                    "timestamp_seconds": round(timestamp, 2),
                    "frame_number": frame_number,
                    "face_number": face_number,
                    "predicted_id": prediction,
                    "best_similarity": round(best_score, 4),
                    "score_margin": round(margin, 4),
                    "status": status
                })

                frame_predictions.append(
                    (x1, y1, x2, y2, prediction, best_score)
                )

            except Exception as error:
                results.append({
                    "timestamp_seconds": round(timestamp, 2),
                    "frame_number": frame_number,
                    "face_number": face_number,
                    "predicted_id": "Error",
                    "best_similarity": "",
                    "score_margin": "",
                    "status": str(error)
                })

        # Annotate sampled frames for manual review
        for x1, y1, x2, y2, prediction, score in frame_predictions:
            label = f"{prediction} ({score:.2f})"

            cv2.rectangle(
                annotated, (x1, y1), (x2, y2),
                (0, 255, 0) if prediction != "Unknown"
                else (0, 165, 255),
                2
            )

            cv2.putText(
                annotated, label,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 2,
                cv2.LINE_AA
            )

        if SAVE_ANNOTATED_FRAMES:
            frame_path = (
                frames_dir /
                f"frame_{sampled_number:05d}.jpg"
            )
            cv2.imwrite(str(frame_path), annotated)

        print(
            f"\rProcessed sample {sampled_number} | "
            f"Time {timestamp:.1f}s | "
            f"Faces detected: {len(faces)}",
            end="",
            flush=True
        )

        frame_number += 1

    cap.release()
    print("\n\nVideo processing complete.")


    # --------------------------------------------------
    # 5. SAVE RESULTS
    # --------------------------------------------------

    csv_path = OUTPUT_DIR / "recognition_results.csv"

    fieldnames = [
        "timestamp_seconds",
        "frame_number",
        "face_number",
        "predicted_id",
        "best_similarity",
        "score_margin",
        "status"
    ]

    with open(
        csv_path, "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=fieldnames
        )
        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 48)
    print("CLASS VIDEO TEST SUMMARY")
    print("=" * 48)
    print(f"Duration: {duration:.1f} seconds")
    print(f"Sampled frames: {sampled_number}")
    print(f"Face records: {len(results)}")
    print("\nPredictions by face detection:")

    for student_id, count in prediction_counts.most_common():
        print(f"  {student_id}: {count}")

    print(f"\nCSV results: {csv_path}")

    if SAVE_ANNOTATED_FRAMES:
        print(f"Annotated frames: {frames_dir}")

    print(
        "\nNote: These are provisional predictions, not "
        "verified attendance or accuracy measurements."
    )


if __name__ == "__main__":
    main()

