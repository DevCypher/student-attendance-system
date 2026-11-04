
import cv2
import csv
import time
import threading
import numpy as np
import onnxruntime as ort

from pathlib import Path
from insightface.app import FaceAnalysis


# =========================================================
# 1. CONFIGURATION
# =========================================================

VIDEO_PATH = Path(r"C:\Mini\testvid2.mp4")

# Use the exact existing project folder name
DATABASE_PATH = Path(
    r"C:\Mini\Student_attendace_system\face_database_v2"
    r"\student_embeddings_v2.npz"
)

OUTPUT_DIR = Path(
    r"C:\Mini\Student_attendace_system\video_test_results"
)

DETECTION_SIZE = (320, 320)

SIMILARITY_THRESHOLD = 0.40
MINIMUM_SCORE_MARGIN = 0.05

WINDOW_NAME = "Class Video - GPU Face Recognition"


# =========================================================
# 2. LOAD DATABASE
# =========================================================

def load_database():
    if not DATABASE_PATH.is_file():
        raise FileNotFoundError(
            f"Database not found:\n{DATABASE_PATH}"
        )

    data = np.load(DATABASE_PATH, allow_pickle=False)

    embeddings = np.asarray(
        data["embeddings"], dtype=np.float32
    )
    student_ids = data["student_ids"].astype(str)

    if len(embeddings) != len(student_ids):
        raise ValueError("Embedding and ID counts do not match.")

    norms = np.linalg.norm(
        embeddings, axis=1, keepdims=True
    )
    embeddings /= np.maximum(norms, 1e-12)

    student_indices = {
        sid: np.flatnonzero(student_ids == sid)
        for sid in sorted(set(student_ids.tolist()))
    }

    print(f"Students loaded: {len(student_indices)}")
    print(f"Embeddings loaded: {len(embeddings)}")

    return embeddings, student_indices


# =========================================================
# 3. DETECT AND RECOGNIZE FACES
# =========================================================

def recognize_frame(frame, app, recognizer,
                    embeddings, student_indices):

    faces = app.get(frame)
    results = []

    height, width = frame.shape[:2]

    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)

        x1 = max(0, min(x1, width - 1))
        x2 = max(0, min(x2, width))
        y1 = max(0, min(y1, height - 1))
        y2 = max(0, min(y2, height))

        fw, fh = x2 - x1, y2 - y1

        if fw < 40 or fh < 40:
            results.append(
                (x1, y1, x2, y2, "Too small", 0.0, False)
            )
            continue

        mx, my = int(fw * 0.10), int(fh * 0.10)

        crop = frame[
            max(0, y1 - my):min(height, y2 + my),
            max(0, x1 - mx):min(width, x2 + mx)
        ]

        if crop.size == 0:
            continue

        try:
            # Use the same direct embedding method as Dataset2.
            vector = np.asarray(
                recognizer.get_feat(crop),
                dtype=np.float32
            ).reshape(-1)

            norm = np.linalg.norm(vector)

            if norm < 1e-10 or not np.all(np.isfinite(vector)):
                continue

            vector /= norm

            # Cosine similarity against all stored embeddings
            similarities = embeddings @ vector

            scores = {}

            for sid, indices in student_indices.items():
                student_similarities = similarities[indices]
                top = np.sort(student_similarities)[-3:]
                scores[sid] = float(np.mean(top))

            ranked = sorted(
                scores.items(),
                key=lambda item: item[1],
                reverse=True
            )

            best_id, best_score = ranked[0]

            second_score = (
                ranked[1][1] if len(ranked) > 1 else -1.0
            )
            margin = best_score - second_score

            accepted = (
                best_score >= SIMILARITY_THRESHOLD
                and margin >= MINIMUM_SCORE_MARGIN
            )

            label = best_id if accepted else "Unknown"

            results.append(
                (
                    x1, y1, x2, y2,
                    label, best_score, accepted
                )
            )

        except Exception as error:
            print(f"\nRecognition error: {error}")

    return results


# =========================================================
# 4. BACKGROUND INFERENCE
# =========================================================

def inference_worker(state, lock, app, recognizer,
                     embeddings, student_indices):

    last_frame_id = -1

    while True:
        with lock:
            if not state["running"]:
                break

            frame_id = state["frame_id"]

            if (
                state["latest_frame"] is None
                or frame_id == last_frame_id
            ):
                frame = None
            else:
                frame = state["latest_frame"].copy()
                last_frame_id = frame_id

        if frame is None:
            time.sleep(0.005)
            continue

        start = time.perf_counter()

        try:
            results = recognize_frame(
                frame, app, recognizer,
                embeddings, student_indices
            )

            elapsed = time.perf_counter() - start

            with lock:
                state["results"] = results
                state["result_frame_id"] = frame_id
                state["inference_seconds"] = elapsed
                state["completed_at"] = time.monotonic()
                state["result_seq"] += 1

        except Exception as error:
            print(f"\nInference error: {error}")
            time.sleep(0.05)


# =========================================================
# 5. MAIN VIDEO TEST
# =========================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    embeddings, student_indices = load_database()

    # Load CUDA/cuDNN libraries from NVIDIA Python packages
    ort.preload_dlls(directory="")

    print("\nAvailable providers:")
    print(ort.get_available_providers())

    print("\nLoading InsightFace on GPU...")

    app = FaceAnalysis(
        name="buffalo_l",
        providers=[
            "CUDAExecutionProvider",
            "CPUExecutionProvider"
        ]
    )
    app.prepare(ctx_id=0, det_size=DETECTION_SIZE)

    recognizer = app.models.get("recognition")

    if recognizer is None:
        raise RuntimeError("Recognition model not loaded.")

    print("\nModel session providers:")
    for name, model in app.models.items():
        print(f"{name}: {model.session.get_providers()}")

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(source_fps) or source_fps <= 0:
        source_fps = 25.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / source_fps

    print(f"\nVideo FPS: {source_fps:.2f}")
    print(f"Video duration: {duration:.1f} seconds")

    state = {
        "running": True,
        "latest_frame": None,
        "frame_id": 0,
        "results": [],
        "result_frame_id": -1,
        "inference_seconds": 0.0,
        "completed_at": 0.0,
        "result_seq": 0
    }

    lock = threading.Lock()

    worker = threading.Thread(
        target=inference_worker,
        args=(
            state, lock, app, recognizer,
            embeddings, student_indices
        ),
        daemon=True
    )
    worker.start()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    csv_path = OUTPUT_DIR / "recognition_results.csv"
    report_rows = []
    last_saved_result_seq = 0
    last_frame = None
    frame_number = 0
    paused = False

    # -----------------------------------------------------
    # PLAYBACK SPEED
    # 1.00 = normal speed
    # 0.75 = 25% slower
    # 0.50 = half speed
    # -----------------------------------------------------
    PLAYBACK_SPEED = 0.75

    frame_delay_ms = max(
        1,
        round(1000 / (source_fps * PLAYBACK_SPEED))
    )

    print("\nVideo recognition started.")
    print("SPACE = pause/resume | Q or ESC = quit")

    try:
        while True:
            if not paused:
                ok, frame = cap.read()

                if not ok:
                    break

                frame_number += 1
                last_frame = frame.copy()

                with lock:
                    state["latest_frame"] = frame.copy()
                    state["frame_id"] += 1

            if last_frame is None:
                break

            display = last_frame.copy()

            with lock:
                results = list(state["results"])
                result_frame_id = state["result_frame_id"]
                inference_seconds = state["inference_seconds"]
                completed_at = state["completed_at"]
                result_seq = state["result_seq"]

            # Record each completed inference result once.
            if result_seq > last_saved_result_seq:
                timestamp = result_frame_id / source_fps

                for face_num, item in enumerate(results, start=1):
                    x1, y1, x2, y2, label, score, accepted = item

                    report_rows.append({
                        "timestamp_seconds": round(timestamp, 2),
                        "frame_number": result_frame_id,
                        "face_number": face_num,
                        "predicted_id": label,
                        "similarity": round(float(score), 4),
                        "accepted_match": accepted
                    })

                last_saved_result_seq = result_seq

            # Only draw results while they're reasonably fresh.
            fresh = (
                completed_at > 0
                and time.monotonic() - completed_at < 1.0
            )

            recognized_count = 0

            if fresh:
                for x1, y1, x2, y2, label, score, accepted in results:
                    color = (
                        (0, 200, 0)
                        if accepted
                        else (0, 165, 255)
                    )

                    recognized_count += int(accepted)

                    cv2.rectangle(
                        display, (x1, y1), (x2, y2), color, 2
                    )

                    cv2.putText(
                        display,
                        f"{label} ({score:.2f})",
                        (x1, max(22, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, color, 2, cv2.LINE_AA
                    )

            # On-screen status panel
            cv2.rectangle(
                display, (8, 8), (375, 125), (25, 25, 25), -1
            )

            cv2.putText(
                display, "CLASS VIDEO - GPU RECOGNITION",
                (18, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.52, (0, 255, 255), 2
            )

            cv2.putText(
                display,
                f"Video time: {frame_number / source_fps:.1f}s",
                (18, 53), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1
            )

            cv2.putText(
                display,
                f"Faces: {len(results) if fresh else 0}"
                f" | Matches: {recognized_count}",
                (18, 76), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1
            )

            cv2.putText(
                display,
                f"Inference: {inference_seconds * 1000:.0f} ms",
                (18, 99), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1
            )

            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(
                0 if paused else frame_delay_ms
            ) & 0xFF

            if key in (ord("q"), 27):
                break
            elif key == ord(" "):
                paused = not paused

    finally:
        with lock:
            state["running"] = False

        worker.join(timeout=2.0)
        cap.release()
        cv2.destroyAllWindows()

        # Save the collected recognition results
        fields = [
            "timestamp_seconds",
            "frame_number",
            "face_number",
            "predicted_id",
            "similarity",
            "accepted_match"
        ]

        with open(
            csv_path, "w", newline="", encoding="utf-8-sig"
        ) as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(report_rows)

        print("\nVideo test finished.")
        print(f"Frames read: {frame_number}")
        print(f"Recognition records: {len(report_rows)}")
        print(f"CSV report: {csv_path}")


if __name__ == "__main__":
    main()
