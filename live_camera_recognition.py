
import cv2
import time
import threading
import numpy as np
import onnxruntime as ort

from pathlib import Path
from insightface.app import FaceAnalysis


# =========================================================
# 1. CONFIGURATION
# =========================================================

DATABASE_PATH = Path(
    r"C:\Mini\Student_attendace_system\face_database_v2"
    r"\student_embeddings_v2.npz"
)

CAMERA_INDEX = 0

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# Smaller detection input generally means faster inference.
DETECTION_SIZE = (320, 320)

# Recognition thresholds are provisional; validate them.
SIMILARITY_THRESHOLD = 0.40
MINIMUM_SCORE_MARGIN = 0.05

WINDOW_NAME = "Student Attendance - GPU Recognition"


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

    print(f"Students: {len(student_indices)}")
    print(f"Stored embeddings: {len(embeddings)}")

    return embeddings, student_indices


# =========================================================
# 3. RECOGNIZE FACES IN ONE FRAME
# =========================================================

def recognize_frame(frame, app, recognizer,
                    known_embeddings, student_indices):

    faces = app.get(frame)
    results = []

    height, width = frame.shape[:2]

    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)

        x1 = max(0, min(x1, width - 1))
        x2 = max(0, min(x2, width))
        y1 = max(0, min(y1, height - 1))
        y2 = max(0, min(y2, height))

        face_width = x2 - x1
        face_height = y2 - y1

        if face_width < 40 or face_height < 40:
            results.append(
                (x1, y1, x2, y2, "Too small", 0.0, False)
            )
            continue

        # Expand the detected face slightly.
        mx = int(face_width * 0.10)
        my = int(face_height * 0.10)

        crop = frame[
            max(0, y1 - my):min(height, y2 + my),
            max(0, x1 - mx):min(width, x2 + mx)
        ]

        if crop.size == 0:
            continue

        try:
            # Use the same direct embedding method as database
            # generation, since Dataset2 already contains crops.
            embedding = np.asarray(
                recognizer.get_feat(crop),
                dtype=np.float32
            ).reshape(-1)

            norm = np.linalg.norm(embedding)

            if (
                norm < 1e-10
                or not np.all(np.isfinite(embedding))
            ):
                continue

            embedding /= norm

            # Compare against stored student image embeddings.
            similarities = known_embeddings @ embedding

            # Average each student's top three image similarities.
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
# 4. BACKGROUND INFERENCE WORKER
# =========================================================

def inference_worker(state, lock, app, recognizer,
                     known_embeddings, student_indices):

    last_processed_id = -1

    while True:
        with lock:
            if not state["running"]:
                break

            frame_id = state["frame_id"]

            if (
                state["latest_frame"] is None
                or frame_id == last_processed_id
            ):
                frame = None
            else:
                frame = state["latest_frame"].copy()
                last_processed_id = frame_id

        if frame is None:
            time.sleep(0.005)
            continue

        start = time.perf_counter()

        try:
            results = recognize_frame(
                frame,
                app,
                recognizer,
                known_embeddings,
                student_indices
            )

            elapsed = time.perf_counter() - start

            with lock:
                state["results"] = results
                state["result_frame_id"] = frame_id
                state["inference_seconds"] = elapsed
                state["completed_at"] = time.monotonic()

        except Exception as error:
            print(f"\nInference worker error: {error}")
            time.sleep(0.05)


# =========================================================
# 5. MAIN WEBCAM APPLICATION
# =========================================================

def main():
    known_embeddings, student_indices = load_database()

    # Load CUDA/cuDNN DLLs from installed NVIDIA packages.
    ort.preload_dlls(directory="")

    print("\nAvailable ONNX Runtime providers:")
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
        raise RuntimeError("Recognition model not found.")

    print("\nModel session providers:")

    for name, model in app.models.items():
        print(f"{name}: {model.session.get_providers()}")

    # Open the webcam using the Windows DirectShow backend.
    camera = cv2.VideoCapture(
        CAMERA_INDEX, cv2.CAP_DSHOW
    )

    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(CAMERA_INDEX)

    if not camera.isOpened():
        raise RuntimeError(
            "Cannot open webcam. Try CAMERA_INDEX = 1."
        )

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    state = {
        "running": True,
        "latest_frame": None,
        "frame_id": 0,
        "results": [],
        "result_frame_id": -1,
        "inference_seconds": 0.0,
        "completed_at": 0.0
    }

    lock = threading.Lock()

    worker = threading.Thread(
        target=inference_worker,
        args=(
            state, lock, app, recognizer,
            known_embeddings, student_indices
        ),
        daemon=True
    )
    worker.start()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    previous_time = time.perf_counter()
    display_fps = 0.0

    print("\nLive GPU recognition started.")
    print("Q / ESC: quit | SPACE: pause display")

    paused = False
    last_display_frame = None

    try:
        while True:
            if not paused:
                ok, frame = camera.read()

                if not ok:
                    continue

                # Mirror the image like a normal webcam preview.
                frame = cv2.flip(frame, 1)
                last_display_frame = frame.copy()

                with lock:
                    state["latest_frame"] = frame.copy()
                    state["frame_id"] += 1

            if last_display_frame is None:
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                continue

            display = last_display_frame.copy()

            with lock:
                results = list(state["results"])
                inference_seconds = state["inference_seconds"]
                completed_at = state["completed_at"]

            # Avoid showing stale boxes for too long if inference lags.
            results_are_fresh = (
                completed_at > 0
                and time.monotonic() - completed_at < 0.7
            )

            recognized_count = 0
            face_count = len(results) if results_are_fresh else 0

            if results_are_fresh:
                for (
                    x1, y1, x2, y2,
                    label, score, accepted
                ) in results:

                    color = (
                        (0, 200, 0)
                        if accepted
                        else (0, 165, 255)
                    )

                    if accepted:
                        recognized_count += 1

                    cv2.rectangle(
                        display,
                        (x1, y1),
                        (x2, y2),
                        color,
                        2
                    )

                    cv2.putText(
                        display,
                        f"{label} ({score:.2f})",
                        (x1, max(22, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        color,
                        2,
                        cv2.LINE_AA
                    )

            now = time.perf_counter()
            delta = now - previous_time

            if delta > 0:
                instantaneous_fps = 1.0 / delta
                display_fps = (
                    0.9 * display_fps
                    + 0.1 * instantaneous_fps
                    if display_fps else instantaneous_fps
                )

            previous_time = now

            # Status overlay.
            cv2.rectangle(
                display, (8, 8), (350, 118),
                (25, 25, 25), -1
            )

            status = (
                "GPU INFERENCE"
                if "CUDAExecutionProvider"
                in ort.get_available_providers()
                else "CPU INFERENCE"
            )

            cv2.putText(
                display, status,
                (18, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 255, 255), 2
            )

            cv2.putText(
                display,
                f"Display FPS: {display_fps:.1f}",
                (18, 53),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1
            )

            cv2.putText(
                display,
                f"Faces: {face_count}  Matches: {recognized_count}",
                (18, 76),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1
            )

            cv2.putText(
                display,
                f"Inference: {inference_seconds * 1000:.0f} ms",
                (18, 99),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1
            )

            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break

            elif key == ord(" "):
                paused = not paused

    finally:
        with lock:
            state["running"] = False

        worker.join(timeout=2.0)
        camera.release()
        cv2.destroyAllWindows()

    print("\nWebcam recognition stopped.")


if __name__ == "__main__":
    main()
