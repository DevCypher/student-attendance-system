
import cv2
import time
import os
from ultralytics import YOLO

# ---------------- CONFIGURATION ----------------

VIDEO_PATH = r"C:\Mini\testvid2.mp4"

MODEL_PATH = r"C:\Mini\Student_attendace_system\yolo_experiment\yolov8n-face.pt"

OUTPUT_PATH = r"C:\Mini\Student_attendace_system\yolo_experiment\results\yolo_test1_annotated.mp4"

CONFIDENCE_THRESHOLD = 0.25
IMAGE_SIZE = 960

# ------------------------------------------------

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

print("Loading YOLO face detection model...")

model = YOLO(MODEL_PATH)

print("YOLO model loaded successfully!")

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

video_fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video resolution: {frame_width} x {frame_height}")
print(f"Video FPS: {video_fps:.2f}")
print(f"Total frames: {total_frames}")

output_fps = video_fps if video_fps > 0 else 25

out = cv2.VideoWriter(
    OUTPUT_PATH,
    cv2.VideoWriter_fourcc(*"mp4v"),
    output_fps,
    (frame_width, frame_height)
)

if not out.isOpened():
    raise RuntimeError("Could not create output video.")

frame_count = 0
total_faces = 0
max_faces = 0

start_time = time.time()

print("\nStarting detection...")
print("Press Q to stop early.\n")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = model.predict(
        source=frame,
        imgsz=IMAGE_SIZE,
        conf=CONFIDENCE_THRESHOLD,
        max_det=500,
        verbose=False,
        device="cpu"
    )

    result = results[0]

    face_count = len(result.boxes)

    frame_count += 1
    total_faces += face_count
    max_faces = max(max_faces, face_count)

    annotated_frame = result.plot()

    cv2.putText(
        annotated_frame,
        f"Faces: {face_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    out.write(annotated_frame)

    cv2.imshow("YOLO Face Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

end_time = time.time()

elapsed_time = end_time - start_time

cap.release()
out.release()
cv2.destroyAllWindows()

print("\n------------- RESULTS -------------")

print(f"Frames processed: {frame_count}")
print(f"Total face detections: {total_faces}")
print(f"Maximum faces in one frame: {max_faces}")

if elapsed_time > 0:
    print(f"Processing time: {elapsed_time:.2f} seconds")
    print(f"Processing FPS: {frame_count / elapsed_time:.2f}")

print(f"\nSaved video:\n{OUTPUT_PATH}")
