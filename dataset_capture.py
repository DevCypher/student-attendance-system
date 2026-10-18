import cv2
import os
import time

# -----------------------------
# SETTINGS
# -----------------------------

CAMERA_INDEX = 0
IMAGES_PER_VIEW = 15
CAPTURE_INTERVAL = 1.0

DATASET_DIR = "dataset"


# -----------------------------
# GET STUDENT ID
# -----------------------------

student_id = input("Enter student ID: ").strip()

if not student_id:
    print("Student ID cannot be empty.")
    exit()


# -----------------------------
# CREATE FOLDERS
# -----------------------------

views = ["front", "left", "right"]

for view in views:
    folder = os.path.join(DATASET_DIR, student_id, view)
    os.makedirs(folder, exist_ok=True)


# -----------------------------
# OPEN CAMERA
# -----------------------------

camera = cv2.VideoCapture(CAMERA_INDEX)

if not camera.isOpened():
    print("Could not open camera.")
    exit()


# -----------------------------
# CAPTURE FUNCTION
# -----------------------------

def capture_view(view):
    folder = os.path.join(DATASET_DIR, student_id, view)

    image_number = 1
    last_capture_time = 0

    while image_number <= IMAGES_PER_VIEW:

        success, frame = camera.read()

        if not success:
            print("Could not read frame.")
            break

        current_time = time.time()

        # Capture an image every CAPTURE_INTERVAL seconds
        if current_time - last_capture_time >= CAPTURE_INTERVAL:

            filename = os.path.join(
                folder,
                f"{image_number:03d}.jpg"
            )

            cv2.imwrite(filename, frame)

            image_number += 1
            last_capture_time = current_time

        # Display information on the camera feed
        text = f"{view.upper()}  {image_number - 1}/{IMAGES_PER_VIEW}"

        cv2.putText(
            frame,
            text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow("Dataset Capture", frame)

        # Press Q to stop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return False

    return True


# -----------------------------
# CAPTURE EACH VIEW
# -----------------------------

for view in views:

    print()
    print("--------------------------------")
    print(f"Prepare for {view.upper()} view")
    print("Press ENTER when ready...")
    input()

    success = capture_view(view)

    if not success:
        break


# -----------------------------
# CLEAN UP
# -----------------------------

camera.release()
cv2.destroyAllWindows()

print()
print("Dataset capture completed.")