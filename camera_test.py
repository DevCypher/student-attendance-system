import cv2

print("Starting camera test...")

camera = cv2.VideoCapture(0)

print("Camera object created.")

if not camera.isOpened():
    print("ERROR: Camera could not be opened.")
else:
    print("Camera opened successfully.")

success, frame = camera.read()

if success:
    print("Frame captured successfully.")
    print("Frame size:", frame.shape)

    cv2.imshow("Webcam Test", frame)
    cv2.waitKey(0)
else:
    print("ERROR: Could not capture a frame.")

camera.release()
cv2.destroyAllWindows()

print("Camera test finished.")