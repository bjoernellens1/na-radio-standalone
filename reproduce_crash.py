import cv2
import sys

try:
    print("Attempting to open camera 99...")
    cap = cv2.VideoCapture(99)
    if not cap.isOpened():
        print("Camera 99 could not be opened (isOpened() returned False).")
    else:
        print("Camera 99 opened successfully.")
        ret, frame = cap.read()
        if ret:
            print("Frame read successfully.")
        else:
            print("Failed to read frame.")
    cap.release()
except Exception as e:
    print(f"Caught exception: {e}")
except BaseException as e:
    print(f"Caught BaseException: {e}")

print("Finished.")
