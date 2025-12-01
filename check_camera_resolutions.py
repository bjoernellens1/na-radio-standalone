import cv2

def check_resolutions():
    # List of common resolutions to check
    resolutions = [
        (1920, 1080),
        (1280, 720),
        (1024, 768),
        (800, 600),
        (640, 480),
        (320, 240)
    ]
    
    # Try to open the camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open camera")
        return

    print("Checking camera resolutions...")
    supported = []
    for w, h in resolutions:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        
        actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
        if int(actual_w) == w and int(actual_h) == h:
            supported.append((w, h))
            print(f"Supported: {w}x{h}")
        else:
            print(f"Not supported: {w}x{h} (got {int(actual_w)}x{int(actual_h)})")
            
    cap.release()
    return supported

if __name__ == "__main__":
    check_resolutions()
