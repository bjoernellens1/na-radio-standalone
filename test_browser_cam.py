import sys
import os
import requests
import cv2
import numpy as np

def test_process_frame():
    # Create a dummy image
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(img, 'Test Frame', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    ret, buffer = cv2.imencode('.jpg', img)
    
    url = 'http://localhost:5000/process_frame'
    files = {'frame': ('test.jpg', buffer.tobytes(), 'image/jpeg')}
    
    try:
        resp = requests.post(url, files=files)
        print(f"Status: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            print("Success:", data['success'])
            print("Predictions:", data['predictions'])
            print("Image received (len):", len(data['image']))
        else:
            print("Error:", resp.text)
            
    except Exception as e:
        print(f"Request failed: {e}")
        # Note: This test expects the server to be running. 
        # Since we just modified the code, we need to restart the server/container.

if __name__ == "__main__":
    test_process_frame()
