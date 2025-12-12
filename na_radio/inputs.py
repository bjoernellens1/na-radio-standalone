import os
import glob
import time
import cv2

class ImageFolderCapture:
    def __init__(self, path):
        self.path = path
        self.images = []
        self.current_idx = 0
        self.last_switch = 0
        self.interval = 1.0 # 1 second per image
        
        if os.path.isdir(path):
            # Folder mode
            exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp')
            for ext in exts:
                self.images.extend(glob.glob(os.path.join(path, ext)))
                self.images.extend(glob.glob(os.path.join(path, ext.upper())))
            self.images.sort()
        elif os.path.isfile(path):
            # Single image mode
            self.images = [path]
            
        if not self.images:
            raise ValueError(f"No images found in {path}")
            
        print(f"ImageFolderCapture loaded {len(self.images)} images from {path}")

    def isOpened(self):
        return bool(self.images)

    def read(self):
        if not self.images:
            return False, None
            
        now = time.time()
        if len(self.images) > 1 and now - self.last_switch > self.interval:
            self.current_idx = (self.current_idx + 1) % len(self.images)
            self.last_switch = now
            
        img_path = self.images[self.current_idx]
        try:
            # Use cv2 to read
            frame = cv2.imread(img_path)
            if frame is None:
                print(f"Failed to read {img_path}")
                # Try next one immediately
                self.current_idx = (self.current_idx + 1) % len(self.images)
                return True, None 
            return True, frame
        except Exception as e:
            print(f"Error reading {img_path}: {e}")
            return False, None

    def release(self):
        pass
