import threading
import time
import cv2
import numpy as np
import torch
import gc
import os
import glob
from typing import Optional, List, Tuple

from .encoders import load_encoder
from .utils import get_device, preprocess_frame, cosine_similarity_matrix
from .inputs import ImageFolderCapture



class Manager:
    def __init__(self, device_index=0, video_file=None, labels=None, encoder_name='radio', device=None):
        self.input_config = {
            'type': 'browser_webcam',
            'value': None,
            'update_needed': False
        }
        if video_file:
            self.input_config['type'] = 'video'
            self.input_config['value'] = video_file
            
        self.labels = labels or ["person", "car", "dog", "cat", "tree"]
        self.device = device or get_device()
        self.encoder_name = encoder_name
        
        # State
        self.camera_open = False
        self.current_frame = None
        self.current_fps = 0.0
        self.current_pred = []
        self.last_inference_time = 0.0
        self.heatmap_enabled = False
        self.latest_heatmap = None
        
        # Locks
        self.frame_lock = threading.Lock()
        self.model_lock = threading.Lock()
        self.inference_lock = threading.Lock()
        self.heatmap_lock = threading.Lock()
        self.input_lock = threading.Lock()
        self.label_vecs_lock = threading.Lock()
        
        # Model
        self.current_encoder = None
        self.current_label_vecs = None
        self.predictions_enabled = False
        self.label_update_needed = True
        
        # Threads
        self.running = False
        self.capture_thread = None
        self.inference_thread = None

    def start(self):
        self.running = True
        
        # Lazy loading: Do not load model here
        # self.load_model(self.encoder_name)
        
        # Start threads
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        
        self.inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.inference_thread.start()
        
        print("Manager started.")

    def stop(self):
        self.running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=1.0)
        if self.inference_thread:
            self.inference_thread.join(timeout=1.0)
        print("Manager stopped.")

    def load_model(self, model_name):
        print(f"Switching to model: {model_name}")
        try:
            old_encoder = None
            with self.model_lock:
                old_encoder = self.current_encoder
                self.current_encoder = None
            
            # Wait for inference to pause
            with self.inference_lock:
                pass

            if old_encoder is not None:
                if hasattr(old_encoder, 'unload'):
                    old_encoder.unload()
                del old_encoder
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            enc, name = load_encoder(preferred=model_name, device=self.device)
            
            with self.model_lock:
                self.current_encoder = enc
                self.encoder_name = name
            
            self.predictions_enabled = False
            self.label_update_needed = True
            print(f"Model loaded: {name}")
            return True
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False

    def set_input(self, type, value):
        with self.input_lock:
            self.input_config['type'] = type
            self.input_config['value'] = value
            self.input_config['update_needed'] = True

    def update_labels(self, new_labels):
        self.labels = new_labels
        self.label_update_needed = True
        self.predictions_enabled = False

    def get_status(self):
        encoder_status = self.encoder_name if self.current_encoder is not None else "Not Loaded"
        return {
            'camera_open': self.camera_open,
            'encoder': encoder_status,
            'fps': round(self.current_fps, 1),
            'inference_time_ms': round(self.last_inference_time, 1),
            'predictions': self.current_pred
        }

    def _capture_loop(self):
        cap = None
        
        def open_source():
            nonlocal cap
            src_type = self.input_config['type']
            val = self.input_config['value']
            
            if cap is not None:
                cap.release()
                
            print(f"Opening source: {src_type} = {val}")
            try:
                if src_type == 'webcam':
                    print("Server-side webcam is disabled. Use browser webcam.")
                    cap = None
                elif src_type == 'video':
                    try:
                        cap = cv2.VideoCapture(val)
                    except Exception as e:
                        print(f"Failed to open video {val}: {e}")
                        cap = None
                elif src_type == 'image' or src_type == 'folder':
                    cap = ImageFolderCapture(val)
                elif src_type == 'browser_webcam':
                    cap = None
                    return True
                else:
                    print(f"Unknown source type: {src_type}")
                    cap = None
            except Exception as e:
                print(f"Failed to open source: {e}")
                cap = None

            if cap is None or not cap.isOpened():
                return False
            return True

        if not open_source():
            self.camera_open = False
        else:
            self.camera_open = True
            
        frame_count = 0
        start_time = time.time()
        
        while self.running:
            update = False
            with self.input_lock:
                if self.input_config['update_needed']:
                    update = True
                    self.input_config['update_needed'] = False
            
            if update:
                if open_source():
                    self.camera_open = True
                else:
                    self.camera_open = False
            
            if cap is None or not self.camera_open:
                time.sleep(0.5)
                continue

            ret, frame = cap.read()
            if not ret:
                if self.input_config['type'] == 'video':
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                time.sleep(0.1)
                continue
                
            if frame is None:
                time.sleep(0.01)
                continue

            with self.frame_lock:
                self.current_frame = frame.copy()
                
            frame_count += 1
            elapsed = time.time() - start_time
            if elapsed >= 1.0:
                self.current_fps = frame_count / elapsed
                frame_count = 0
                start_time = time.time()
                
            time.sleep(0.001)

    def predict(self, frame):
        """
        Run inference on a single frame synchronously.
        Returns (predictions, heatmap)
        """
        if frame is None:
            return [], None

        # Check if encoder is loaded
        encoder = None
        with self.model_lock:
            encoder = self.current_encoder
        
        if encoder is None:
            return [], None

        preds = []
        heatmap = None
        
        # Label updates
        if self.label_update_needed:
            with self.label_vecs_lock:
                self.current_label_vecs = None
            self.label_update_needed = False
            print(f"Labels updated: {self.labels}")

        encoder = None
        with self.model_lock:
            encoder = self.current_encoder

        if encoder is not None and self.labels:
            with self.inference_lock:
                now = time.time()
                
                # Compute label vecs
                need_compute = False
                with self.label_vecs_lock:
                    if self.current_label_vecs is None:
                        need_compute = True
                        
                if need_compute:
                    try:
                        if hasattr(encoder, 'encode_labels'):
                            # print("Encoding labels...")
                            vecs = encoder.encode_labels(self.labels)
                            with self.label_vecs_lock:
                                self.current_label_vecs = vecs
                            self.predictions_enabled = True
                            # print("Labels encoded.")
                        else:
                            # Encoder doesn't support labels (e.g. Yolo might handle differently)
                            pass
                    except Exception as e:
                        print(f"Failed to encode labels: {e}")
                
                # Predict
                try:
                    t0 = time.time()
                    local_vecs = None
                    with self.label_vecs_lock:
                        local_vecs = self.current_label_vecs
                        
                    if hasattr(encoder, 'predict'):
                        preds = encoder.predict(frame)
                    elif local_vecs is not None:
                        # Default cosine similarity
                        desired_res = getattr(encoder, 'input_resolution', (512, 512))
                        t = preprocess_frame(frame, input_resolution=desired_res).to(encoder.device)
                        with torch.no_grad():
                            vec = encoder.encode_image_to_vector(t)
                            sims = cosine_similarity_matrix(vec, local_vecs)
                            sims = sims.cpu().numpy()[0]
                        
                        pairs = list(zip(self.labels, sims.tolist()))
                        pairs.sort(key=lambda x: x[1], reverse=True)
                        preds = pairs[:5]
                        
                    self.last_inference_time = (time.time() - t0) * 1000
                    
                    # Heatmap logic (simplified)
                    if self.heatmap_enabled and preds and hasattr(encoder, 'compute_heatmap'):
                         # ... (Heatmap implementation similar to original)
                         # For now, let's assume compute_heatmap returns a heatmap image
                         # This part was missing implementation in original code too, 
                         # but let's try to call it if it exists.
                         try:
                             # Assuming compute_heatmap takes frame and text/label
                             # This is model specific.
                             # For now, just return None or placeholder if not fully implemented
                             pass
                         except Exception as e:
                             print(f"Heatmap computation failed: {e}")

                except Exception as e:
                    print(f"Inference failed: {e}")
                    preds = []

        return preds, heatmap

    def _inference_loop(self):
        while self.running:
            frame_to_process = None
            with self.frame_lock:
                if self.current_frame is not None:
                    frame_to_process = self.current_frame.copy()
            
            if frame_to_process is None:
                time.sleep(0.1)
                continue
            
            # If input type is browser_webcam, we don't run loop inference, 
            # we rely on explicit calls to predict() from web server.
            # But if we are in video mode, we might want this loop.
            # Let's check input type.
            input_type = self.input_config['type']
            if input_type == 'browser_webcam':
                time.sleep(0.5)
                continue

            preds, _ = self.predict(frame_to_process)
            self.current_pred = preds
            time.sleep(0.01)
