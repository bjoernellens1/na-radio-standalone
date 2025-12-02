from flask import Flask, render_template, Response, jsonify, request
import sys
import pathlib
import threading
import time
import cv2
import numpy as np
import torch
import gc
import urllib.request
import tarfile
import zipfile
import pickle
import shutil
import tempfile


proj_root = pathlib.Path(__file__).resolve().parent.parent
if str(proj_root) not in sys.path:
    sys.path.insert(0, str(proj_root))

from naradio import load_encoder, preprocess_frame, cosine_similarity_matrix

app = Flask(__name__, static_folder='static', template_folder='templates')
import glob

# shared state across threads
frame_lock = threading.Lock()
current_frame = None
current_pred = []
# camera status
camera_open = False
# encoder info
encoder_name = None
current_encoder = None
model_lock = threading.Lock()
# predictions enabled flag (set when label vectors are available)
predictions_enabled = False
# monitoring
current_fps = 0.0
last_inference_time = 0.0
# dynamic labels
current_labels = []
label_update_needed = False
# resolution control
resolution_update_needed = False
pending_resolution = None
# object marking
heatmap_enabled = False
# async state
latest_heatmap = None
heatmap_lock = threading.Lock()
inference_lock = threading.Lock()
training_samples = [] # List of (feature_vector, label_string)
training_lock = threading.Lock()

# Input source configuration
input_config = {
    'type': 'webcam', # webcam, video, image, folder
    'value': 0,       # index or path
    'update_needed': False
}
input_lock = threading.Lock()

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
                return True, None # Return True but None frame to signal skip? 
                # Better: recurse or loop? Let's just return None and let loop handle
            return True, frame
        except Exception as e:
            print(f"Error reading {img_path}: {e}")
            return False, None

    def release(self):
        pass





def capture_loop(device_index=0, camera_file=None):
    global current_frame, camera_open, current_fps, input_config
    
    # Initialize config from args if not set
    with input_lock:
        if camera_file is not None:
            input_config['type'] = 'video'
            input_config['value'] = camera_file
        else:
            input_config['type'] = 'webcam'
            input_config['value'] = device_index
            
    cap = None
    
    def open_source():
        nonlocal cap
        src_type = input_config['type']
        val = input_config['value']
        
        if cap is not None:
            cap.release()
            
        print(f"Opening source: {src_type} = {val}")
        
        try:
            if src_type == 'webcam':
                # Try to cast to int
                try:
                    idx = int(val)
                except:
                    idx = 0
                cap = cv2.VideoCapture(idx)
            elif src_type == 'video':
                cap = cv2.VideoCapture(val)
            elif src_type == 'image' or src_type == 'folder':
                cap = ImageFolderCapture(val)
            else:
                print(f"Unknown source type: {src_type}")
                cap = None
        except Exception as e:
            print(f"Failed to open source: {e}")
            cap = None

        if cap is None or not cap.isOpened():
            print(f"Source not opened: {src_type} {val}")
            return False
        return True

    if not open_source():
        camera_open = False
    else:
        camera_open = True
        
    frame_count = 0
    start_time = time.time()
    
    while True:
        # Check for source update
        update = False
        with input_lock:
            if input_config['update_needed']:
                update = True
                input_config['update_needed'] = False
        
        if update:
            if open_source():
                camera_open = True
            else:
                camera_open = False
        
        if cap is None or not camera_open:
            time.sleep(0.5)
            # Try to reopen periodically if it failed?
            # For now just wait for user to change source
            continue

        ret, frame = cap.read()
        if not ret:
            # If video file, rewind
            if input_config['type'] == 'video':
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            elif input_config['type'] in ('image', 'folder'):
                # Should not happen with ImageFolderCapture unless empty
                time.sleep(0.1)
                continue
                
            # Webcam failure
            print("Webcam read failure")
            camera_open = False
            time.sleep(1.0)
            continue
            
        if frame is None:
             # Skip empty frames (e.g. from ImageFolderCapture error)
             time.sleep(0.01)
             continue

        with frame_lock:
            current_frame = frame.copy()
            
        frame_count += 1
        elapsed = time.time() - start_time
        if elapsed >= 1.0:
            current_fps = frame_count / elapsed
            frame_count = 0
            start_time = time.time()
            
        time.sleep(0.001) # Yield slightly to avoid hogging CPU


def inference_loop(label_vecs=None, labels=None, input_resolution=(512, 512)):
    global current_pred, predictions_enabled, last_inference_time, current_labels, label_update_needed, latest_heatmap, current_encoder
    
    # Use global current_labels if labels argument is not provided or to init
    global current_labels, label_update_needed
    if labels:
        current_labels = labels
    
    # Local state for the loop
    loop_labels = list(current_labels)
    label_vecs_local = label_vecs
    
    retry_interval = 5.0
    next_label_retry = 0.0
    
    while True:
        # Get latest frame
        frame_to_process = None
        with frame_lock:
            if current_frame is not None:
                frame_to_process = current_frame.copy()
        
        if frame_to_process is None:
            time.sleep(0.1)
            continue

        preds = []
        
        # Check for label updates
        if label_update_needed:
            loop_labels = list(current_labels)
            label_vecs_local = None # Force re-compute
            label_update_needed = False
            print(f"Labels updated in inference loop: {loop_labels}")

        # Check for resolution updates
        global resolution_update_needed, pending_resolution
        if resolution_update_needed and pending_resolution:
            input_resolution = pending_resolution
            if hasattr(encoder, 'input_resolution'):
                try:
                    encoder.input_resolution = input_resolution
                    print(f"Encoder resolution updated to {input_resolution}")
                except Exception as e:
                    print(f"Failed to update encoder resolution: {e}")
            else:
                print(f"Updated local resolution to {input_resolution} (encoder property not found)")
            
            resolution_update_needed = False
            # Optional: Clear label vectors if resolution change affects them (unlikely for RADIO/CLIP but safe)
            # label_vecs_local = None 


        encoder = None
        with model_lock:
            encoder = current_encoder

        if encoder is not None and loop_labels:
            with inference_lock:
                now = time.time()
                if label_vecs_local is None and now >= next_label_retry:
                    next_label_retry = now + retry_interval
                    try:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        
                        # Batched encoding to avoid OOM with large vocabularies
                        batch_size = 8
                        vecs_list = []
                        for i in range(0, len(loop_labels), batch_size):
                            batch = loop_labels[i:i+batch_size]
                            if i % 40 == 0:
                                print(f"Encoding batch {i}/{len(loop_labels)}")
                            
                            # Periodic cache clearing to avoid fragmentation
                            if i > 0 and i % 100 == 0 and torch.cuda.is_available():
                                 torch.cuda.empty_cache()
                                 
                            with torch.no_grad():
                                v = encoder.encode_labels(batch)
                                vecs_list.append(v.cpu()) # Move to CPU temporarily to save GPU memory
                            
                            # Yield control to allow other threads/processes to run
                            time.sleep(0.02)
                        
                        # Concatenate and move back to GPU
                        label_vecs_local = torch.cat(vecs_list, dim=0).to(encoder.device)
                        
                        predictions_enabled = True
                        print(f'Precomputed {len(loop_labels)} label vectors successfully')
                    except Exception as e:
                        print(f'Failed to precompute label vectors; retrying in {int(retry_interval)}s', repr(e))
                        label_vecs_local = None
                
                if label_vecs_local is not None or (hasattr(encoder, 'predict_custom') and hasattr(encoder, 'custom_head') and encoder.custom_head is not None) or hasattr(encoder, 'predict'):
                    try:
                        t0 = time.time()
                        if hasattr(encoder, 'predict_custom') and hasattr(encoder, 'custom_head') and encoder.custom_head is not None:
                             # Use custom head for prediction
                             desired_res = getattr(encoder, 'input_resolution', input_resolution)
                             t = preprocess_frame(frame_to_process, input_resolution=desired_res).to(encoder.device)
                             with torch.no_grad():
                                 vec = encoder.encode_image_to_vector(t)
                             preds = encoder.predict_custom(vec)
                        elif hasattr(encoder, 'predict'):
                             # Use encoder's native predict method (e.g. Yolo)
                             preds = encoder.predict(frame_to_process)
                        else:
                             preds = _compute_predictions(frame_to_process, encoder, label_vecs_local, loop_labels, input_resolution)


                        last_inference_time = (time.time() - t0) * 1000  # ms
                        
                        # Compute heatmap for top prediction if enabled
                        new_heatmap = None
                        if heatmap_enabled and preds and hasattr(encoder, 'compute_heatmap'):
                            top_label = preds[0][0]
                            try:
                                top_vec = None
                                if top_label in loop_labels:
                                    idx = loop_labels.index(top_label)
                                    top_vec = label_vecs_local[idx].unsqueeze(0)
                                elif hasattr(encoder, 'encode_labels'):
                                    # Try to encode on the fly (e.g. for DINOv2 custom labels not in UI list)
                                    try:
                                        top_vec = encoder.encode_labels([top_label])
                                    except:
                                        pass

                                if hasattr(encoder, 'compute_visualization'):
                                    # Use encoder's custom visualization (e.g. Yolo bounding boxes)
                                    # Pass original frame (numpy)
                                    vis = encoder.compute_visualization(frame_to_process)
                                    new_heatmap = (vis, 'replace')
                                    
                                elif top_vec is not None:
                                    # Preprocess frame again (could optimize this)
                                    desired_res = getattr(encoder, 'input_resolution', input_resolution)
                                    t = preprocess_frame(frame_to_process, input_resolution=desired_res).to(encoder.device)
                                    
                                    hm = encoder.compute_heatmap(t, top_vec)
                                    hm_np = hm.detach().cpu().numpy()
                                    
                                    # Overlay heatmap
                                    hm_uint8 = (hm_np * 255).astype(np.uint8)
                                    heatmap_img = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)
                                    
                                    # Resize heatmap to frame size if needed
                                    if heatmap_img.shape[:2] != frame_to_process.shape[:2]:
                                        heatmap_img = cv2.resize(heatmap_img, (frame_to_process.shape[1], frame_to_process.shape[0]))
                                    
                                    new_heatmap = (heatmap_img, 'blend')


                                    
                            except NotImplementedError:
                                # Encoder doesn't support heatmap (e.g. Yolo)
                                pass
                            except Exception as e:
                                # Only print if it's not a "not in list" error which we handled
                                print("Heatmap error:", e)

                        
                        with heatmap_lock:
                            latest_heatmap = new_heatmap
                                
                    except Exception as e:
                        import traceback
                        print('Prediction error:', repr(e))
                        traceback.print_exc()
                        preds = []


        with frame_lock: # Reuse frame_lock for preds to keep it simple, or could use a new lock
            current_pred = preds

        time.sleep(0.01) # Don't spin too fast if inference is super fast


def _compute_predictions(frame, encoder, label_vecs, labels, input_resolution):
    """Preprocess frame and compute similarities. Separated for readability."""
    # Choose preprocessing based on encoder capabilities
    if hasattr(encoder, 'clip_preprocess') and getattr(encoder, 'clip_preprocess') is not None:
        from PIL import Image
        pil = Image.fromarray(frame[:, :, ::-1])
        t = encoder.clip_preprocess(pil).unsqueeze(0).to(encoder.device)
    else:
        desired_res = getattr(encoder, 'input_resolution', input_resolution)
        t = preprocess_frame(frame, input_resolution=desired_res).to(encoder.device)
    with torch.no_grad():
        vec = encoder.encode_image_to_vector(t)
        if vec.dim() == 1:
            vec = vec.unsqueeze(0)
        sims = cosine_similarity_matrix(vec, label_vecs)
        sims = sims.cpu().numpy()[0]
    pairs = list(zip(labels, sims.tolist()))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs[:5]


def gen_frames():
    global current_frame, latest_heatmap
    while True:
        img = None
        hm = None
        
        with frame_lock:
            if current_frame is None:
                img = np.zeros((480, 640, 3), dtype=np.uint8)
                try:
                    import cv2 as _cv2
                    if not camera_open:
                        _cv2.putText(img, 'No camera device detected. Did you pass --device /dev/video0?', (10, 30), _cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                except Exception:
                    pass
            else:
                img = current_frame.copy()
        
        with heatmap_lock:
            if latest_heatmap is not None:
                if isinstance(latest_heatmap, tuple):
                    hm = (latest_heatmap[0].copy(), latest_heatmap[1])
                else:
                    hm = latest_heatmap.copy()

        
        # Blend if heatmap exists and is enabled
        if heatmap_enabled and hm is not None:
            # Check if hm is tuple (img, mode)
            mode = 'blend'
            if isinstance(hm, tuple):
                hm_img, mode = hm
            else:
                hm_img = hm

             # Resize hm if needed (should match img)
            if hm_img.shape[:2] != img.shape[:2]:
                hm_img = cv2.resize(hm_img, (img.shape[1], img.shape[0]))
            
            if mode == 'replace':
                img = hm_img
            else:
                img = cv2.addWeighted(img, 0.6, hm_img, 0.4, 0)


        ret, buffer = cv2.imencode('.jpg', img)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.033) # ~30 FPS display loop


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/predictions')
def predictions():
    global current_pred
    return jsonify(current_pred)


@app.route('/status')
def status():
    return jsonify({
        'camera_open': camera_open,
        'encoder': encoder_name,
        'predictions_enabled': predictions_enabled,
        'fps': round(current_fps, 1),
        'inference_time_ms': round(last_inference_time, 1),
        'heatmap_enabled': heatmap_enabled,
        'memory_usage': get_memory_usage()
    })

def get_memory_usage():
    import psutil
    import os
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024 # MB


@app.route('/toggle_heatmap', methods=['POST'])
def toggle_heatmap():
    global heatmap_enabled
    heatmap_enabled = not heatmap_enabled
    return jsonify({'heatmap_enabled': heatmap_enabled})


from web.vocabularies import IMAGENET_CLASSES, COCO_CLASSES

@app.route('/update_labels', methods=['POST'])
def update_labels():
    global predictions_enabled, current_labels, label_update_needed
    data = request.json
    mode = data.get('mode', 'custom')
    
    new_labels = []
    if mode == 'imagenet':
        new_labels = IMAGENET_CLASSES
    elif mode == 'coco':
        new_labels = COCO_CLASSES
    else:
        # Custom mode
        new_labels_str = data.get('labels', '')
        if not new_labels_str:
            return jsonify({'error': 'No labels provided'}), 400
        new_labels = [l.strip() for l in new_labels_str.split(',') if l.strip()]
        
    if not new_labels:
        return jsonify({'error': 'Invalid labels'}), 400

    # Update global state
    current_labels = new_labels
    label_update_needed = True
    predictions_enabled = False # Disable until re-computed
    
    return jsonify({'success': True, 'labels': new_labels, 'mode': mode})

@app.route('/reset_inference', methods=['POST'])
def reset_inference():
    global predictions_enabled, label_update_needed, current_labels
    
    print("Resetting inference state...")
    predictions_enabled = False
    
    # Force label update to trigger re-encoding
    label_update_needed = True
    
    # Clear CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("CUDA cache cleared")
        
    return jsonify({'success': True})

@app.route('/update_resolution', methods=['POST'])
def update_resolution():
    global resolution_update_needed, pending_resolution
    data = request.json
    width = data.get('width')
    height = data.get('height')
    
    if not width or not height:
        return jsonify({'error': 'Width and height required'}), 400
        
    # Set pending resolution (H, W) for encoder
    pending_resolution = (int(height), int(width))
    resolution_update_needed = True
    
    return jsonify({'success': True, 'resolution': pending_resolution})


@app.route('/change_model', methods=['POST'])
def change_model():
    global current_encoder, encoder_name, predictions_enabled, label_update_needed
    data = request.json
    new_model = data.get('model')
    
    if not new_model:
        return jsonify({'error': 'Model name required'}), 400
        
    print(f"Switching to model: {new_model}")
    
    # Stop inference temporarily? The loop checks for encoder is None but we replace it atomically-ish
    # But we should probably pause it.
    
    try:
        old_encoder = None
        with model_lock:
            # Unload old model if possible (Python GC should handle it if we drop reference)
            old_encoder = current_encoder
            current_encoder = None
        
        # Wait for any ongoing inference to finish
        with inference_lock:
            pass

        if old_encoder is not None:
            print("Unloading old encoder...")
            if hasattr(old_encoder, 'unload'):
                old_encoder.unload()
            del old_encoder
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Load new model
        # We need to know device, etc. We'll reuse global args or defaults
        # For simplicity, use defaults or what was passed to start_server (we need to store them)
        # We'll assume default device for now or store it.
        # Let's use get_device()
        from utils import get_device
        device = get_device()
        
        enc, name = load_encoder(preferred=new_model, device=device)
        
        with model_lock:
            current_encoder = enc
            encoder_name = name
        
        # Trigger label re-encoding
        predictions_enabled = False
        label_update_needed = True
            
        return jsonify({'success': True, 'model': encoder_name})

    except Exception as e:
        print(f"Failed to switch model: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/enable_openvino', methods=['POST'])
def enable_openvino():
    global current_encoder, encoder_name, predictions_enabled, label_update_needed
    
    with model_lock:
        if current_encoder is None:
            return jsonify({'error': 'No encoder loaded'}), 400
            
        if hasattr(current_encoder, 'convert_to_openvino'):
            print("Attempting to switch to OpenVINO backend...")
            
            # Pause inference?
            with inference_lock:
                 ov_encoder = current_encoder.convert_to_openvino()
                 
            if ov_encoder is not None:
                # Replace current encoder
                current_encoder = ov_encoder
                encoder_name = f"{encoder_name} (OpenVINO)"
                
                # Trigger label re-encoding (OpenVINO wrapper handles it via original encoder, but good to refresh)
                predictions_enabled = False
                label_update_needed = True
                
                return jsonify({'success': True, 'model': encoder_name})
            else:
                return jsonify({'error': 'OpenVINO conversion failed'}), 500
        else:
             return jsonify({'error': 'Current encoder does not support OpenVINO'}), 400


@app.route('/capture_sample', methods=['POST'])
def capture_sample():
    global training_samples
    data = request.json
    label = data.get('label')
    if not label:
        return jsonify({'error': 'Label required'}), 400
    
    with frame_lock:
        if current_frame is None:
             return jsonify({'error': 'No frame available'}), 400
        frame = current_frame.copy()
        
    with model_lock:
        if current_encoder is None:
             return jsonify({'error': 'No encoder loaded'}), 400
        
        # Encode frame
        try:
            # Preprocess
            desired_res = getattr(current_encoder, 'input_resolution', (512, 512))
            if hasattr(current_encoder, 'clip_preprocess') and getattr(current_encoder, 'clip_preprocess') is not None:
                from PIL import Image
                pil = Image.fromarray(frame[:, :, ::-1])
                t = current_encoder.clip_preprocess(pil).unsqueeze(0).to(current_encoder.device)
            else:
                t = preprocess_frame(frame, input_resolution=desired_res).to(current_encoder.device)
            
            with torch.no_grad():
                vec = current_encoder.encode_image_to_vector(t)
                # Keep on CPU to save VRAM
                vec = vec.cpu()
                
            with training_lock:
                training_samples.append((vec, label))
                count = len(training_samples)
                
            return jsonify({'success': True, 'samples_count': count})
        except Exception as e:
            print(f"Capture failed: {e}")
            return jsonify({'error': str(e)}), 500

@app.route('/train_classifier', methods=['POST'])
def train_classifier():
    global training_samples
    with model_lock:
        if current_encoder is None:
             return jsonify({'error': 'No encoder loaded'}), 400
        if not hasattr(current_encoder, 'train_custom_head'):
             return jsonify({'error': 'Current encoder does not support custom training'}), 400
        
        with training_lock:
            if not training_samples:
                 return jsonify({'error': 'No samples captured'}), 400
            
            features = torch.cat([s[0] for s in training_samples], dim=0).to(current_encoder.device)
            labels = [s[1] for s in training_samples]
            unique_labels = sorted(list(set(labels)))
            
            try:
                current_encoder.train_custom_head(features, labels, unique_labels)
                return jsonify({'success': True, 'classes': unique_labels})
            except Exception as e:
                print(f"Training failed: {e}")
                return jsonify({'error': str(e)}), 500

@app.route('/reset_classifier', methods=['POST'])
def reset_classifier():
    global training_samples
    with training_lock:
        training_samples = []
    
    with model_lock:
        if current_encoder is not None and hasattr(current_encoder, 'reset_custom_head'):
            current_encoder.reset_custom_head()
            
    return jsonify({'success': True})

@app.route('/load_dataset', methods=['POST'])
def load_dataset():
    global training_samples
    data = request.json
    dataset_path = data.get('path')
    if not dataset_path:
        return jsonify({'error': 'Path required'}), 400
    
    if not os.path.isdir(dataset_path):
        return jsonify({'error': 'Directory not found'}), 400

    with model_lock:
        if current_encoder is None:
             return jsonify({'error': 'No encoder loaded'}), 400
        
        count = 0
        classes_found = set()
        
        try:
            # Walk through directory
            for root, dirs, files in os.walk(dataset_path):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                        # Infer label from parent folder name
                        label = os.path.basename(root)
                        if label == os.path.basename(dataset_path):
                            # Image in root folder, skip or use 'unknown'?
                            continue
                            
                        img_path = os.path.join(root, file)
                        
                        try:
                            # Load and preprocess
                            from PIL import Image
                            img = Image.open(img_path).convert('RGB')
                            
                            desired_res = getattr(current_encoder, 'input_resolution', (512, 512))
                            if hasattr(current_encoder, 'clip_preprocess') and getattr(current_encoder, 'clip_preprocess') is not None:
                                t = current_encoder.clip_preprocess(img).unsqueeze(0).to(current_encoder.device)
                            else:
                                # Convert to tensor manually if needed, but preprocess_frame expects numpy BGR usually?
                                # Let's use preprocess_frame logic adapted for PIL or just convert PIL to numpy
                                import numpy as np
                                frame = np.array(img)[:, :, ::-1].copy() # RGB to BGR
                                t = preprocess_frame(frame, input_resolution=desired_res).to(current_encoder.device)

                            with torch.no_grad():
                                vec = current_encoder.encode_image_to_vector(t)
                                vec = vec.cpu()
                            
                            with training_lock:
                                training_samples.append((vec, label))
                                classes_found.add(label)
                                count += 1
                                
                        except Exception as e:
                            print(f"Failed to load {img_path}: {e}")
                            
            return jsonify({'success': True, 'count': count, 'classes': list(classes_found)})
            
        except Exception as e:
            print(f"Dataset load failed: {e}")
            return jsonify({'error': str(e)}), 500




@app.route('/download_dataset', methods=['POST'])
def download_dataset():
    global training_samples
    data = request.json
    ds_type = data.get('type')
    url = data.get('url')
    
    if ds_type == 'cifar10':
        url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    elif ds_type == 'url':
        if not url:
             return jsonify({'error': 'URL required'}), 400
    else:
        return jsonify({'error': 'Invalid type'}), 400

    with model_lock:
        if current_encoder is None:
             return jsonify({'error': 'No encoder loaded'}), 400
        
        # Create temp dir
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                print(f"Downloading {url}...")
                file_name = url.split('/')[-1]
                file_path = os.path.join(temp_dir, file_name)
                
                # Download
                urllib.request.urlretrieve(url, file_path)
                
                count = 0
                classes_found = set()
                
                # Extract and Process
                if ds_type == 'cifar10':
                    with tarfile.open(file_path, 'r:gz') as tar:
                        tar.extractall(path=temp_dir)
                    
                    cifar_dir = os.path.join(temp_dir, 'cifar-10-batches-py')
                    
                    # Load meta
                    with open(os.path.join(cifar_dir, 'batches.meta'), 'rb') as f:
                        meta = pickle.load(f, encoding='bytes')
                    label_names = [x.decode('utf-8') for x in meta[b'label_names']]
                    
                    # Load batches
                    batches = [f'data_batch_{i}' for i in range(1, 6)] + ['test_batch']
                    
                    for b_name in batches:
                        b_path = os.path.join(cifar_dir, b_name)
                        with open(b_path, 'rb') as f:
                            batch = pickle.load(f, encoding='bytes')
                        
                        images = batch[b'data']
                        labels = batch[b'labels']
                        
                        # Reshape images: (N, 3072) -> (N, 3, 32, 32) -> (N, 32, 32, 3)
                        images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
                        
                        for i in range(len(images)):
                            img = images[i] # RGB numpy array
                            label = label_names[labels[i]]
                            
                            # Preprocess
                            desired_res = getattr(current_encoder, 'input_resolution', (512, 512))
                            # CIFAR images are small (32x32), upscaling might be needed or handled by preprocess
                            # preprocess_frame handles resizing
                            # img is RGB, preprocess_frame expects BGR if using cv2 logic?
                            # preprocess_frame in naradio.py:
                            # def preprocess_frame(frame, input_resolution=(512, 512)):
                            #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) ...
                            # So it expects BGR.
                            
                            # Convert RGB to BGR for consistency with preprocess_frame
                            frame_bgr = img[:, :, ::-1].copy()
                            
                            t = preprocess_frame(frame_bgr, input_resolution=desired_res).to(current_encoder.device)
                            
                            with torch.no_grad():
                                vec = current_encoder.encode_image_to_vector(t)
                                vec = vec.cpu()
                            
                            with training_lock:
                                training_samples.append((vec, label))
                                classes_found.add(label)
                                count += 1
                                
                            if count % 100 == 0:
                                print(f"Processed {count} images...")
                                
                else: # Generic Zip/Tar
                    if file_name.endswith('.zip'):
                        with zipfile.ZipFile(file_path, 'r') as zip_ref:
                            zip_ref.extractall(temp_dir)
                    elif file_name.endswith(('.tar.gz', '.tgz')):
                        with tarfile.open(file_path, 'r:gz') as tar:
                            tar.extractall(path=temp_dir)
                    
                    # Walk and load like load_dataset
                    for root, dirs, files in os.walk(temp_dir):
                        for file in files:
                            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                                # Infer label from parent folder name
                                label = os.path.basename(root)
                                # Skip if label is temp_dir name or extracted folder root if unstructured
                                # Heuristic: if parent is temp_dir, maybe use 'unknown'?
                                # Let's assume structured dataset
                                
                                img_path = os.path.join(root, file)
                                try:
                                    from PIL import Image
                                    img = Image.open(img_path).convert('RGB')
                                    
                                    desired_res = getattr(current_encoder, 'input_resolution', (512, 512))
                                    if hasattr(current_encoder, 'clip_preprocess') and getattr(current_encoder, 'clip_preprocess') is not None:
                                        t = current_encoder.clip_preprocess(img).unsqueeze(0).to(current_encoder.device)
                                    else:
                                        import numpy as np
                                        frame = np.array(img)[:, :, ::-1].copy() # RGB to BGR
                                        t = preprocess_frame(frame, input_resolution=desired_res).to(current_encoder.device)

                                    with torch.no_grad():
                                        vec = current_encoder.encode_image_to_vector(t)
                                        vec = vec.cpu()
                                    
                                    with training_lock:
                                        training_samples.append((vec, label))
                                        classes_found.add(label)
                                        count += 1
                                except Exception as e:
                                    pass

                return jsonify({'success': True, 'count': count, 'classes': list(classes_found)})

            except Exception as e:
                print(f"Download/Process failed: {e}")
                return jsonify({'error': str(e)}), 500


@app.route('/set_input_source', methods=['POST'])
def set_input_source():
    global input_config
    data = request.json
    src_type = data.get('type')
    value = data.get('value')
    
    if not src_type or value is None:
        return jsonify({'error': 'Type and value required'}), 400
        
    print(f"Request to set input source: {src_type} = {value}")
    
    # Validate
    if src_type == 'webcam':
        try:
            int(value)
        except:
            return jsonify({'error': 'Webcam index must be an integer'}), 400
    elif src_type == 'video':
        # Check if file exists or is URL
        if not (str(value).startswith('http') or os.path.exists(value)):
             return jsonify({'error': 'Video file not found'}), 400
    elif src_type == 'folder' or src_type == 'image':
        if not os.path.exists(value):
             return jsonify({'error': 'Path not found'}), 400
             
    with input_lock:
        input_config['type'] = src_type
        input_config['value'] = value
        input_config['update_needed'] = True
        
    return jsonify({'success': True})


def start_server(host='0.0.0.0', port=5000, device_index=0, video_file=None,
                 labels_str=None, encoder_device=None, force_gpu=False, min_cc=7.0):
    # parse labels

    labels = [l.strip() for l in (labels_str or 'person,car,dog,cat,tree').split(',') if l.strip()]
    # load encoder
    device_opt = None if (encoder_device is None or encoder_device == '') else encoder_device
    enc, name = load_encoder(preferred='radio',
                             device=device_opt,
                             input_resolution=(512, 512),
                             force_gpu=force_gpu,
                             min_cc=min_cc)
    global current_encoder, encoder_name
    current_encoder = enc
    encoder_name = name
    # precompute label vectors once
    try:
        label_vecs = enc.encode_labels(labels).to(enc.device)
        global predictions_enabled
        predictions_enabled = True
    except Exception as e:
        print('Failed to encode labels; will attempt to compute per-frame but this is expensive:', repr(e))
        label_vecs = None
        predictions_enabled = False
    # start capture thread
    capture_thr = threading.Thread(target=capture_loop, args=(device_index, video_file), daemon=True)
    capture_thr.start()
    
    # start inference thread
    inference_thr = threading.Thread(target=inference_loop, args=(label_vecs, labels, (512,512)), daemon=True)
    inference_thr.start()
    
    print(f'Starting web server with encoder: {name} on host {host}:{port}')
    app.run(host=host, port=port, threaded=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', default=5000, type=int)
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--video-file', default=None)
    parser.add_argument('--labels', default='person,car,dog,cat,tree')
    parser.add_argument('--encoder-device', default=None, help="Force encoder device (e.g., 'cuda' or 'cpu')")
    parser.add_argument('--force-gpu', action='store_true', help='Force GPU encoder even if compute capability checks fail')
    parser.add_argument('--min-cc', default=7.0, type=float, help='Minimum GPU compute capability required for automatic GPU use')
    args = parser.parse_args()
    import os
    # Allow overriding via environment variables for Docker/Compose setups
    env_host = os.environ.get('HOST')
    env_port = os.environ.get('PORT')
    env_device = os.environ.get('DEVICE')
    env_video_file = os.environ.get('VIDEO_FILE')
    env_labels = os.environ.get('LABELS')
    host = env_host or args.host
    port = int(env_port) if env_port else args.port
    device_idx = int(env_device) if env_device else args.device
    video_file = env_video_file or args.video_file
    labels = env_labels or args.labels
    env_encoder_device = os.environ.get('ENCODER_DEVICE')
    encoder_device = env_encoder_device if env_encoder_device is not None else args.encoder_device
    env_force_gpu = os.environ.get('FORCE_GPU')
    force_gpu = args.force_gpu
    if env_force_gpu is not None:
        force_gpu = env_force_gpu.lower() in ('1', 'true', 'yes', 'on')
    env_min_cc = os.environ.get('MIN_CC')
    min_cc = float(env_min_cc) if env_min_cc else args.min_cc
    start_server(host=host,
                 port=port,
                 device_index=device_idx,
                 video_file=video_file,
                 labels_str=labels,
                 encoder_device=encoder_device,
                 force_gpu=force_gpu,
                 min_cc=min_cc)
