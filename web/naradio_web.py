from flask import Flask, render_template, Response, jsonify, request
import sys
import pathlib
import threading
import time
import cv2
import numpy as np
import torch

proj_root = pathlib.Path(__file__).resolve().parent.parent
if str(proj_root) not in sys.path:
    sys.path.insert(0, str(proj_root))

from naradio import load_encoder, preprocess_frame, cosine_similarity_matrix

app = Flask(__name__, static_folder='static', template_folder='templates')

# shared state across threads
frame_lock = threading.Lock()
current_frame = None
current_pred = []
# camera status
camera_open = False
# encoder info
encoder_name = None
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


def capture_loop(device_index=0, camera_file=None):
    global current_frame, camera_open, current_fps
    cap = None
    if camera_file is None:
        cap = cv2.VideoCapture(device_index)
    else:
        cap = cv2.VideoCapture(camera_file)
    
    if not cap.isOpened():
        print(f"Camera device {device_index} not opened; cap.isOpened() == False. camera_file={camera_file}")
        camera_open = False
        return
    else:
        camera_open = True
        
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            # If video file, rewind
            if camera_file is not None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            time.sleep(0.1)
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


def inference_loop(encoder=None, label_vecs=None, labels=None, input_resolution=(512, 512)):
    global current_pred, predictions_enabled, last_inference_time, current_labels, label_update_needed, latest_heatmap
    
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


        if encoder is not None and loop_labels:
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
            
            if label_vecs_local is not None:
                try:
                    t0 = time.time()
                    preds = _compute_predictions(frame_to_process, encoder, label_vecs_local, loop_labels, input_resolution)
                    last_inference_time = (time.time() - t0) * 1000  # ms
                    
                    # Compute heatmap for top prediction if enabled
                    new_heatmap = None
                    if heatmap_enabled and preds and hasattr(encoder, 'compute_heatmap'):
                        top_label = preds[0][0]
                        try:
                            # Find index of top label
                            idx = loop_labels.index(top_label)
                            top_vec = label_vecs_local[idx].unsqueeze(0)
                            
                            # Preprocess frame again (could optimize this)
                            desired_res = getattr(encoder, 'input_resolution', input_resolution)
                            t = preprocess_frame(frame_to_process, input_resolution=desired_res).to(encoder.device)
                            
                            hm = encoder.compute_heatmap(t, top_vec)
                            hm_np = hm.cpu().numpy()
                            
                            # Overlay heatmap
                            hm_uint8 = (hm_np * 255).astype(np.uint8)
                            heatmap_img = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)
                            
                            # Resize heatmap to frame size if needed
                            if heatmap_img.shape[:2] != frame_to_process.shape[:2]:
                                heatmap_img = cv2.resize(heatmap_img, (frame_to_process.shape[1], frame_to_process.shape[0]))
                            
                            new_heatmap = heatmap_img
                                
                        except Exception as e:
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
                hm = latest_heatmap.copy()
        
        # Blend if heatmap exists and is enabled
        if heatmap_enabled and hm is not None:
             # Resize hm if needed (should match img)
            if hm.shape[:2] != img.shape[:2]:
                hm = cv2.resize(hm, (img.shape[1], img.shape[0]))
            img = cv2.addWeighted(img, 0.6, hm, 0.4, 0)

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
        'heatmap_enabled': heatmap_enabled
    })


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
    global encoder_name
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
    inference_thr = threading.Thread(target=inference_loop, args=(enc, label_vecs, labels, (512,512)), daemon=True)
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
