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


def camera_loop(device_index=0, input_resolution=(512, 512), camera_file=None, encoder=None, label_vecs=None, labels=None, interval=0.1):
    global current_frame, current_pred, predictions_enabled
    cap = None
    if camera_file is None:
        cap = cv2.VideoCapture(device_index)
    else:
        cap = cv2.VideoCapture(camera_file)
    global camera_open
    if not cap.isOpened():
        print(f"Camera device {device_index} not opened; cap.isOpened() == False. camera_file={camera_file}")
        camera_open = False
    else:
        camera_open = True
    label_vecs_local = label_vecs
    retry_interval = 5.0  # seconds between encode-label attempts when they fail
    next_label_retry = 0.0
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue
        # Update latest frame outside of expensive computation
        with frame_lock:
            current_frame = frame.copy()

        preds = []
        if encoder is not None and labels:
            now = time.time()
            if label_vecs_local is None and now >= next_label_retry:
                next_label_retry = now + retry_interval
                try:
                    label_vecs_local = encoder.encode_labels(labels).to(encoder.device)
                    predictions_enabled = True
                    print('Precomputed label vectors successfully')
                except Exception as e:
                    print(f'Failed to precompute label vectors; retrying in {int(retry_interval)}s', repr(e))
                    label_vecs_local = None
            if label_vecs_local is not None:
                try:
                    preds = _compute_predictions(frame, encoder, label_vecs_local, labels, input_resolution)
                except Exception as e:
                    import traceback
                    print('Prediction error:', repr(e))
                    traceback.print_exc()
                    preds = []

        with frame_lock:
            current_pred = preds

        time.sleep(interval)


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
    global current_frame
    while True:
        with frame_lock:
            if current_frame is None:
                img = np.zeros((480, 640, 3), dtype=np.uint8)
                # Overlay helpful message if camera is not connected
                try:
                    import cv2 as _cv2
                    if not camera_open:
                        _cv2.putText(img, 'No camera device detected. Did you pass --device /dev/video0?', (10, 30), _cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                except Exception:
                    pass
            else:
                img = current_frame.copy()
        ret, buffer = cv2.imencode('.jpg', img)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.05)


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
    })


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
    # start camera thread
    thr = threading.Thread(target=camera_loop, args=(device_index, (512,512), video_file, enc, label_vecs, labels), daemon=True)
    thr.start()
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
