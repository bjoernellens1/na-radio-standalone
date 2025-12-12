from flask import Flask, render_template, Response, jsonify, request
import sys
import pathlib
import threading
import time
import cv2
import numpy as np
import os

proj_root = pathlib.Path(__file__).resolve().parent.parent
if str(proj_root) not in sys.path:
    sys.path.insert(0, str(proj_root))

from na_radio.manager import Manager

app = Flask(__name__, static_folder='static', template_folder='templates')

# Global Manager instance
manager = None

def get_manager():
    global manager
    if manager is None:
        # Initialize with defaults if not started via start_server
        manager = Manager()
        manager.start()
    return manager

@app.route('/')
def index():
    return render_template('index.html')

def gen_frames():
    mgr = get_manager()
    while True:
        frame = None
        with mgr.frame_lock:
            if mgr.current_frame is not None:
                frame = mgr.current_frame.copy()
        
        if frame is None:
            # Create black frame with text
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, 'No camera input', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Overlay heatmap if enabled (handled in Manager or here?)
        # Manager has logic for heatmap? 
        # In Manager._inference_loop, I added a comment # ... (Heatmap implementation similar to original)
        # I should probably implement heatmap overlay in Manager or expose heatmap data.
        # Manager has self.latest_heatmap.
        
        with mgr.heatmap_lock:
            hm = mgr.latest_heatmap
            
        if mgr.heatmap_enabled and hm is not None:
             # Resize hm if needed
            if hm.shape[:2] != frame.shape[:2]:
                hm = cv2.resize(hm, (frame.shape[1], frame.shape[0]))
            frame = cv2.addWeighted(frame, 0.6, hm, 0.4, 0)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.033)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/predictions')
def predictions():
    mgr = get_manager()
    return jsonify(mgr.current_pred)

@app.route('/status')
def status():
    mgr = get_manager()
    return jsonify(mgr.get_status())

@app.route('/toggle_heatmap', methods=['POST'])
def toggle_heatmap():
    mgr = get_manager()
    mgr.heatmap_enabled = not mgr.heatmap_enabled
    return jsonify({'heatmap_enabled': mgr.heatmap_enabled})

@app.route('/update_labels', methods=['POST'])
def update_labels():
    mgr = get_manager()
    data = request.json
    mode = data.get('mode', 'custom')
    
    new_labels = []
    if mode == 'imagenet':
        from web.vocabularies import IMAGENET_CLASSES
        new_labels = IMAGENET_CLASSES
    elif mode == 'coco':
        from web.vocabularies import COCO_CLASSES
        new_labels = COCO_CLASSES
    else:
        new_labels_str = data.get('labels', '')
        if not new_labels_str:
            return jsonify({'error': 'No labels provided'}), 400
        new_labels = [l.strip() for l in new_labels_str.split(',') if l.strip()]
        
    mgr.update_labels(new_labels)
    return jsonify({'success': True, 'labels': new_labels, 'mode': mode})

@app.route('/change_model', methods=['POST'])
def change_model():
    mgr = get_manager()
    data = request.json
    new_model = data.get('model')
    if not new_model:
        return jsonify({'error': 'Model name required'}), 400
        
    success = mgr.load_model(new_model)
    if success:
        return jsonify({'success': True, 'model': mgr.encoder_name})
    else:
        return jsonify({'error': 'Failed to load model'}), 500

@app.route('/set_input_source', methods=['POST'])
def set_input_source():
    mgr = get_manager()
    data = request.json
    src_type = data.get('type')
    value = data.get('value')
    
    if not src_type or value is None:
        return jsonify({'error': 'Type and value required'}), 400
        
    mgr.set_input(src_type, value)
    return jsonify({'success': True})

def start_server(host='0.0.0.0', port=5000, device_index=0, video_file=None,
                 labels_str=None, encoder_device=None, force_gpu=False, min_cc=7.0):
    
    labels = [l.strip() for l in (labels_str or 'person,car,dog,cat,tree').split(',') if l.strip()]
    
    global manager
    manager = Manager(
        device_index=device_index,
        video_file=video_file,
        labels=labels,
        encoder_name='radio', # Default
        device=encoder_device
    )
    manager.start()
    
    print(f'Starting web server on host {host}:{port}')
    try:
        app.run(host=host, port=port, threaded=True)
    except Exception as e:
        print(f"Failed to start server on {host}:{port}: {e}")
        if host != '0.0.0.0':
            print("Falling back to 0.0.0.0")
            app.run(host='0.0.0.0', port=port, threaded=True)
        else:
            raise e

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', default=5000, type=int)
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--video-file', default=None)
    parser.add_argument('--labels', default='person,car,dog,cat,tree')
    parser.add_argument('--encoder-device', default=None)
    
    args = parser.parse_args()
    
    start_server(
        host=args.host,
        port=args.port,
        device_index=args.device,
        video_file=args.video_file,
        labels_str=args.labels,
        encoder_device=args.encoder_device
    )
