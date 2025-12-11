import argparse
import time
import sys
import signal
from .manager import Manager

def main():
    parser = argparse.ArgumentParser(description="NA-Radio Standalone CLI")
    parser.add_argument("--input", type=str, default="0", help="Input source (0 for webcam, path for video/image)")
    parser.add_argument("--model", type=str, default="radio", help="Model name")
    parser.add_argument("--labels", type=str, default="person,car,dog,cat,tree", help="Comma separated labels")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--duration", type=int, default=0, help="Duration to run in seconds (0 for infinite)")
    
    args = parser.parse_args()
    
    # Parse input
    input_val = args.input
    try:
        input_val = int(input_val)
    except ValueError:
        pass # Keep as string
        
    labels = [l.strip() for l in args.labels.split(",") if l.strip()]
    
    manager = Manager(
        device_index=input_val if isinstance(input_val, int) else 0,
        video_file=input_val if isinstance(input_val, str) else None,
        labels=labels,
        encoder_name=args.model,
        device=args.device
    )
    
    def signal_handler(sig, frame):
        print("Exiting...")
        manager.stop()
        sys.exit(0)
        
    signal.signal(signal.SIGINT, signal_handler)
    
    print("Starting Manager...")
    manager.start()
    
    start_time = time.time()
    try:
        while True:
            status = manager.get_status()
            print(f"FPS: {status['fps']}, Inference: {status['inference_time_ms']}ms, Preds: {status['predictions']}", end='\r')
            
            if args.duration > 0 and (time.time() - start_time) > args.duration:
                break
                
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        manager.stop()

if __name__ == "__main__":
    main()
