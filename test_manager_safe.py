import sys
import os
import time
import threading

# Add project root to path
sys.path.insert(0, os.getcwd())

from na_radio.manager import Manager

def test_manager_safe():
    print("Initializing Manager with non-existent device 99...")
    # Initialize manager with a device index that is unlikely to exist
    mgr = Manager(device_index=99)
    
    print("Starting Manager...")
    mgr.start()
    
    # Give it some time to try opening the camera
    time.sleep(2)
    
    status = mgr.get_status()
    print(f"Manager status: {status}")
    
    if status['camera_open']:
        print("FAILURE: Camera reported as open, but it should not be.")
    else:
        print("SUCCESS: Camera reported as closed, as expected.")
        
    print("Stopping Manager...")
    mgr.stop()
    print("Manager stopped.")

if __name__ == "__main__":
    try:
        test_manager_safe()
    except Exception as e:
        print(f"Caught exception during test: {e}")
        sys.exit(1)
