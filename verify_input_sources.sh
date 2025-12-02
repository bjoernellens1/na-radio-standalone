#!/bin/bash
set -e

# Create dummy images
mkdir -p test_images
convert -size 100x100 xc:red test_images/img1.jpg || touch test_images/img1.jpg
convert -size 100x100 xc:blue test_images/img2.jpg || touch test_images/img2.jpg

# Start server
source .venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python3 web/naradio_web.py --port 5005 &
PID=$!

echo "Server started with PID $PID"
sleep 10

# Test 1: Set to Image Folder
echo "Testing Image Folder..."
curl -X POST http://localhost:5005/set_input_source \
     -H "Content-Type: application/json" \
     -d "{\"type\": \"folder\", \"value\": \"$(pwd)/test_images\"}"

# Test 2: Set to Webcam (might fail if no webcam, but should return success or handled error)
echo "Testing Webcam..."
curl -X POST http://localhost:5005/set_input_source \
     -H "Content-Type: application/json" \
     -d "{\"type\": \"webcam\", \"value\": 0}"

# Test 3: Set to Invalid Video
echo "Testing Invalid Video..."
curl -X POST http://localhost:5005/set_input_source \
     -H "Content-Type: application/json" \
     -d "{\"type\": \"video\", \"value\": \"/non/existent/video.mp4\"}"

# Cleanup
kill $PID
rm -rf test_images
echo "Done"
