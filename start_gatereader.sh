#!/bin/bash

# Gatereader startup script
echo "Starting Gatereader Object Detection API..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install/upgrade dependencies
echo "Installing dependencies..."
pip install -r gatereader_requirements.txt

# Check if HEF model exists
if [ ! -f "yolov5s.hef" ]; then
    echo "Warning: yolov5s.hef model file not found!"
    echo "Please place your YOLOv5 HEF model file in this directory."
    echo "You can download it from Hailo Model Zoo or convert your own model."
fi

# Start the server
echo "Starting Gatereader server on port 5001..."
python gatereader_server.py
