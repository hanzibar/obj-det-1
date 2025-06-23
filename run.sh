#!/bin/bash
# Object Detection Service Runner
# This script sets up the Hailo environment and runs the detection service

echo "Starting Hailo Object Detection Service..."
echo "Monitoring: /home/jdneff/gatekeeper/captures"
echo "Press Ctrl+C to stop"
echo ""

# Source the Hailo environment
source /home/jdneff/Projects/hailo-rpi5-examples/setup_env.sh

# Get the python path from the virtual environment
VENV_PYTHON=$(which python3)

# Run the object detection service with sudo, preserving environment and using venv python
echo "Starting object detection service with sudo..."
sudo -E env PATH="$PATH" "$VENV_PYTHON" main.py
