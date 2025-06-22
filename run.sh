#!/bin/bash
# Object Detection Service Runner
# This script sets up the Hailo environment and runs the detection service

echo "Starting Hailo Object Detection Service..."
echo "Monitoring: /home/jdneff/gatekeeper/captures"
echo "Press Ctrl+C to stop"
echo ""

# Source the Hailo environment and run our service
source /home/jdneff/Projects/hailo-rpi5-examples/setup_env.sh && python3 main.py
