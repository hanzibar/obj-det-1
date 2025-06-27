#!/usr/bin/env python3
"""
Hailo AI Object Detection Service

A systemctl service that monitors ~/Projects/gatekeeper/captures for new images,
processes them with object detection, updates JSON metadata, and generates
annotated images with bounding boxes.

Features:
- Initial processing of all existing images on startup
- Continuous monitoring for new images using inotify
- JSON metadata updates with detection results
- Annotated image generation
- Robust error handling and logging
- Systemctl service integration
"""

import os
import sys
import time
import json
import signal
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/jdneff/Projects/obj-det-1/hailo-detection-service.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('HailoDetectionService')

class HailoDetectionService:
    """Main service class for Hailo AI object detection monitoring."""
    
    def __init__(self):
        self.base_dir = Path('/home/jdneff/Projects/obj-det-1')
        self.monitor_dir = Path('/home/jdneff/Projects/gatekeeper/captures')
        self.model_path = self.base_dir / 'models/yolov8s_h8l.hef'
        self.labels_path = self.base_dir / 'coco.txt'
        self.detection_script = self.base_dir / 'object_detection.py'
        
        self.observer = None
        self.running = False
        
        # Ensure directories exist
        self.monitor_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Hailo Detection Service initialized")
        logger.info(f"Monitoring directory: {self.monitor_dir}")
        logger.info(f"Detection script: {self.detection_script}")
    
    def should_process_image(self, image_path: Path) -> bool:
        """Check if an image should be processed."""
        # Skip if it's a detected/annotated image (ends with _detected)
        if '_detected' in image_path.stem:
            return False
        
        # Skip if not a valid image extension
        if not image_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            return False
        
        # Check if already processed (detected version exists)
        detected_version = image_path.parent / f"{image_path.stem}_detected.jpg"
        if detected_version.exists():
            logger.debug(f"Skipping {image_path.name} - already processed")
            return False
        
        return True

    def process_image(self, image_path: Path) -> bool:
        """Process a single image with object detection."""
        try:
            if not image_path.exists():
                return False
            
            # Check if we should process this image
            if not self.should_process_image(image_path):
                return False
            
            logger.info(f"Processing image: {image_path.name}")
            
            # Run object detection on single image
            cmd = [
                'python3', str(self.detection_script),
                '--folder-mode',
                '-n', str(self.model_path),
                '-i', str(image_path.parent),
                '-l', str(self.labels_path)
            ]
            
            # Create a temporary folder with just this image for processing
            temp_dir = self.base_dir / 'temp_processing'
            temp_dir.mkdir(exist_ok=True)
            
            # Copy image and JSON to temp directory
            temp_image = temp_dir / image_path.name
            temp_json = temp_dir / image_path.with_suffix('.json').name
            
            # Copy files
            subprocess.run(['cp', str(image_path), str(temp_image)], check=True)
            if image_path.with_suffix('.json').exists():
                subprocess.run(['cp', str(image_path.with_suffix('.json')), str(temp_json)], check=True)
            
            # Run detection on temp directory
            cmd = [
                'python3', str(self.detection_script),
                '--folder-mode',
                '-n', str(self.model_path),
                '-i', str(temp_dir),
                '-l', str(self.labels_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                # Copy results back
                if temp_json.exists():
                    subprocess.run(['cp', str(temp_json), str(image_path.with_suffix('.json'))], check=True)
                
                # Copy annotated image back
                temp_detected = temp_dir / f"{image_path.stem}_detected.jpg"
                if temp_detected.exists():
                    detected_image = image_path.parent / f"{image_path.stem}_detected.jpg"
                    subprocess.run(['cp', str(temp_detected), str(detected_image)], check=True)
                
                logger.info(f"Successfully processed: {image_path.name}")
                
                # Clean up temp directory
                subprocess.run(['rm', '-rf', str(temp_dir)], check=True)
                return True
            else:
                logger.error(f"Detection failed for {image_path.name}: {result.stderr}")
                # Clean up temp directory
                subprocess.run(['rm', '-rf', str(temp_dir)], check=True)
                return False
                
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            return False
    
    def process_existing_images(self):
        """Process all existing images in the monitor directory."""
        logger.info("Processing existing images...")
        
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(self.monitor_dir.glob(ext))
        
        # Filter out images that shouldn't be processed
        images_to_process = [img for img in image_files if self.should_process_image(img)]
        
        if not images_to_process:
            logger.info("No new images found to process")
            return
        
        logger.info(f"Found {len(images_to_process)} new images to process (out of {len(image_files)} total)")
        
        processed = 0
        for image_path in sorted(images_to_process):
            if self.process_image(image_path):
                processed += 1
            
            # Small delay to prevent overwhelming the system
            time.sleep(0.1)
        
        logger.info(f"Processed {processed}/{len(images_to_process)} existing images")
    
    def start_monitoring(self):
        """Start monitoring the directory for new images."""
        logger.info("Starting directory monitoring...")
        
        event_handler = ImageEventHandler(self)
        self.observer = Observer()
        self.observer.schedule(event_handler, str(self.monitor_dir), recursive=False)
        self.observer.start()
        
        logger.info(f"Monitoring started for: {self.monitor_dir}")
    
    def start(self):
        """Start the service."""
        logger.info("Starting Hailo Detection Service...")
        self.running = True
        
        # Process existing images first
        self.process_existing_images()
        
        # Start monitoring for new images
        self.start_monitoring()
        
        # Keep service running
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Service interrupted by user")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the service."""
        logger.info("Stopping Hailo Detection Service...")
        self.running = False
        
        if self.observer:
            self.observer.stop()
            self.observer.join()
        
        logger.info("Service stopped")

class ImageEventHandler(FileSystemEventHandler):
    """File system event handler for new images."""
    
    def __init__(self, service: HailoDetectionService):
        self.service = service
        super().__init__()
    
    def on_created(self, event):
        """Handle new file creation."""
        if event.is_directory:
            return
        
        file_path = Path(event.src_path)
        
        # Check if it's an image file that should be processed
        if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            # Skip if it's a detected image or already processed
            if not self.service.should_process_image(file_path):
                logger.debug(f"Skipping new file: {file_path.name} - already processed or detected image")
                return
                
            logger.info(f"New image detected: {file_path.name}")
            
            # Small delay to ensure file is fully written
            time.sleep(0.5)
            
            # Process the new image
            self.service.process_image(file_path)

def signal_handler(signum, frame):
    """Handle system signals for graceful shutdown."""
    logger.info(f"Received signal {signum}, shutting down...")
    sys.exit(0)

def main():
    """Main entry point."""
    # Set up signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    # Create and start service
    service = HailoDetectionService()
    service.start()

if __name__ == '__main__':
    main()
