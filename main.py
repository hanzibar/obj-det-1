#!/usr/bin/env python3
"""
Object Detection Service for Gatekeeper
Monitors gatekeeper captures folder and adds AI object detection to existing JSON files.
"""

import os
import json
import time
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from detector import detect_objects

# Configuration
MONITOR_FOLDER = "/home/jdneff/Projects/gatekeeper/captures"
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png'}

class ImageHandler(FileSystemEventHandler):
    def __init__(self):
        self.processed_files = set()
        
    def on_created(self, event):
        if event.is_directory:
            return
            
        file_path = event.src_path
        file_ext = Path(file_path).suffix.lower()
        
        # Only process supported image files
        if file_ext not in SUPPORTED_EXTENSIONS:
            return
            
        print(f"New image detected: {file_path}")
        
        # Wait a moment for file to be fully written
        time.sleep(0.5)
        
        # Process the image
        self.process_image(file_path)
    
    def process_image(self, image_path):
        """Process an image and update its corresponding JSON file."""
        try:
            # Skip if already processed
            if image_path in self.processed_files:
                return
                
            print(f"Processing {image_path}...")
            
            # Find corresponding JSON file
            image_file = Path(image_path)
            json_path = image_file.with_suffix('.json')
            
            if not json_path.exists():
                print(f"Warning: No JSON file found for {image_path}")
                return
            
            # Check if already has object detection results
            try:
                with open(json_path, 'r') as f:
                    existing_data = json.load(f)
                
                # Check if ai_analysis already has object_detection
                if (existing_data.get('ai_analysis', {}).get('object_detection') is not None):
                    print(f"Skipping {image_path} - already has object detection results")
                    self.processed_files.add(image_path)
                    return
                    
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error reading existing JSON {json_path}: {e}")
                return
            
            # Run object detection
            detections = detect_objects(image_path)
            
            # Prepare detection results
            detection_results = {
                "timestamp": time.time(),
                "detection_count": len(detections),
                "detections": detections
            }
            
            # Update the existing JSON file
            existing_data['ai_analysis']['object_detection'] = detection_results
            
            # Write back to JSON file
            with open(json_path, 'w') as f:
                json.dump(existing_data, f, indent=2)
            
            print(f"Successfully added {len(detections)} detections to {json_path}")
            self.processed_files.add(image_path)
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

def scan_existing_files():
    """Scan for existing images that haven't been processed yet."""
    print("Scanning for existing unprocessed images...")
    
    folder_path = Path(MONITOR_FOLDER)
    if not folder_path.exists():
        print(f"Error: Monitor folder {MONITOR_FOLDER} does not exist!")
        return []
    
    unprocessed = []
    
    for image_file in folder_path.glob("*.jpg"):
        json_file = image_file.with_suffix('.json')
        
        if json_file.exists():
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Check if needs object detection
                if data.get('ai_analysis', {}).get('object_detection') is None:
                    unprocessed.append(str(image_file))
                    
            except (json.JSONDecodeError, KeyError):
                continue
    
    print(f"Found {len(unprocessed)} unprocessed images")
    return unprocessed

def main():
    print("Starting Gatekeeper Object Detection Service...")
    print(f"Monitoring folder: {MONITOR_FOLDER}")
    print("Target objects: cars, trucks, bicycles, people")
    
    # Verify monitor folder exists
    if not os.path.exists(MONITOR_FOLDER):
        print(f"Error: Monitor folder {MONITOR_FOLDER} does not exist!")
        return
    
    print("Initializing Hailo AI detector...")
    
    # Test detector initialization
    try:
        test_detections = detect_objects.__defaults__  # This will trigger import and init
        print("Hailo AI detector ready!")
    except Exception as e:
        print(f"Error initializing detector: {e}")
        return
    
    # Create event handler and observer
    event_handler = ImageHandler()
    observer = Observer()
    observer.schedule(event_handler, MONITOR_FOLDER, recursive=False)
    
    # Process existing unprocessed files
    existing_files = scan_existing_files()
    for image_path in existing_files[:5]:  # Process first 5 to avoid overwhelming
        event_handler.process_image(image_path)
    
    # Start monitoring
    observer.start()
    print("Gatekeeper Object Detection Service is running. Press Ctrl+C to stop.")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("\nStopping service...")
    
    observer.join()
    print("Service stopped.")

if __name__ == "__main__":
    main()
