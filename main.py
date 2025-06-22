import time
import json
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from detector import detect_objects

# Configuration
MONITORED_FOLDER = '/home/jdneff/gatekeeper/captures'

class NewImageHandler(FileSystemEventHandler):
    """Handles the event when a new file is created."""
    
    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"New image detected: {event.src_path}")
            process_image(event.src_path)

def process_image(image_path):
    """Processes a new image to detect objects and updates the JSON file."""
    print(f"Processing {image_path}...")
    
    try:
        detected_objects = detect_objects(image_path)
        update_json_file(image_path, detected_objects)
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

def update_json_file(image_path, detected_objects):
    """Creates/updates JSON file with detection results."""
    json_path = os.path.splitext(image_path)[0] + '.json'
    
    output_data = {
        'image_path': image_path,
        'detections': detected_objects,
        'timestamp': time.time(),
        'detection_count': len(detected_objects)
    }

    try:
        with open(json_path, 'w') as f:
            json.dump(output_data, f, indent=4)
        print(f"Successfully wrote {len(detected_objects)} detections to {json_path}")
    except IOError as e:
        print(f"Error writing to JSON file: {e}")

def main():
    """Starts the folder monitoring service."""
    print(f"Starting object detection service...")
    print(f"Monitoring folder: {MONITORED_FOLDER}")
    print(f"Target objects: cars, trucks, bicycles, people")
    
    # Check if monitored folder exists
    if not os.path.isdir(MONITORED_FOLDER):
        print(f"Error: Monitored folder '{MONITORED_FOLDER}' does not exist.")
        print("Please create the folder or update the MONITORED_FOLDER path.")
        return

    # Initialize the detector (this will load the AI model)
    print("Initializing Hailo AI detector...")
    try:
        from detector import initialize_detector
        initialize_detector()
        print("Hailo AI detector ready!")
    except Exception as e:
        print(f"Error initializing detector: {e}")
        return

    # Set up file monitoring
    event_handler = NewImageHandler()
    observer = Observer()
    observer.schedule(event_handler, MONITORED_FOLDER, recursive=False)
    observer.start()

    print("Object detection service is running. Press Ctrl+C to stop.")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("\nStopping object detection service...")
    
    observer.join()
    print("Object detection service stopped.")

if __name__ == "__main__":
    main()
