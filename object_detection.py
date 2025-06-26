#!/usr/bin/env python3

import argparse
import os
import sys
from pathlib import Path
import numpy as np
from loguru import logger
import queue
import threading
import cv2
from typing import List
from object_detection_utils import ObjectDetectionUtils
import json
from datetime import datetime

# Add the parent directory to the system path to access utils module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import HailoAsyncInference, load_images_opencv, validate_images, divide_list_to_batches


CAMERA_CAP_WIDTH = 1920
CAMERA_CAP_HEIGHT = 1080 
        
def parse_args() -> argparse.Namespace:
    """
    Initialize argument parser for the script.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Detection Example")
    parser.add_argument(
        "-n", "--net", 
        help="Path for the network in HEF format.",
        default="yolov7.hef"
    )
    parser.add_argument(
        "-i", "--input", 
        default="zidane.jpg",
        help="Path to the input - either an image or a folder of images."
    )
    parser.add_argument(
        "-b", "--batch_size", 
        default=1,
        type=int,
        required=False,
        help="Number of images in one batch"
    )
    parser.add_argument(
        "-l", "--labels", 
        default="coco.txt",
        help="Path to a text file containing labels. If no labels file is provided, coco2017 will be used."
    )
    parser.add_argument(
        "-s", "--save_stream_output",
        action="store_true",
        help="Save the output of the inference from a stream."
    )
    parser.add_argument(
        "--folder-mode",
        action="store_true",
        help="Process entire folder of images with JSON files and print detection metadata to terminal."
    )

    args = parser.parse_args()

    # Validate paths
    if not os.path.exists(args.net):
        raise FileNotFoundError(f"Network file not found: {args.net}")
    if not os.path.exists(args.labels):
        raise FileNotFoundError(f"Labels file not found: {args.labels}")

    return args


def preprocess(
    images: List[np.ndarray],
    cap: cv2.VideoCapture,
    batch_size: int,
    input_queue: queue.Queue,
    width: int,
    height: int,
    utils: ObjectDetectionUtils
) -> None:
    """
    Preprocess and enqueue images or camera frames into the input queue as they are ready.

    Args:
        images (List[np.ndarray], optional): List of images as NumPy arrays.
        camera (bool, optional): Boolean indicating whether to use the camera stream.
        batch_size (int): Number of images per batch.
        input_queue (queue.Queue): Queue for input images.
        width (int): Model input width.
        height (int): Model input height.
        utils (ObjectDetectionUtils): Utility class for object detection preprocessing.
    """
    if cap is None:
        preprocess_images(images, batch_size, input_queue, width, height, utils)
    else:
        preprocess_from_cap(cap, batch_size, input_queue, width, height, utils)

    input_queue.put(None)  # Add sentinel value to signal end of input

def preprocess_from_cap(cap: cv2.VideoCapture, batch_size: int, input_queue: queue.Queue, width: int, height: int, utils: ObjectDetectionUtils) -> None:
    """
    Process frames from the camera stream and enqueue them.

    Args:
        batch_size (int): Number of images per batch.
        input_queue (queue.Queue): Queue for input images.
        width (int): Model input width.
        height (int): Model input height.
        utils (ObjectDetectionUtils): Utility class for object detection preprocessing.
    """
    frames = []
    processed_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frames.append(frame)
        processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed_frame = utils.preprocess(processed_frame, width, height)
        processed_frames.append(processed_frame)

        if len(frames) == batch_size:
            input_queue.put((frames, processed_frames))
            processed_frames, frames = [], []


def preprocess_images(images: List[np.ndarray], batch_size: int, input_queue: queue.Queue, width: int, height: int, utils: ObjectDetectionUtils) -> None:
    """
    Process a list of images and enqueue them.

    Args:
        images (List[np.ndarray]): List of images as NumPy arrays.
        batch_size (int): Number of images per batch.
        input_queue (queue.Queue): Queue for input images.
        width (int): Model input width.
        height (int): Model input height.
        utils (ObjectDetectionUtils): Utility class for object detection preprocessing.
    """
    for batch in divide_list_to_batches(images, batch_size):
        input_tuple = ([image for image in batch], [utils.preprocess(image, width, height) for image in batch])
        input_queue.put(input_tuple)

def postprocess(
    output_queue: queue.Queue,
    cap: cv2.VideoCapture,
    save_stream_output: bool,
    utils: ObjectDetectionUtils,
    input_path: str = None,
    folder_mode: bool = False,
    input_images: List[str] = None
) -> None:
    """
    Process and visualize the output results.

    Args:
        output_queue (queue.Queue): Queue for output results.
        camera (bool): Flag indicating if the input is from a camera.
        save_stream_output (bool): Flag indicating if the camera output should be saved.
        utils (ObjectDetectionUtils): Utility class for object detection visualization.
        input_path (str): Path to input file to determine output directory.
        folder_mode (bool): Flag indicating if the input is a folder.
        input_images (List[str]): List of input image paths for folder mode.
    """
    image_id = 0
    out = None
    
    # Determine output path based on input path
    if input_path and 'test/' in input_path:
        output_path = Path('test')
    else:
        output_path = Path('output')

    while True:
        result = output_queue.get()
        if result is None:
            break  # Exit the loop if sentinel value is received

        original_frame, infer_results = result

        # Deals with the expanded results from hailort versions < 4.19.0
        if len(infer_results) == 1:
            infer_results = infer_results[0]

        detections = utils.extract_detections(infer_results)

        if folder_mode:
            # Get current image path
            current_image_path = input_images[image_id] if input_images and image_id < len(input_images) else None
            
            # Print detailed detection metadata
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n=== Detection Results for {Path(current_image_path).name if current_image_path else f'Image {image_id}'} ===")
            print(f"Timestamp: {timestamp}")
            print(f"Objects detected: {detections['num_detections']}")
            
            # Prepare detection metadata for JSON
            detection_metadata = {
                "detection_timestamp": timestamp,
                "object_count": detections['num_detections'],
                "objects_detected": [],
                "model_used": "yolov8s_h8l.hef",
                "detection_method": "hailo_application_code_examples"
            }
            
            if detections['num_detections'] > 0:
                boxes = detections['detection_boxes']
                scores = detections['detection_scores'] 
                classes = detections['detection_classes']
                
                for i in range(detections['num_detections']):
                    class_id = classes[i]
                    label = utils.labels[class_id] if class_id < len(utils.labels) else f"Class_{class_id}"
                    confidence = scores[i]
                    bbox = boxes[i]
                    
                    # Add to metadata
                    detection_metadata["objects_detected"].append({
                        "label": label,
                        "confidence": float(confidence),
                        "bbox": [float(x) for x in bbox]
                    })
                    
                    print(f"  Object {i+1}:")
                    print(f"    Label: {label}")
                    print(f"    Confidence: {confidence:.2f}")
                    print(f"    Bounding Box: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")
            else:
                print("  No objects detected")
            
            # Update JSON file if it exists
            if current_image_path:
                json_path = Path(current_image_path).with_suffix('.json')
                if json_path.exists():
                    try:
                        with open(json_path, 'r') as f:
                            json_data = json.load(f)
                        
                        # Update with detection metadata
                        json_data.update(detection_metadata)
                        
                        with open(json_path, 'w') as f:
                            json.dump(json_data, f, indent=2)
                        
                        print(f"  Updated JSON: {json_path.name}")
                    except Exception as e:
                        print(f"  Failed to update JSON: {e}")
                
                # Generate output image with overlay
                frame_with_detections = utils.draw_detections(detections, original_frame)
                
                # Save annotated image in same directory as original
                output_image_path = Path(current_image_path).parent / f"{Path(current_image_path).stem}_detected.jpg"
                cv2.imwrite(str(output_image_path), frame_with_detections)
                print(f"  Saved annotated image: {output_image_path.name}")
            
            print("=" * 50)
        else:
            frame_with_detections = utils.draw_detections(
                detections, original_frame,
            )
            
            if cap is not None:
                # Display output
                cv2.imshow("Output", frame_with_detections)
                if save_stream_output:
                    out.write(frame_with_detections)
            else:
                cv2.imwrite(str(output_path / f"output_{image_id}.png"), frame_with_detections)

        # Wait for key press "q"
        image_id += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            # Close the window and release the camera
            if save_stream_output:
                out.release()  # Release the VideoWriter object
            cap.release()
            cv2.destroyAllWindows()
            break

    if cap is not None and save_stream_output:
            out.release()  # Release the VideoWriter object
    output_queue.task_done()  # Indicate that processing is complete


def infer(
    input_images: List[str],
    save_stream_output: bool,
    net_path: str,
    labels_path: str,
    batch_size: int,
    folder_mode: bool
) -> None:
    """
    Initialize queues, HailoAsyncInference instance, and run the inference.

    Args:
        input_images (List[str]): List of image paths to process.
        save_stream_output (bool): Flag indicating if the camera output should be saved.
        net_path (str): Path to the network in HEF format.
        labels_path (str): Path to a text file containing labels.
        batch_size (int): Number of images per batch.
        folder_mode (bool): Flag indicating if processing multiple images from folder.
    """
    det_utils = ObjectDetectionUtils(labels_path)
    
    cap = None
    images = []
    
    if folder_mode:
        # Process multiple images from folder
        all_images = []
        for img_path in input_images:
            img_images = load_images_opencv(img_path)
            all_images.extend(img_images)
        images = all_images
        logger.info(f"Processing {len(images)} images in folder mode")
    elif len(input_images) == 1:
        input_path = input_images[0]
        if input_path == "camera":
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_CAP_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_CAP_HEIGHT)
        elif any(input_path.lower().endswith(suffix) for suffix in ['.mp4', '.avi', '.mov', '.mkv']):
            cap = cv2.VideoCapture(input_path)
        else:
            images = load_images_opencv(input_path)

        # Validate images
        try:
            validate_images(images, batch_size)
        except ValueError as e:
            logger.error(e)
            return

    input_queue = queue.Queue()
    output_queue = queue.Queue()

    hailo_inference = HailoAsyncInference(
        net_path, input_queue, output_queue, batch_size, send_original_frame=True
    )
    height, width, _ = hailo_inference.get_input_shape()

    preprocess_thread = threading.Thread(
        target=preprocess,
        args=(images, cap, batch_size, input_queue, width, height, det_utils)
    )
    postprocess_thread = threading.Thread(
        target=postprocess,
        args=(output_queue, cap, save_stream_output, det_utils, input_images[0] if input_images else None, folder_mode, input_images)
    )

    preprocess_thread.start()
    postprocess_thread.start()

    hailo_inference.run()
    
    preprocess_thread.join()
    output_queue.put(None)  # Signal process thread to exit
    postprocess_thread.join()

    logger.info('Inference was successful!')


def find_images_in_folder(folder_path: str) -> List[str]:
    """
    Find all images in a folder.
    
    Args:
        folder_path (str): Path to the folder containing images.
        
    Returns:
        List[str]: List of image file paths.
    """
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"Folder does not exist: {folder_path}")
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    
    for image_file in folder.iterdir():
        if image_file.suffix.lower() in image_extensions:
            image_files.append(str(image_file))
    
    logger.info(f"Found {len(image_files)} images in folder")
    return sorted(image_files)


def main() -> None:
    """
    Main function to run the script.
    """
    # Parse command line arguments
    args = parse_args()

    # Start the inference
    if args.folder_mode:
        images = find_images_in_folder(args.input)
    else:
        images = [args.input]
    infer(images, args.save_stream_output, args.net, args.labels, args.batch_size, args.folder_mode)


if __name__ == "__main__":
    main()
