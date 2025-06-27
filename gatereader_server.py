from flask import Flask, request, jsonify
from PIL import Image
import io
import numpy as np
import base64
from PIL import ImageOps
import cv2
import json
import logging
from hailo_platform import (HEF, VDevice, HailoStreamInterface, InferVStreams, ConfigureParams,
                            InputVStreamParams, OutputVStreamParams, InputVStreams, OutputVStreams, FormatType)

from flask_cors import CORS
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HailoObjectDetector:
    def __init__(self, hef_path="yolov5s.hef"):
        self.hef_path = hef_path
        self.hef = None
        self.vdevice = None
        self.network_group = None
        self.network_group_params = None
        self.input_vstreams = None
        self.output_vstreams = None
        self.input_vstream_info = None
        self.output_vstream_info = None
        self._initialize_hailo()
    
    def _initialize_hailo(self):
        """Initialize Hailo device and model"""
        try:
            # Load HEF file
            self.hef = HEF(self.hef_path)
            
            # Create VDevice
            self.vdevice = VDevice(device_ids=["0"])
            
            # Configure network group
            self.network_group = self.vdevice.configure(self.hef)[0]
            self.network_group_params = self.network_group.create_params()
            
            # Get input/output stream info
            self.input_vstream_info = self.hef.get_input_vstream_infos()[0]
            self.output_vstream_info = self.hef.get_output_vstream_infos()
            
            logger.info("Hailo AI initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Hailo AI: {e}")
            raise
    
    def preprocess_image(self, image):
        """Preprocess image for Hailo inference"""
        # Get expected input shape
        height, width, channels = self.input_vstream_info.shape
        
        # Resize image to expected dimensions
        image_resized = cv2.resize(image, (width, height))
        
        # Convert to RGB if needed
        if len(image_resized.shape) == 3 and image_resized.shape[2] == 3:
            image_resized = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1] range
        image_normalized = image_resized.astype(np.float32) / 255.0
        
        # Add batch dimension
        return np.expand_dims(image_normalized, axis=0)
    
    def postprocess_detections(self, raw_detections, original_shape, confidence_threshold=0.5):
        """Convert raw Hailo detections to standardized format"""
        detections = []
        
        # Parse raw detections (format depends on model)
        # This is a simplified example - adjust based on your specific model output
        for detection in raw_detections:
            if len(detection) >= 6:  # x1, y1, x2, y2, confidence, class_id
                x1, y1, x2, y2, confidence, class_id = detection[:6]
                
                if confidence >= confidence_threshold:
                    # Scale coordinates to original image size
                    orig_height, orig_width = original_shape[:2]
                    x1 = int(x1 * orig_width)
                    y1 = int(y1 * orig_height)
                    x2 = int(x2 * orig_width)
                    y2 = int(y2 * orig_height)
                    
                    # Convert to bbox format (4 corners)
                    bbox = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                    
                    detections.append({
                        'object_class': self.get_class_name(int(class_id)),
                        'bbox': bbox,
                        'confidence': float(confidence),
                        'class_id': int(class_id)
                    })
        
        return detections
    
    def get_class_name(self, class_id):
        """Get class name from class ID (COCO classes for YOLOv5)"""
        coco_classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
        
        if 0 <= class_id < len(coco_classes):
            return coco_classes[class_id]
        return f"unknown_{class_id}"
    
    def detect_objects(self, image_np):
        """Perform object detection on image"""
        try:
            # Preprocess image
            input_data = self.preprocess_image(image_np)
            
            # Create input/output streams
            input_vstreams_params = InputVStreamParams.make_from_network_group(self.network_group, quantized=False, format_type=FormatType.FLOAT32)
            output_vstreams_params = OutputVStreamParams.make_from_network_group(self.network_group, quantized=False, format_type=FormatType.FLOAT32)
            
            with InferVStreams(self.network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
                # Run inference
                raw_detections = infer_pipeline.infer(input_data)
                
                # Postprocess results
                detections = self.postprocess_detections(raw_detections, image_np.shape)
                
                return detections
                
        except Exception as e:
            logger.error(f"Object detection failed: {e}")
            return []

# Initialize the detector once when the server starts
try:
    detector = HailoObjectDetector()
    logger.info("Hailo object detector initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize object detector: {e}")
    detector = None

def crop_and_encode_image(image, bbox):
    """Crop image based on bounding box and encode as base64"""
    try:
        # Extract coordinates from bbox
        x_coords = [point[0] for point in bbox]
        y_coords = [point[1] for point in bbox]
        
        left = max(0, min(x_coords))
        top = max(0, min(y_coords))
        right = min(image.width, max(x_coords))
        bottom = min(image.height, max(y_coords))
        
        # Crop the image
        cropped = image.crop((left, top, right, bottom))
        
        # Convert to base64
        buffer = io.BytesIO()
        cropped.save(buffer, format='JPEG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        return img_str
    except Exception as e:
        logger.error(f"Failed to crop and encode image: {e}")
        return None

@app.route('/extract', methods=['POST'])
def extract_objects():
    """Basic object detection endpoint (mirrors truckreader's /extract)"""
    if detector is None:
        return jsonify({"error": "Object detector not initialized"}), 500
        
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files["image"]
    
    try:
        image = Image.open(file.stream).convert("RGB")
        image_np = np.array(image)
    except Exception as e:
        return jsonify({"error": "Invalid image file", "details": str(e)}), 400
    
    # Perform object detection
    detections = detector.detect_objects(image_np)
    
    # Format results to match truckreader structure
    formatted_results = [{
        'text': detection['object_class'],  # Use object class as 'text' for compatibility
        'bbox': detection['bbox'],
        'confidence': detection['confidence'],
        'object_class': detection['object_class'],
        'class_id': detection['class_id']
    } for detection in detections]
    
    return jsonify({
        "results": formatted_results,
        "full_text": ', '.join([detection['object_class'] for detection in detections]),
        "image_size": {
            "width": int(image_np.shape[1]),
            "height": int(image_np.shape[0])
        }
    })

@app.route('/extract_with_crops', methods=['POST'])
def extract_objects_with_crops():
    """Object detection with cropped object images (mirrors truckreader's /extract_with_crops)"""
    if detector is None:
        return jsonify({"error": "Object detector not initialized"}), 500
        
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files["image"]
    
    try:
        image = Image.open(file.stream).convert("RGB")
        image_np = np.array(image)
    except Exception as e:
        return jsonify({"error": "Invalid image file", "details": str(e)}), 400
    
    # Perform object detection
    detections = detector.detect_objects(image_np)
    
    # Format results with cropped images
    formatted_results = [{
        'text': detection['object_class'],  # Use object class as 'text' for compatibility
        'bbox': detection['bbox'],
        'confidence': detection['confidence'],
        'cropped_image': crop_and_encode_image(image, detection['bbox']),
        'object_class': detection['object_class'],
        'class_id': detection['class_id']
    } for detection in detections]
    
    # Debug output (like truckreader)
    print_formatted_results = [{
        'text': detection['object_class'],
        'bbox': detection['bbox'],
        'confidence': detection['confidence'],
        'object_class': detection['object_class'],
        'class_id': detection['class_id']
    } for detection in detections]
    
    print(print_formatted_results)
    
    return jsonify({
        "results": formatted_results,
        "full_text": ', '.join([detection['object_class'] for detection in detections]),
        "image_size": {
            "width": int(image_np.shape[1]),
            "height": int(image_np.shape[0])
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "detector_initialized": detector is not None
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)  # Different port from truckreader
