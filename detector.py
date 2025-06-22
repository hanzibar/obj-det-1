import cv2
import numpy as np
from hailo_platform import (HEF, VDevice, HailoStreamInterface, InferVStreams, 
                           ConfigureParams, InputVStreamParams, OutputVStreamParams, FormatType)

# COCO class names for our target objects
COCO_CLASSES = {
    1: 'person',
    2: 'bicycle', 
    3: 'car',
    8: 'truck'
}

# Target classes we want to detect
TARGET_CLASSES = [1, 2, 3, 8]

class HailoDetector:
    def __init__(self, hef_path):
        """Initialize the Hailo detector with a model file."""
        self.hef_path = hef_path
        self.device = None
        self.infer_pipeline = None
        self.network_group = None
        self._setup_device()
    
    def _setup_device(self):
        """Set up the Hailo device and inference pipeline."""
        try:
            self.device = VDevice()
            hef = HEF(self.hef_path)
            
            # Configure the network
            configure_params = ConfigureParams.create_from_hef(hef=hef, interface=HailoStreamInterface.PCIe)
            self.network_group = self.device.configure(hef, configure_params)[0]
            
            # Create input/output stream parameters
            input_vstreams_params = InputVStreamParams.make_from_network_group(
                self.network_group, quantized=False, format_type=FormatType.UINT8
            )
            output_vstreams_params = OutputVStreamParams.make_from_network_group(
                self.network_group, quantized=False, format_type=FormatType.FLOAT32
            )
            
            # Create inference pipeline
            self.infer_pipeline = InferVStreams(self.network_group, input_vstreams_params, output_vstreams_params)
            
            print(f"Hailo detector initialized with model: {self.hef_path}")
            
        except Exception as e:
            print(f"Error initializing Hailo detector: {e}")
            raise
    
    def preprocess_image(self, image_path):
        """Load and preprocess image for inference."""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size (typically 640x640 for YOLO)
        # TODO: Get actual input size from model
        input_size = (640, 640)
        resized = cv2.resize(image_rgb, input_size)
        
        # Normalize to 0-255 range (UINT8)
        preprocessed = resized.astype(np.uint8)
        
        # Add batch dimension
        batch_input = np.expand_dims(preprocessed, axis=0)
        
        return batch_input, image.shape[:2]  # Return original image dimensions
    
    def postprocess_detections(self, raw_output, original_shape, confidence_threshold=0.5):
        """Convert raw model output to detection results."""
        detections = []
        
        try:
            # Handle NMS-processed output (yolov8s/yolov8_nms_postprocess)
            if isinstance(raw_output, dict) and 'yolov8s/yolov8_nms_postprocess' in raw_output:
                nms_output = raw_output['yolov8s/yolov8_nms_postprocess']
                
                # The NMS output is a list of arrays, one for each COCO class (80 classes)
                # Each array contains detections for that class: [x1, y1, x2, y2, confidence]
                
                # Get original image dimensions
                orig_h, orig_w = original_shape
                
                for class_id, class_detections in enumerate(nms_output):
                    # Only process our target classes
                    if class_id not in TARGET_CLASSES:
                        continue
                    
                    # Convert to numpy array if needed
                    if hasattr(class_detections, 'numpy'):
                        det_array = class_detections.numpy()
                    else:
                        det_array = np.array(class_detections)
                    
                    # Process each detection for this class
                    for detection in det_array:
                        if len(detection) >= 5:  # [x1, y1, x2, y2, confidence]
                            x1, y1, x2, y2, confidence = detection[:5]
                            
                            # Filter by confidence
                            if confidence < confidence_threshold:
                                continue
                            
                            # Coordinates are already in model coordinates (640x640), scale to original
                            model_size = 640
                            scale_x = orig_w / model_size
                            scale_y = orig_h / model_size
                            
                            # Scale coordinates
                            x1 = int(x1 * scale_x)
                            y1 = int(y1 * scale_y)
                            x2 = int(x2 * scale_x)
                            y2 = int(y2 * scale_y)
                            
                            # Clamp to image boundaries
                            x1 = max(0, min(x1, orig_w))
                            y1 = max(0, min(y1, orig_h))
                            x2 = max(0, min(x2, orig_w))
                            y2 = max(0, min(y2, orig_h))
                            
                            detections.append({
                                'class_id': class_id,
                                'confidence': float(confidence),
                                'bbox': [x1, y1, x2, y2]
                            })
                
                return detections
            
            # Fallback: Handle raw YOLO output (if no NMS)
            else:
                if isinstance(raw_output, dict):
                    output_tensor = list(raw_output.values())[0]
                else:
                    output_tensor = raw_output[0] if isinstance(raw_output, list) else raw_output
                
                # Convert to numpy if needed
                if hasattr(output_tensor, 'numpy'):
                    output_tensor = output_tensor.numpy()
                elif not hasattr(output_tensor, 'shape'):
                    return []
                
                # Reshape if needed: [batch, 84, 8400] -> [8400, 84]
                if len(output_tensor.shape) == 3:
                    output_tensor = output_tensor[0].transpose()  # [8400, 84]
                
                # Extract bounding boxes and scores
                boxes = output_tensor[:, :4]  # [x_center, y_center, width, height]
                scores = output_tensor[:, 4:]  # [80 class scores]
                
                # Get original image dimensions
                orig_h, orig_w = original_shape
                model_size = 640
                scale_x = orig_w / model_size
                scale_y = orig_h / model_size
                
                # Process each detection
                for i in range(len(boxes)):
                    class_scores = scores[i]
                    class_id = np.argmax(class_scores)
                    confidence = class_scores[class_id]
                    
                    if confidence < confidence_threshold:
                        continue
                    
                    if class_id not in TARGET_CLASSES:
                        continue
                    
                    x_center, y_center, width, height = boxes[i]
                    x_center *= scale_x
                    y_center *= scale_y
                    width *= scale_x
                    height *= scale_y
                    
                    x1 = int(x_center - width / 2)
                    y1 = int(y_center - height / 2)
                    x2 = int(x_center + width / 2)
                    y2 = int(y_center + height / 2)
                    
                    x1 = max(0, min(x1, orig_w))
                    y1 = max(0, min(y1, orig_h))
                    x2 = max(0, min(x2, orig_w))
                    y2 = max(0, min(y2, orig_h))
                    
                    detections.append({
                        'class_id': int(class_id),
                        'confidence': float(confidence),
                        'bbox': [x1, y1, x2, y2]
                    })
                
                # Apply NMS if we processed raw output
                if detections:
                    detections = self._apply_nms(detections, iou_threshold=0.5)
                
                return detections
            
        except Exception as e:
            print(f"Error in postprocessing: {e}")
            return []
    
    def _apply_nms(self, detections, iou_threshold=0.5):
        """Apply Non-Maximum Suppression to remove overlapping detections."""
        if not detections:
            return []
        
        # Sort by confidence (highest first)
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        keep = []
        while detections:
            # Keep the highest confidence detection
            current = detections.pop(0)
            keep.append(current)
            
            # Remove detections with high IoU with current detection
            detections = [det for det in detections 
                         if self._calculate_iou(current['bbox'], det['bbox']) < iou_threshold]
        
        return keep
    
    def _calculate_iou(self, box1, box2):
        """Calculate Intersection over Union (IoU) of two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection area
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union area
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def detect_objects(self, image_path):
        """Main detection function - processes an image and returns detections."""
        try:
            # Preprocess image
            input_data, original_shape = self.preprocess_image(image_path)
            
            # Run inference
            with self.infer_pipeline as pipeline:
                network_group_params = self.network_group.create_params()
                with self.network_group.activate(network_group_params):
                    raw_output = pipeline.infer(input_data)
            
            # Postprocess results
            all_detections = self.postprocess_detections(raw_output, original_shape)
            
            # Filter for target classes
            filtered_detections = [
                det for det in all_detections 
                if det.get('class_id') in TARGET_CLASSES
            ]
            
            # Convert to our output format
            results = []
            for det in filtered_detections:
                results.append({
                    'class_id': det['class_id'],
                    'class_name': COCO_CLASSES.get(det['class_id'], 'unknown'),
                    'confidence': det['confidence'],
                    'bbox': det['bbox']  # [x1, y1, x2, y2]
                })
            
            print(f"Detected {len(results)} target objects in {image_path}")
            return results
            
        except Exception as e:
            print(f"Error detecting objects in {image_path}: {e}")
            return []
    
    def cleanup(self):
        """Clean up resources."""
        if self.device:
            self.device = None

# Global detector instance
_detector = None

def initialize_detector(hef_path="/usr/share/hailo-models/yolov8s_h8l.hef"):
    """Initialize the global detector instance."""
    global _detector
    if _detector is None:
        _detector = HailoDetector(hef_path)
    return _detector

def detect_objects(image_path):
    """Convenience function for object detection."""
    detector = initialize_detector()
    return detector.detect_objects(image_path)
