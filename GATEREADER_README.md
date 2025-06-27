# Gatereader - Object Detection REST API

⚠️ **WARNING: This code is currently untested and may require adjustments for your specific Hailo AI setup and model configuration.**

Gatereader is a REST API service that provides object detection capabilities using Hailo AI hardware acceleration. It mirrors the API structure of truckreader but performs object detection instead of OCR.

## Architecture

- **Flask REST API** - Same structure as truckreader
- **Hailo AI Integration** - Hardware-accelerated object detection
- **Compatible API** - Drop-in replacement for truckreader endpoints
- **Production Ready** - Error handling, logging, and health checks

## API Endpoints

### POST /extract
Basic object detection endpoint that returns detected objects with bounding boxes and confidence scores.

**Request:**
- Multipart form data with `image` file

**Response:**
```json
{
  "results": [
    {
      "text": "person",
      "bbox": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
      "confidence": 0.85,
      "object_class": "person",
      "class_id": 0
    }
  ],
  "full_text": "person, car, bicycle",
  "image_size": {
    "width": 1920,
    "height": 1080
  }
}
```

### POST /extract_with_crops
Object detection with cropped images of detected objects.

**Request:**
- Multipart form data with `image` file

**Response:**
```json
{
  "results": [
    {
      "text": "person",
      "bbox": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
      "confidence": 0.85,
      "cropped_image": "base64_encoded_cropped_image",
      "object_class": "person",
      "class_id": 0
    }
  ],
  "full_text": "person, car, bicycle",
  "image_size": {
    "width": 1920,
    "height": 1080
  }
}
```

### GET /health
Health check endpoint to verify service status.

**Response:**
```json
{
  "status": "healthy",
  "detector_initialized": true
}
```

## Installation

1. Install dependencies:
```bash
pip install -r gatereader_requirements.txt
```

2. Ensure Hailo AI hardware and drivers are properly installed

3. Place your YOLOv5 HEF model file in the project directory (default: `yolov5s.hef`)

## Usage

### Start the server:
```bash
python gatereader_server.py
```

The server will start on `http://0.0.0.0:5001`

### Test with curl:
```bash
# Basic object detection
curl -X POST -F "image=@test_image.jpg" http://localhost:5001/extract

# Object detection with crops
curl -X POST -F "image=@test_image.jpg" http://localhost:5001/extract_with_crops

# Health check
curl http://localhost:5001/health
```

## Integration with Gatekeeper

Gatekeeper can call gatereader instead of dropping files:

```python
import requests

# Send image to gatereader
with open('captured_image.jpg', 'rb') as f:
    response = requests.post('http://localhost:5001/extract', 
                           files={'image': f})
    
detections = response.json()
# Process detections...
```

## Supported Object Classes

The service detects 80 COCO object classes including:
- person, bicycle, car, motorcycle, airplane, bus, train, truck
- boat, traffic light, fire hydrant, stop sign, parking meter
- And many more...

## Configuration

- **Port**: 5001 (configurable in `gatereader_server.py`)
- **Model**: YOLOv5 HEF file (configurable in `HailoObjectDetector` constructor)
- **Confidence Threshold**: 0.5 (configurable in `postprocess_detections`)

## Error Handling

The API returns appropriate HTTP status codes:
- 200: Success
- 400: Bad request (invalid image, missing file)
- 500: Server error (detector not initialized, inference failure)

## Logging

All operations are logged with appropriate levels:
- INFO: Successful initialization and operations
- ERROR: Failures and exceptions

Logs are output to stdout and can be redirected to files as needed.
