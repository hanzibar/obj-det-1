# Hailo AI Object Detection

A complete, self-contained object detection system using Hailo-8L AI accelerator with YOLOv8s models. This project provides both single image detection and batch processing capabilities for Raspberry Pi 5 with Hailo AI Hat.

## 🚀 Quick Start

### Prerequisites
- Raspberry Pi 5 with Hailo-8L AI accelerator (13 TOPS)
- Python 3.11+
- Git

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd obj-det-1
```

2. **Install Python dependencies:**
```bash
pip install -r requirements.txt --break-system-packages
```

3. **Test the detection:**
```bash
python3 object_detection.py -n models/yolov8s_h8l.hef -i test/test_image.jpg -l coco.txt
```

**Expected output:**
```
2025-06-25 17:03:32.290 | INFO     | __main__:infer:289 - Inference was successful!
```

**Results:** Check `output/output_0.png` for the annotated image with detected objects and bounding boxes.

## 📁 Project Structure

```
obj-det-1/
├── models/                          # AI models and libraries
│   ├── yolov8s_h8l.hef             # Main object detection (22MB)
│   ├── yolov6n_h8l.hef             # Alternative object detection (8MB)
│   ├── yolov5n_seg_h8l.hef         # Instance segmentation (9MB)
│   ├── yolov8s_pose_h8l.hef        # Pose estimation (23MB)
│   ├── scdepthv3_h8l.hef           # Depth estimation (32MB)
│   └── libyolo_hailortpp_postprocess.so  # Post-processing library
├── test/                           # Test files and results
│   ├── test_image.jpg              # Sample test image
│   ├── hailo_detection_result.png  # Example detection result
│   └── remote_detection_result.png # Headless test result
├── output/                         # Detection output directory
├── object_detection.py             # Main detection script
├── object_detection_utils.py       # Detection utilities
├── utils.py                        # Hailo inference utilities
├── coco.txt                        # COCO dataset labels (80 classes)
├── requirements.txt                # Python dependencies
├── obj_det_service.py              # Service implementation
└── README.md                       # This file
```

## 🎯 Usage

### Single Image Detection

**Basic detection:**
```bash
python3 object_detection.py -n models/yolov8s_h8l.hef -i path/to/image.jpg -l coco.txt
```

**Using different models:**
```bash
# Faster, smaller model
python3 object_detection.py -n models/yolov6n_h8l.hef -i test/test_image.jpg -l coco.txt

# Instance segmentation
python3 object_detection.py -n models/yolov5n_seg_h8l.hef -i test/test_image.jpg -l coco.txt

# Pose estimation
python3 object_detection.py -n models/yolov8s_pose_h8l.hef -i test/test_image.jpg -l coco.txt
```

### Command Line Options

```bash
python3 object_detection.py [OPTIONS]

Options:
  -n, --net PATH        Path to HEF model file (required)
  -i, --input PATH      Path to input image (required)
  -l, --labels PATH     Path to labels file (required)
  -b, --batch-size INT  Batch size for processing (default: 1)
  -s, --silent          Suppress output messages
```

### Remote/Headless Operation

The detection works perfectly over SSH without GUI:
```bash
# Connect via SSH
ssh user@raspberry-pi-ip

# Run detection
cd obj-det-1
python3 object_detection.py -n models/yolov8s_h8l.hef -i test/test_image.jpg -l coco.txt

# Copy results back to local machine
scp user@raspberry-pi-ip:~/obj-det-1/output/output_0.png ./
```

## 🔧 Advanced Usage

### Service Mode (Continuous Processing)

The `obj_det_service.py` provides advanced features:
- Directory monitoring for new images
- JSON metadata integration
- Batch processing
- Real-time detection service

```bash
# Process existing images in a directory
python3 obj_det_service.py --process-existing /path/to/images

# Monitor directory for new images
python3 obj_det_service.py --monitor /path/to/images

# Test single image with JSON output
python3 obj_det_service.py --test-image test/test_image.jpg
```

## 🎨 Detected Object Classes

The models detect 80 COCO dataset classes including:
- **People**: person
- **Vehicles**: car, truck, bus, motorcycle, bicycle
- **Animals**: dog, cat, horse, cow, sheep, bird
- **Objects**: bottle, cup, laptop, phone, book, chair, table
- **And 65+ more classes** (see `coco.txt` for complete list)

## 🔧 Hardware Requirements

### Verified Compatible Hardware
- **Raspberry Pi 5** (4GB/8GB)
- **Hailo-8L AI Accelerator** (13 TOPS)
  - Form Factor: M.2 B+M KEY MODULE
  - Device Architecture: HAILO8L
  - Firmware Version: 4.20.0+

### Performance
- **YOLOv8s**: ~50-100ms per image (depending on image size)
- **YOLOv6n**: ~30-60ms per image (faster, smaller model)
- **Memory**: ~2GB RAM recommended for batch processing

## 🐛 Troubleshooting

### Common Issues

**1. "HAILO_OUT_OF_PHYSICAL_DEVICES" error:**
```bash
# Check if another process is using the Hailo device
sudo lsof /dev/hailo*
# Kill conflicting processes if needed
```

**2. "ModuleNotFoundError" for dependencies:**
```bash
# Reinstall requirements
pip install -r requirements.txt --break-system-packages --force-reinstall
```

**3. Permission denied errors:**
```bash
# Make scripts executable
chmod +x object_detection.py
chmod +x obj_det_service.py
```

**4. Model compatibility errors:**
- Ensure you're using Hailo-8L models (ending with `_h8l.hef`)
- Hailo-8 models (26 TOPS) will NOT work on Hailo-8L (13 TOPS)

### Verification Commands

**Check Hailo device:**
```bash
hailo fw-control identify
```

**Test basic functionality:**
```bash
# Should show "Inference was successful!"
python3 object_detection.py -n models/yolov8s_h8l.hef -i test/test_image.jpg -l coco.txt
```

## 📊 Model Comparison

| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| YOLOv6n | 8MB | Fastest | Good | Real-time, resource-constrained |
| YOLOv8s | 22MB | Fast | Better | Balanced performance |
| YOLOv5n_seg | 9MB | Medium | Good + Segmentation | Instance segmentation |
| YOLOv8s_pose | 23MB | Medium | Good + Pose | Human pose estimation |
| SCDepthV3 | 32MB | Slow | Specialized | Depth estimation |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Test with your Hailo hardware
4. Submit a pull request

## 📄 License

This project incorporates code from Hailo-Application-Code-Examples and is subject to their licensing terms.

## 🔗 Related Projects

- [Hailo-Application-Code-Examples](https://github.com/hailo-ai/Hailo-Application-Code-Examples)
- [hailo-rpi5-examples](https://github.com/hailo-ai/hailo-rpi5-examples)
- [Hailo Developer Zone](https://hailo.ai/developer-zone/)

---

**Ready to detect objects with AI? Start with the Quick Start guide above! 🚀**
