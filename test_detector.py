#!/usr/bin/env python3
"""
Simple test script to verify Hailo detector setup
"""
import os
import sys

def test_imports():
    """Test if all required modules can be imported."""
    print("Testing imports...")
    
    try:
        import cv2
        print("✓ OpenCV imported successfully")
    except ImportError as e:
        print(f"✗ OpenCV import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✓ NumPy imported successfully")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        from hailo_platform import HEF, VDevice
        print("✓ Hailo platform imported successfully")
    except ImportError as e:
        print(f"✗ Hailo platform import failed: {e}")
        return False
    
    try:
        import watchdog
        print("✓ Watchdog imported successfully")
    except ImportError as e:
        print(f"✗ Watchdog import failed: {e}")
        return False
    
    return True

def test_model_file():
    """Test if the model file exists."""
    model_path = "/usr/share/hailo-models/yolov8s_h8l.hef"
    print(f"\nTesting model file: {model_path}")
    
    if os.path.exists(model_path):
        print("✓ Model file exists")
        file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
        print(f"  File size: {file_size:.1f} MB")
        return True
    else:
        print("✗ Model file not found")
        return False

def test_hailo_device():
    """Test if Hailo device is accessible."""
    print("\nTesting Hailo device...")
    
    try:
        from hailo_platform import VDevice
        device = VDevice()
        print("✓ Hailo device created successfully")
        return True
    except Exception as e:
        print(f"✗ Hailo device test failed: {e}")
        return False

def test_monitored_folder():
    """Test if monitored folder exists."""
    folder_path = "/home/jdneff/gatekeeper/captures"
    print(f"\nTesting monitored folder: {folder_path}")
    
    if os.path.exists(folder_path):
        print("✓ Monitored folder exists")
        return True
    else:
        print("✗ Monitored folder not found")
        return False

def main():
    """Run all tests."""
    print("=== Hailo Object Detection Setup Test ===\n")
    
    tests = [
        test_imports,
        test_model_file,
        test_hailo_device,
        test_monitored_folder
    ]
    
    results = []
    for test in tests:
        results.append(test())
        print()
    
    print("=== Test Summary ===")
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"✓ All {total} tests passed! Setup is ready.")
        return 0
    else:
        print(f"✗ {total - passed} of {total} tests failed. Please fix issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
