#!/usr/bin/env python3
"""
Test script for gatereader API
"""
import requests
import json
import sys
import os
from pathlib import Path

def test_health_endpoint(base_url="http://localhost:5001"):
    """Test the health endpoint"""
    try:
        response = requests.get(f"{base_url}/health")
        print(f"Health check status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_extract_endpoint(image_path, base_url="http://localhost:5001"):
    """Test the /extract endpoint"""
    if not os.path.exists(image_path):
        print(f"Image file not found: {image_path}")
        return False
    
    try:
        with open(image_path, 'rb') as f:
            files = {'image': f}
            response = requests.post(f"{base_url}/extract", files=files)
        
        print(f"Extract endpoint status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Detected objects: {len(result['results'])}")
            print(f"Full text: {result['full_text']}")
            print(f"Image size: {result['image_size']}")
            
            for i, detection in enumerate(result['results']):
                print(f"  {i+1}. {detection['object_class']} (confidence: {detection['confidence']:.2f})")
        else:
            print(f"Error response: {response.text}")
        
        return response.status_code == 200
    except Exception as e:
        print(f"Extract test failed: {e}")
        return False

def test_extract_with_crops_endpoint(image_path, base_url="http://localhost:5001"):
    """Test the /extract_with_crops endpoint"""
    if not os.path.exists(image_path):
        print(f"Image file not found: {image_path}")
        return False
    
    try:
        with open(image_path, 'rb') as f:
            files = {'image': f}
            response = requests.post(f"{base_url}/extract_with_crops", files=files)
        
        print(f"Extract with crops endpoint status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Detected objects with crops: {len(result['results'])}")
            print(f"Full text: {result['full_text']}")
            
            for i, detection in enumerate(result['results']):
                has_crop = detection.get('cropped_image') is not None
                print(f"  {i+1}. {detection['object_class']} (confidence: {detection['confidence']:.2f}, crop: {has_crop})")
        else:
            print(f"Error response: {response.text}")
        
        return response.status_code == 200
    except Exception as e:
        print(f"Extract with crops test failed: {e}")
        return False

def main():
    """Main test function"""
    base_url = "http://localhost:5001"
    
    print("=== Gatereader API Test ===")
    print(f"Testing API at: {base_url}")
    print()
    
    # Test health endpoint
    print("1. Testing health endpoint...")
    health_ok = test_health_endpoint(base_url)
    print()
    
    if not health_ok:
        print("❌ Health check failed. Make sure gatereader server is running.")
        sys.exit(1)
    
    # Find a test image
    test_image = None
    possible_paths = [
        "/home/jdneff/Projects/captures",  # Gatekeeper captures
        "/home/jdneff/Projects/obj-det-1",  # Current directory
        "."  # Current working directory
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                import glob
                images = glob.glob(os.path.join(path, ext))
                if images:
                    test_image = images[0]
                    break
            if test_image:
                break
    
    if not test_image:
        print("❌ No test image found. Please provide an image file.")
        print("Usage: python test_gatereader.py [image_path]")
        sys.exit(1)
    
    if len(sys.argv) > 1:
        test_image = sys.argv[1]
    
    print(f"Using test image: {test_image}")
    print()
    
    # Test extract endpoint
    print("2. Testing /extract endpoint...")
    extract_ok = test_extract_endpoint(test_image, base_url)
    print()
    
    # Test extract_with_crops endpoint
    print("3. Testing /extract_with_crops endpoint...")
    crops_ok = test_extract_with_crops_endpoint(test_image, base_url)
    print()
    
    # Summary
    print("=== Test Summary ===")
    print(f"Health check: {'✅' if health_ok else '❌'}")
    print(f"Extract endpoint: {'✅' if extract_ok else '❌'}")
    print(f"Extract with crops: {'✅' if crops_ok else '❌'}")
    
    if all([health_ok, extract_ok, crops_ok]):
        print("\n🎉 All tests passed! Gatereader API is working correctly.")
    else:
        print("\n❌ Some tests failed. Check the server logs for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()
