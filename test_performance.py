#!/usr/bin/env python3
"""
Test script to generate performance tracking data for the Cat-Dog classifier
"""

import requests
import json
import time
import os
from PIL import Image
import numpy as np

def create_synthetic_images():
    """Create synthetic test images for performance testing"""
    test_dir = "test_images"
    os.makedirs(test_dir, exist_ok=True)
    
    # Create synthetic cat-like image (darker with patterns)
    cat_image = np.zeros((224, 224, 3), dtype=np.uint8)
    cat_image[:100, :100] = [60, 40, 20]  # Dark brown patches
    cat_image[50:150, 50:150] = [80, 60, 40]  # Lighter brown
    cat_image = Image.fromarray(cat_image)
    cat_image.save(f"{test_dir}/synthetic_cat.jpg")
    
    # Create synthetic dog-like image (brighter with different patterns)
    dog_image = np.zeros((224, 224, 3), dtype=np.uint8)
    dog_image[:, :] = [120, 90, 60]  # Base brown
    dog_image[75:149, 75:149] = [140, 110, 80]  # Lighter center
    dog_image = Image.fromarray(dog_image)
    dog_image.save(f"{test_dir}/synthetic_dog.jpg")
    
    # Create ambiguous image
    ambiguous_image = np.ones((224, 224, 3), dtype=np.uint8) * 100
    ambiguous_image = Image.fromarray(ambiguous_image)
    ambiguous_image.save(f"{test_dir}/synthetic_ambiguous.jpg")
    
    return [
        (f"{test_dir}/synthetic_cat.jpg", "Cat"),
        (f"{test_dir}/synthetic_dog.jpg", "Dog"), 
        (f"{test_dir}/synthetic_ambiguous.jpg", "Cat")  # Intentionally wrong label for testing
    ]

def test_api_with_performance_tracking(base_url="http://192.168.49.2:30080"):
    """Test the API with performance tracking"""
    print(f"Testing API at {base_url}")
    
    # Create test images
    test_images = create_synthetic_images()
    
    results = []
    
    print("\n=== Testing Health Endpoint ===")
    try:
        response = requests.get(f"{base_url}/health")
        print(f"Health check: {response.status_code}")
        if response.status_code == 200:
            health_data = response.json()
            print(f"Model status: {health_data.get('model_status')}")
            print(f"Total requests: {health_data.get('uptime_stats', {}).get('total_requests', 0)}")
    except Exception as e:
        print(f"Health check failed: {e}")
        return
    
    print("\n=== Testing Predictions with True Labels ===")
    for image_path, true_label in test_images:
        try:
            print(f"\nTesting {image_path} (true label: {true_label})")
            
            with open(image_path, 'rb') as f:
                files = {'file': f}
                data = {'true_label': true_label}
                
                start_time = time.time()
                response = requests.post(f"{base_url}/predict", files=files, data=data)
                end_time = time.time()
                
                if response.status_code == 200:
                    result = response.json()
                    latency = (end_time - start_time) * 1000
                    
                    print(f"✓ Prediction: {result['predicted_class']}")
                    print(f"  Confidence: {result['confidence']:.3f}")
                    print(f"  Request ID: {result['request_id']}")
                    print(f"  Processing time: {result['processing_time_ms']:.2f}ms")
                    print(f"  Total latency: {latency:.2f}ms")
                    print(f"  Correct: {'Yes' if result['predicted_class'] == true_label else 'No'}")
                    
                    results.append({
                        'image': os.path.basename(image_path),
                        'true_label': true_label,
                        'predicted': result['predicted_class'],
                        'confidence': result['confidence'],
                        'correct': result['predicted_class'] == true_label,
                        'processing_time_ms': result['processing_time_ms'],
                        'total_latency_ms': latency
                    })
                else:
                    print(f"✗ Error: {response.status_code} - {response.text}")
                    
        except Exception as e:
            print(f"✗ Error testing {image_path}: {e}")
    
    print("\n=== Testing Metrics Endpoint ===")
    try:
        response = requests.get(f"{base_url}/metrics")
        if response.status_code == 200:
            metrics_data = response.json()
            print("✓ Metrics retrieved successfully:")
            print(f"  Total requests: {sum(metrics_data['metrics']['total_requests'].values())}")
            print(f"  Average latency: {metrics_data['metrics']['latency_stats']['avg_latency_ms']:.2f}ms")
            print(f"  P95 latency: {metrics_data['metrics']['latency_stats']['p95_latency_ms']:.2f}ms")
            print(f"  Total predictions: {metrics_data['metrics']['prediction_stats']['total_predictions']}")
        else:
            print(f"✗ Metrics error: {response.status_code}")
    except Exception as e:
        print(f"✗ Error retrieving metrics: {e}")
    
    print("\n=== Testing Performance Endpoint ===")
    try:
        response = requests.get(f"{base_url}/performance")
        if response.status_code == 200:
            perf_data = response.json()
            print("✓ Performance data retrieved successfully:")
            print(f"  Total predictions: {perf_data['total_predictions']}")
            print(f"  Predictions with labels: {perf_data['predictions_with_labels']}")
            if perf_data['accuracy'] is not None:
                print(f"  Accuracy: {perf_data['accuracy']:.3f} ({perf_data['accuracy']*100:.1f}%)")
            print(f"  Average confidence: {perf_data['performance_summary']['avg_confidence']:.3f}")
            print(f"  Average processing time: {perf_data['performance_summary']['avg_processing_time_ms']:.2f}ms")
        else:
            print(f"✗ Performance error: {response.status_code}")
    except Exception as e:
        print(f"✗ Error retrieving performance data: {e}")
    
    print("\n=== Summary ===")
    if results:
        accuracy = sum(1 for r in results if r['correct']) / len(results)
        avg_confidence = sum(r['confidence'] for r in results) / len(results)
        avg_processing_time = sum(r['processing_time_ms'] for r in results) / len(results)
        
        print(f"Test Results:")
        print(f"  Images tested: {len(results)}")
        print(f"  Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        print(f"  Average confidence: {avg_confidence:.3f}")
        print(f"  Average processing time: {avg_processing_time:.2f}ms")
        
        print(f"\nDetailed Results:")
        for r in results:
            status = "✓" if r['correct'] else "✗"
            print(f"  {status} {r['image']}: {r['predicted']} (conf: {r['confidence']:.3f}, time: {r['processing_time_ms']:.1f}ms)")
    
    # Clean up test images
    import shutil
    if os.path.exists("test_images"):
        shutil.rmtree("test_images")
    
    print("\n🎉 Performance testing completed!")

if __name__ == "__main__":
    import sys
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://192.168.49.2:30080"
    test_api_with_performance_tracking(base_url)