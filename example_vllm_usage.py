#!/usr/bin/env python3
"""
Example script demonstrating vLLM-based inference for CURE-Bench
"""

import os
from my_models.ii_medical_inference import (
    create_model_instance,
    inference_function,
    test_vllm_connection
)


def main():
    print("🩺 CURE-Bench vLLM Inference Example")
    print("=" * 40)
    
    # Step 1: Test vLLM connection
    print("1. Testing vLLM connection...")
    if test_vllm_connection():
        print("   ✅ vLLM server is reachable!")
    else:
        print("   ❌ vLLM server is not reachable.")
        print("   Please start vLLM server first:")
        print("   vllm serve Intelligent-Internet/II-Medical-8B-1706 --host 0.0.0.0 --port 8000")
        return
    
    # Step 2: Create model instance
    print("\n2. Creating model instance...")
    try:
        model_instance = create_model_instance()
        print(f"   ✅ Model instance created!")
        print(f"   Base URL: {model_instance['base_url']}")
        print(f"   Model: {model_instance['model_name']}")
    except Exception as e:
        print(f"   ❌ Failed to create model instance: {e}")
        return
    
    # Step 3: Test inference with example medical question
    print("\n3. Testing inference with example medical question...")
    
    # Example medical question
    question = "A 45-year-old patient presents with chest pain and shortness of breath. ECG shows ST-elevation in leads V1-V4. What is the most likely diagnosis?"
    
    options = [
        "Anterior myocardial infarction",
        "Posterior myocardial infarction", 
        "Pulmonary embolism",
        "Pneumothorax"
    ]
    
    print(f"   Question: {question}")
    print("   Options:")
    for i, option in enumerate(options):
        print(f"     {chr(65+i)}. {option}")
    
    try:
        # Call inference function
        predicted_answer = inference_function(model_instance, question, options)
        print(f"\n   🎯 Predicted Answer: {predicted_answer}")
        
        # Show which option was selected
        option_index = ord(predicted_answer) - 65
        if 0 <= option_index < len(options):
            selected_option = options[option_index]
            print(f"   📋 Selected Option: {predicted_answer}. {selected_option}")
        
    except Exception as e:
        print(f"   ❌ Inference failed: {e}")
        return
    
    print("\n✅ Example completed successfully!")
    print("\n💡 To run full evaluation:")
    print("   python run.py --config ii-medical-8b-1706.json --dataset cure_bench_pharse_1")


if __name__ == "__main__":
    main() 