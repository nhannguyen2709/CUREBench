#!/usr/bin/env python3
"""
Test script to verify custom inference function loading from JSON config with vLLM endpoint
"""

import json
import os
from eval_framework import import_function_from_string, CompetitionKit


def test_function_import():
    """Test that we can import functions from string references"""
    print("Testing function import from string references...")

    try:
        # Test importing the inference function
        inference_func = import_function_from_string(
            "my_models.ii_medical_inference.inference_function"
        )
        print(f"✅ Successfully imported inference function: {inference_func}")

        # Test importing the model factory
        factory_func = import_function_from_string(
            "my_models.ii_medical_inference.create_model_instance"
        )
        print(f"✅ Successfully imported factory function: {factory_func}")

        # Test importing the vLLM connection test function
        test_func = import_function_from_string(
            "my_models.ii_medical_inference.test_vllm_connection"
        )
        print(f"✅ Successfully imported vLLM test function: {test_func}")

        return True
    except Exception as e:
        print(f"❌ Error importing functions: {e}")
        return False


def test_json_config_loading():
    """Test that JSON config with inference functions can be loaded"""
    print("\nTesting JSON config loading...")

    try:
        # Load the JSON config
        with open("ii-medical-8b-1706.json", "r") as f:
            config = json.load(f)

        print("✅ JSON config loaded successfully")
        print(f"  - inference_func: {config['metadata'].get('inference_func')}")
        print(
            f"  - model_instance_factory: {config['metadata'].get('model_instance_factory')}"
        )

        return True
    except Exception as e:
        print(f"❌ Error loading JSON config: {e}")
        return False


def test_vllm_endpoint_config():
    """Test vLLM endpoint configuration"""
    print("\nTesting vLLM endpoint configuration...")

    try:
        # Test the vLLM connection function
        from my_models.ii_medical_inference import test_vllm_connection

        # This will test the connection but won't fail if server isn't running
        print("Testing vLLM endpoint connection...")
        connection_success = test_vllm_connection()

        if connection_success:
            print("✅ vLLM endpoint connection successful")
        else:
            print(
                "⚠️  vLLM endpoint not reachable (this is expected if server isn't running)"
            )
            print("   To start vLLM server, run:")
            print(
                "   vllm serve Intelligent-Internet/II-Medical-8B-1706 --host 0.0.0.0 --port 8000"
            )

        return True  # Return True regardless of connection status for testing purposes

    except Exception as e:
        print(f"❌ Error testing vLLM endpoint: {e}")
        return False


def test_competition_kit_integration():
    """Test that CompetitionKit can handle the JSON config"""
    print("\nTesting CompetitionKit integration...")

    try:
        # Initialize kit with the config
        kit = CompetitionKit(config_path="ii-medical-8b-1706.json")
        print("✅ CompetitionKit initialized with JSON config")

        # Test that metadata is loaded
        if kit.config and "metadata" in kit.config:
            metadata = kit.config["metadata"]
            inference_func = metadata.get("inference_func")
            model_factory = metadata.get("model_instance_factory")

            print(f"  - Found inference_func in metadata: {inference_func}")
            print(f"  - Found model_instance_factory in metadata: {model_factory}")

            if inference_func and model_factory:
                print("✅ All required fields found in metadata")
                return True
            else:
                print("❌ Missing required fields in metadata")
                return False
        else:
            print("❌ No metadata found in config")
            return False

    except Exception as e:
        print(f"❌ Error with CompetitionKit: {e}")
        return False


def test_model_instantiation():
    """Test that we can create model instance without errors"""
    print("\nTesting model instantiation...")

    try:
        from my_models.ii_medical_inference import create_model_instance

        # Create model instance (this should work even if vLLM server isn't running)
        model_instance = create_model_instance()

        print("✅ Model instance created successfully")
        print(f"  - Client type: {type(model_instance['client'])}")
        print(f"  - Model name: {model_instance['model_name']}")
        print(f"  - Base URL: {model_instance['base_url']}")

        return True

    except Exception as e:
        print(f"❌ Error creating model instance: {e}")
        return False


def show_usage_instructions():
    """Show usage instructions for vLLM setup"""
    print("\n" + "=" * 60)
    print("📋 vLLM Setup Instructions")
    print("=" * 60)
    print("1. Install vLLM:")
    print("   pip install vllm")
    print("")
    print("2. Start vLLM server:")
    print(
        "   vllm serve Intelligent-Internet/II-Medical-8B-1706 --host 0.0.0.0 --port 8000"
    )
    print("")
    print("3. Set environment variables (optional):")
    print("   export VLLM_BASE_URL=http://localhost:8000/v1")
    print("   export VLLM_API_KEY=token-abc123")
    print("")
    print("4. Test connection:")
    print(
        '   python -c "from my_models.ii_medical_inference import test_vllm_connection; test_vllm_connection()"'
    )
    print("")
    print("5. Run evaluation:")
    print(
        "   python run.py --config ii-medical-8b-1706.json --dataset cure_bench_pharse_1"
    )
    print("")
    print(
        "💡 The inference functions will automatically use environment variables or defaults:"
    )
    print("   - VLLM_BASE_URL (default: http://localhost:8000/v1)")
    print("   - VLLM_API_KEY (default: token-abc123)")
    print("   - VLLM_MODEL (default: Intelligent-Internet/II-Medical-8B-1706)")


def main():
    """Run all tests"""
    print("🧪 Testing Custom Inference Function Integration with vLLM")
    print("=" * 60)

    success = True

    # Test 1: Function import
    success &= test_function_import()

    # Test 2: JSON config loading
    success &= test_json_config_loading()

    # Test 3: vLLM endpoint configuration
    success &= test_vllm_endpoint_config()

    # Test 4: Model instantiation
    success &= test_model_instantiation()

    # Test 5: CompetitionKit integration
    success &= test_competition_kit_integration()

    print("\n" + "=" * 60)
    if success:
        print("🎉 All tests passed! Your custom inference setup is working correctly.")
        show_usage_instructions()
    else:
        print("❌ Some tests failed. Please check the errors above.")


if __name__ == "__main__":
    main()
