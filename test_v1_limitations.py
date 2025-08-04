#!/usr/bin/env python3
"""
Test V1 backend limitations - requesting intermediate layers only.
"""

import sys
import os

# Add vllm to the path
sys.path.insert(0, '/Users/spandraj/dev/vllm')

# Force V1 backend
os.environ['VLLM_USE_V1'] = '1'

from vllm import LLM, SamplingParams


def test_v1_intermediate_layers():
    """Test V1 backend with intermediate layers only (should fail gracefully)."""
    print("🔍 Testing V1 Backend Limitations - Intermediate Layers Only")
    print("=" * 60)
    
    print(f"VLLM_USE_V1: {os.environ.get('VLLM_USE_V1')}")
    
    # Load model
    print("Loading model with V1 backend...")
    llm = LLM(
        model="meta-llama/Llama-2-7b-chat-hf",
        max_model_len=128,
        gpu_memory_utilization=0.0,
        enforce_eager=True,
    )
    print("✅ Model loaded")
    
    # Test prompt
    prompt = "Hello world"
    
    print(f"\n📝 Testing with prompt: '{prompt}'")
    
    # Test with intermediate layers only (no final layer)
    print("\n🧪 Test 1: Requesting intermediate layers only [0, 15] (should warn and return None)")
    params1 = SamplingParams(
        max_tokens=3,
        temperature=0.0,
        return_hidden_states=True,
        hidden_states_layers=[0, 15],  # No final layer (31)
    )
    
    print(f"SamplingParams: layers={params1.hidden_states_layers}")
    
    outputs1 = llm.generate([prompt], params1)
    output1 = outputs1[0].outputs[0]
    
    print(f"Result: {output1.hidden_states}")
    print(f"Success: {output1.hidden_states is None}")
    
    print("\n🧪 Test 2: Requesting final layer [31] (should work)")
    params2 = SamplingParams(
        max_tokens=3,
        temperature=0.0,
        return_hidden_states=True,
        hidden_states_layers=[31],  # Final layer
    )
    
    print(f"SamplingParams: layers={params2.hidden_states_layers}")
    
    outputs2 = llm.generate([prompt], params2)
    output2 = outputs2[0].outputs[0]
    
    print(f"Result keys: {list(output2.hidden_states.keys()) if output2.hidden_states else None}")
    print(f"Success: {output2.hidden_states is not None}")
    
    print("\n🧪 Test 3: Requesting mixed layers [0, 15, 31] (should work, only return 31)")
    params3 = SamplingParams(
        max_tokens=3,
        temperature=0.0,
        return_hidden_states=True,
        hidden_states_layers=[0, 15, 31],  # Includes final layer
    )
    
    print(f"SamplingParams: layers={params3.hidden_states_layers}")
    
    outputs3 = llm.generate([prompt], params3)
    output3 = outputs3[0].outputs[0]
    
    print(f"Result keys: {list(output3.hidden_states.keys()) if output3.hidden_states else None}")
    print(f"Success: {output3.hidden_states is not None and 31 in output3.hidden_states}")
    
    return True


if __name__ == "__main__":
    try:
        success = test_v1_intermediate_layers()
        print(f"\n{'✅' if success else '❌'} Test {'completed' if success else 'failed'}")
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)