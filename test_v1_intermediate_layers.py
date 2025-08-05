#!/usr/bin/env python3
"""
Test V1 backend with intermediate layer hidden states support.
"""

import sys
import os

# Add vllm to the path
sys.path.insert(0, '/Users/spandraj/dev/vllm')

# Force V1 backend
os.environ['VLLM_USE_V1'] = '1'

from vllm import LLM, SamplingParams


def test_v1_intermediate_layers():
    """Test V1 backend with intermediate layer hidden states support."""
    print("🔍 Testing V1 Backend with Intermediate Layer Hidden States")
    print("=" * 70)
    
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
    
    # Get model info
    model_layers = llm.llm_engine.model_config.hf_config.num_hidden_layers
    print(f"Model has {model_layers} layers (0 to {model_layers-1})")
    
    # Test 1: Request early layers only
    print("\n🧪 Test 1: Requesting early layers [0, 5, 10] (should work with V1 intermediate support)")
    params1 = SamplingParams(
        max_tokens=3,
        temperature=0.0,
        return_hidden_states=True,
        hidden_states_layers=[0, 5, 10],  # Early layers
    )
    
    print(f"SamplingParams: layers={params1.hidden_states_layers}")
    
    outputs1 = llm.generate([prompt], params1)
    output1 = outputs1[0].outputs[0]
    
    print(f"Result keys: {list(output1.hidden_states.keys()) if output1.hidden_states else None}")
    print(f"Success: {output1.hidden_states is not None}")
    if output1.hidden_states:
        for layer_idx, states in output1.hidden_states.items():
            print(f"  Layer {layer_idx}: {len(states)} values")
    
    # Test 2: Request mixed layers including final 
    print(f"\n🧪 Test 2: Requesting mixed layers [0, 15, {model_layers-1}] (should work)")
    params2 = SamplingParams(
        max_tokens=3,
        temperature=0.0,
        return_hidden_states=True,
        hidden_states_layers=[0, 15, model_layers-1],  # Mixed including final
    )
    
    print(f"SamplingParams: layers={params2.hidden_states_layers}")
    
    outputs2 = llm.generate([prompt], params2)
    output2 = outputs2[0].outputs[0]
    
    print(f"Result keys: {list(output2.hidden_states.keys()) if output2.hidden_states else None}")
    print(f"Success: {output2.hidden_states is not None}")
    if output2.hidden_states:
        for layer_idx, states in output2.hidden_states.items():
            print(f"  Layer {layer_idx}: {len(states)} values")
    
    # Test 3: Request final layer only (should definitely work)
    print(f"\n🧪 Test 3: Requesting final layer [{model_layers-1}] only (should work)")
    params3 = SamplingParams(
        max_tokens=3,
        temperature=0.0,
        return_hidden_states=True,
        hidden_states_layers=[model_layers-1],  # Final layer only
    )
    
    print(f"SamplingParams: layers={params3.hidden_states_layers}")
    
    outputs3 = llm.generate([prompt], params3)
    output3 = outputs3[0].outputs[0]
    
    print(f"Result keys: {list(output3.hidden_states.keys()) if output3.hidden_states else None}")
    print(f"Success: {output3.hidden_states is not None}")
    if output3.hidden_states:
        for layer_idx, states in output3.hidden_states.items():
            print(f"  Layer {layer_idx}: {len(states)} values")
    
    # Test 4: Negative indexing
    print("\n🧪 Test 4: Requesting with negative indexing [-1, -2] (should work)")
    params4 = SamplingParams(
        max_tokens=3,
        temperature=0.0,
        return_hidden_states=True,
        hidden_states_layers=[-1, -2],  # Negative indexing
    )
    
    print(f"SamplingParams: layers={params4.hidden_states_layers}")
    resolved = params4.resolve_hidden_states_layers(model_layers)
    print(f"Resolved layers: {resolved}")
    
    outputs4 = llm.generate([prompt], params4)
    output4 = outputs4[0].outputs[0]
    
    print(f"Result keys: {list(output4.hidden_states.keys()) if output4.hidden_states else None}")
    print(f"Success: {output4.hidden_states is not None}")
    if output4.hidden_states:
        for layer_idx, states in output4.hidden_states.items():
            print(f"  Layer {layer_idx}: {len(states)} values")
    
    # Count successful tests
    successes = []
    successes.append(output1.hidden_states is not None)
    successes.append(output2.hidden_states is not None)  
    successes.append(output3.hidden_states is not None)
    successes.append(output4.hidden_states is not None)
    
    success_count = sum(successes)
    total_tests = len(successes)
    
    print(f"\n📊 Results: {success_count}/{total_tests} tests passed")
    
    return success_count == total_tests


if __name__ == "__main__":
    try:
        success = test_v1_intermediate_layers()
        print(f"\n{'✅' if success else '❌'} Test {'passed' if success else 'failed'}")
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)