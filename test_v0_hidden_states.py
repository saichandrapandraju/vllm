#!/usr/bin/env python3
"""
Test hidden states with V0 backend.
"""

import sys
import os

# Add vllm to the path
sys.path.insert(0, '/Users/spandraj/dev/vllm')

# Force V0 backend
os.environ['VLLM_USE_V1'] = '0'

from vllm import LLM, SamplingParams


def test_v0_hidden_states():
    """Test hidden states with V0 backend."""
    print("🔍 Testing V0 Backend Hidden States")
    print("=" * 50)
    
    print(f"VLLM_USE_V1: {os.environ.get('VLLM_USE_V1')}")
    
    # Load model
    print("Loading model with V0 backend...")
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
    
    # Test with hidden states
    params = SamplingParams(
        max_tokens=3,
        temperature=0.0,
        return_hidden_states=True,
        hidden_states_layers=[0, -1],
    )
    
    print(f"SamplingParams:")
    print(f"  return_hidden_states: {params.return_hidden_states}")
    print(f"  hidden_states_layers: {params.hidden_states_layers}")
    
    # Get model info
    model_layers = llm.llm_engine.model_config.hf_config.num_hidden_layers
    print(f"  Model layers: {model_layers}")
    
    resolved = params.resolve_hidden_states_layers(model_layers)
    print(f"  Resolved layers: {resolved}")
    
    print("\n🚀 Running inference...")
    outputs = llm.generate([prompt], params)
    output = outputs[0].outputs[0]
    
    print(f"\n📊 Results:")
    print(f"Generated: '{output.text}'")
    print(f"Hidden states: {output.hidden_states}")
    print(f"Hidden states type: {type(output.hidden_states)}")
    
    if output.hidden_states:
        print(f"Hidden states keys: {list(output.hidden_states.keys())}")
        for layer_idx, states in output.hidden_states.items():
            print(f"  Layer {layer_idx}: {len(states)} elements")
            if len(states) > 0:
                print(f"    First few values: {states[:5]}")
    
    return output.hidden_states is not None


if __name__ == "__main__":
    try:
        success = test_v0_hidden_states()
        print(f"\n{'✅' if success else '❌'} Test {'passed' if success else 'failed'}")
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)