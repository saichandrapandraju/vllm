#!/usr/bin/env python3
"""
Simple test for hidden states with Llama-2-7b-chat-hf model.
"""

import sys
import os

# Add vllm to the path
sys.path.insert(0, '/Users/spandraj/dev/vllm')

from vllm import LLM, SamplingParams


def test_llama_hidden_states():
    """Test hidden states with Llama-2-7b-chat-hf."""
    print("🦙 Testing hidden states with Llama-2-7b-chat-hf")
    print("=" * 50)
    
    # Load the model
    print("Loading model...")
    llm = LLM(
        model="meta-llama/Llama-2-7b-chat-hf",
        max_model_len=512,
        gpu_memory_utilization=0.5,
    )
    print("✅ Model loaded")
    
    # Test prompt
    prompt = "The future of AI is"
    
    # Test 1: Without hidden states (baseline)
    print(f"\n📝 Test 1: Baseline (no hidden states)")
    print(f"Prompt: {prompt}")
    
    params = SamplingParams(
        max_tokens=10,
        temperature=0.0,
    )
    
    outputs = llm.generate([prompt], params)
    output = outputs[0].outputs[0]
    
    print(f"Generated: {output.text}")
    print(f"Hidden states: {output.hidden_states}")
    
    # Test 2: With hidden states (last layer only)
    print(f"\n📝 Test 2: Hidden states - last layer only")
    
    params = SamplingParams(
        max_tokens=10,
        temperature=0.0,
        return_hidden_states=True,  # Our new parameter
    )
    
    outputs = llm.generate([prompt], params)
    output = outputs[0].outputs[0]
    
    print(f"Generated: {output.text}")
    
    if output.hidden_states:
        print(f"Hidden states available: {list(output.hidden_states.keys())}")
        for layer_idx, states in output.hidden_states.items():
            print(f"  Layer {layer_idx}: {len(states)} values")
            print(f"    Sample values: {states[:5]}...")
        print("✅ Hidden states returned successfully!")
    else:
        print("❌ No hidden states returned")
    
    # Test 3: Multiple specific layers
    print(f"\n📝 Test 3: Hidden states - multiple layers")
    
    params = SamplingParams(
        max_tokens=10,
        temperature=0.0,
        return_hidden_states=True,
        hidden_states_layers=[0, 15, -1],  # First, middle, last
    )
    
    outputs = llm.generate([prompt], params)
    output = outputs[0].outputs[0]
    
    print(f"Generated: {output.text}")
    print(f"Requested layers: [0, 15, -1]")
    
    if output.hidden_states:
        print(f"Returned layers: {list(output.hidden_states.keys())}")
        for layer_idx, states in output.hidden_states.items():
            print(f"  Layer {layer_idx}: {len(states)} values")
        print("✅ Multiple layer hidden states returned!")
    else:
        print("❌ No hidden states returned")
    
    print(f"\n🎉 Test completed!")


def main():
    """Run the test."""
    try:
        test_llama_hidden_states()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests passed!")
        print("\n📚 Usage in your code:")
        print("""
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-2-7b-chat-hf")

# Get last layer hidden states
params = SamplingParams(
    max_tokens=20,
    return_hidden_states=True
)

# Get specific layers
params = SamplingParams(
    max_tokens=20,
    return_hidden_states=True,
    hidden_states_layers=[0, 15, -1]
)

outputs = llm.generate(["Hello world"], params)
hidden_states = outputs[0].outputs[0].hidden_states
""")
    else:
        print("❌ Tests failed")
    
    sys.exit(0 if success else 1)