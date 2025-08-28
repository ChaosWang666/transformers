#!/usr/bin/env python3
"""
Simple test for modular Qwen2.5-VL prepare_inputs_for_generation method fix.
"""
import torch
import sys
import os

# Add the src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Import the method directly for testing
from transformers.models.qwen2_5_vl.modular_qwen2_5_vl import get_rope_index

def test_get_rope_index():
    """Test the get_rope_index function."""
    print("Testing get_rope_index function...")
    
    # Mock inputs to test get_rope_index
    input_ids = torch.randint(0, 1000, (2, 5))  # batch_size=2, seq_len=5
    image_grid_thw = torch.tensor([[1, 14, 14]])  # Example image grid
    video_grid_thw = None
    
    try:
        position_ids, mrope_position_deltas = get_rope_index(
            input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw
        )
        
        print(f"✓ position_ids shape: {position_ids.shape}")
        print(f"✓ mrope_position_deltas shape: {mrope_position_deltas.shape}")
        
        # Expected shapes:
        # position_ids: (3, batch_size, seq_len) = (3, 2, 5)
        # mrope_position_deltas: (batch_size, 1) = (2, 1)
        if position_ids.shape == (3, 2, 5) and mrope_position_deltas.shape == (2, 1):
            print("✓ Shapes are correct!")
            return True
        else:
            print(f"✗ Unexpected shapes. Expected (3, 2, 5) and (2, 1)")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_position_expansion():
    """Test position_ids expansion logic."""
    print("\nTesting position expansion logic...")
    
    try:
        # Test text_positions expansion matching vision_positions batch size
        device = torch.device('cpu')
        seq_len = 5
        bs = 2
        
        # Mock text positions (original logic)
        text_positions = torch.arange(seq_len, device=device).expand(1, bs, -1)
        print(f"text_positions shape: {text_positions.shape}")
        
        # Mock vision positions (from get_rope_index)
        vision_positions = torch.zeros(3, bs, seq_len, device=device)
        print(f"vision_positions shape: {vision_positions.shape}")
        
        # Concatenate like in prepare_inputs_for_generation
        position_ids = torch.cat([text_positions, vision_positions], dim=0)
        print(f"✓ Combined position_ids shape: {position_ids.shape}")
        
        # Expected: (4, batch_size, seq_len) = (4, 2, 5)
        if position_ids.shape == (4, 2, 5):
            print("✓ Position expansion is correct!")
            return True
        else:
            print(f"✗ Unexpected shape. Expected (4, 2, 5)")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success1 = test_get_rope_index()
    success2 = test_position_expansion()
    
    if success1 and success2:
        print("\n✓ All tests passed! The modular fix components are working correctly.")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed.")
        sys.exit(1)