#!/usr/bin/env python3
"""
Test script to verify the TorchZScoreSustainMissingData fix.
"""

import numpy as np
import sys
import os

# Add the pySuStaIn directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pySuStaIn'))

def test_torch_zscore_sustain_missing_data():
    """Test that TorchZScoreSustainMissingData can be instantiated without attribute errors."""
    
    try:
        from pySuStaIn.TorchZScoreSustainMissingData import TorchZScoreSustainMissingData
        
        # Create minimal test data
        data = np.random.randn(10, 3)  # 10 subjects, 3 biomarkers
        Z_vals = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])  # 3 biomarkers, 3 thresholds
        Z_max = np.array([5, 5, 5])  # Max z-scores
        biomarker_labels = ['biomarker1', 'biomarker2', 'biomarker3']
        
        print("Creating TorchZScoreSustainMissingData instance...")
        
        # Try to create an instance
        sustain = TorchZScoreSustainMissingData(
            data=data,
            Z_vals=Z_vals,
            Z_max=Z_max,
            biomarker_labels=biomarker_labels,
            N_startpoints=1,
            N_S_max=1,
            N_iterations_MCMC=10,
            output_folder="./temp",
            dataset_name="test",
            use_parallel_startpoints=False,
            seed=42,
            use_gpu=False  # Use CPU to avoid GPU requirements
        )
        
        print("✓ TorchZScoreSustainMissingData instance created successfully!")
        
        # Test that the attribute is accessible
        print("Testing attribute access...")
        sustain_data = sustain._ZScoreSustainMissingData__sustainData
        print(f"✓ __sustainData attribute accessible: {type(sustain_data)}")
        
        # Test that we can get the number of stages
        num_stages = sustain_data.getNumStages()
        print(f"✓ Number of stages: {num_stages}")
        
        # Test that we can get the data
        data_shape = sustain_data.data.shape
        print(f"✓ Data shape: {data_shape}")
        
        print("\n🎉 All tests passed! The attribute access issue has been fixed.")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_torch_zscore_sustain_missing_data()
    sys.exit(0 if success else 1)
