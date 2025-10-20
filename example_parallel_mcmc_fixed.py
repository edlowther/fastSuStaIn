#!/usr/bin/env python3
"""
Example: Parallel MCMC for SuStaIn using existing use_parallel_startpoints argument.

This example demonstrates how to use the parallel MCMC functionality while
avoiding the dill compatibility issues with Python 3.11.
"""

import numpy as np
import time
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def create_sample_data(n_subjects=100, n_biomarkers=5):
    """Create sample data for testing."""
    np.random.seed(42)
    
    # Generate synthetic z-score data
    data = np.random.randn(n_subjects, n_biomarkers)
    
    # Create z-score thresholds (3 thresholds per biomarker)
    Z_vals = np.array([[1, 2, 3] for _ in range(n_biomarkers)])
    Z_max = np.array([5.0] * n_biomarkers)
    
    # Create biomarker labels
    biomarker_labels = [f'biomarker_{i}' for i in range(n_biomarkers)]
    
    return data, Z_vals, Z_max, biomarker_labels

def example_parallel_mcmc_basic():
    """Basic example of parallel MCMC usage."""
    print("=== Basic Parallel MCMC Example ===")
    
    # Create sample data
    data, Z_vals, Z_max, biomarker_labels = create_sample_data(n_subjects=50, n_biomarkers=3)
    print(f"Created sample data: {data.shape[0]} subjects, {data.shape[1]} biomarkers")
    
    try:
        # Import the parallel components
        from pySuStaIn.parallel_mcmc import ParallelMCMCManager, combine_mcmc_results
        
        # Create a parallel manager
        manager = ParallelMCMCManager(
            n_chains=4,
            n_workers=2,
            backend='thread',  # Use thread backend to avoid dill issues
            use_gpu=False
        )
        
        print(f"Created parallel manager: {manager.n_chains} chains, {manager.n_workers} workers")
        
        # Simulate MCMC results (in real usage, these would come from actual MCMC runs)
        print("\nSimulating MCMC results...")
        
        # Create mock results for demonstration
        samples_sequences = []
        samples_fs = []
        samples_likelihoods = []
        chain_times = []
        
        for i in range(manager.n_chains):
            # Simulate different chain results
            np.random.seed(42 + i)
            n_iterations = 100
            
            # Mock sequence samples (N_S, N, n_iterations)
            seq_samples = np.random.rand(1, 3, n_iterations)
            samples_sequences.append(seq_samples)
            
            # Mock fraction samples (N_S, n_iterations)
            f_samples = np.random.rand(1, n_iterations)
            samples_fs.append(f_samples)
            
            # Mock likelihood samples (n_iterations,)
            likelihood_samples = np.random.rand(n_iterations)
            samples_likelihoods.append(likelihood_samples)
            
            # Mock chain time
            chain_times.append(1.0 + i * 0.1)
        
        # Combine results
        combined_sequences, combined_fs, combined_likelihoods, stats = combine_mcmc_results(
            samples_sequences, samples_fs, samples_likelihoods, chain_times
        )
        
        print("✓ Successfully combined MCMC results")
        print(f"  - Combined sequences shape: {combined_sequences.shape}")
        print(f"  - Combined fs shape: {combined_fs.shape}")
        print(f"  - Combined likelihoods shape: {combined_likelihoods.shape}")
        print(f"  - Total iterations: {stats['total_iterations']}")
        print(f"  - Speedup: {stats['speedup']:.2f}x")
        print(f"  - Efficiency: {stats['efficiency']:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic example failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def example_parallel_mcmc_integration():
    """Example of integrating parallel MCMC with existing SuStaIn code."""
    print("\n=== Parallel MCMC Integration Example ===")
    
    try:
        # Import the integration function
        from pySuStaIn.parallel_mcmc import integrate_parallel_mcmc_with_sustain
        
        print("✓ Successfully imported integration function")
        
        # Create a mock SuStaIn instance for demonstration
        class MockSuStaIn:
            def __init__(self):
                self.N_iterations_MCMC = 1000
                self.use_gpu = False
                self.global_rng = np.random.default_rng(42)
            
            def _optimise_mcmc_settings(self, sustainData, seq_init, f_init):
                # Mock optimization
                return 0.1, 0.01
            
            def _estimate_uncertainty_sustain_model(self, sustainData, seq_init, f_init):
                # Mock uncertainty estimation
                n_s = seq_init.shape[0]
                n = seq_init.shape[1]
                n_iterations = 100
                
                samples_sequence = np.random.rand(n_s, n, n_iterations)
                samples_f = np.random.rand(n_s, n_iterations)
                samples_likelihood = np.random.rand(n_iterations)
                
                best_idx = np.argmax(samples_likelihood)
                ml_sequence = samples_sequence[:, :, best_idx]
                ml_f = samples_f[:, best_idx]
                ml_likelihood = samples_likelihood[best_idx]
                
                return ml_sequence, ml_f, ml_likelihood, samples_sequence, samples_f, samples_likelihood
            
            def _perform_mcmc(self, sustainData, seq_init, f_init, n_iterations, seq_sigma, f_sigma):
                # Mock MCMC run
                n_s = seq_init.shape[0]
                n = seq_init.shape[1]
                
                # Generate mock samples
                samples_sequence = np.random.rand(n_s, n, n_iterations)
                samples_f = np.random.rand(n_s, n_iterations)
                samples_likelihood = np.random.rand(n_iterations)
                
                # Find best result
                best_idx = np.argmax(samples_likelihood)
                ml_sequence = samples_sequence[:, :, best_idx]
                ml_f = samples_f[:, best_idx]
                ml_likelihood = samples_likelihood[best_idx]
                
                return ml_sequence, ml_f, ml_likelihood, samples_sequence, samples_f, samples_likelihood
        
        # Create mock instance
        sustain_instance = MockSuStaIn()
        
        # Integrate parallel MCMC
        integrate_parallel_mcmc_with_sustain(
            sustain_instance,
            use_parallel_mcmc=True,
            n_mcmc_chains=4,
            mcmc_backend='thread'
        )
        
        print("✓ Successfully integrated parallel MCMC")
        print(f"  - Parallel manager created: {hasattr(sustain_instance, 'parallel_mcmc_manager')}")
        print(f"  - Method overridden: {sustain_instance._estimate_uncertainty_sustain_model != MockSuStaIn._estimate_uncertainty_sustain_model}")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration example failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def example_benchmarking():
    """Example of benchmarking parallel MCMC performance."""
    print("\n=== Parallel MCMC Benchmarking Example ===")
    
    try:
        from pySuStaIn.parallel_mcmc import benchmark_parallel_mcmc
        
        # Create mock SuStaIn instance
        class MockSuStaIn:
            def __init__(self):
                self.N_iterations_MCMC = 100
                self.use_gpu = False
                self.global_rng = np.random.default_rng(42)
            
            def _optimise_mcmc_settings(self, sustainData, seq_init, f_init):
                return 0.1, 0.01
            
            def _perform_mcmc(self, sustainData, seq_init, f_init, n_iterations, seq_sigma, f_sigma):
                # Mock MCMC run with some computation
                time.sleep(0.01)  # Simulate computation
                
                n_s = seq_init.shape[0]
                n = seq_init.shape[1]
                
                samples_sequence = np.random.rand(n_s, n, n_iterations)
                samples_f = np.random.rand(n_s, n_iterations)
                samples_likelihood = np.random.rand(n_iterations)
                
                best_idx = np.argmax(samples_likelihood)
                ml_sequence = samples_sequence[:, :, best_idx]
                ml_f = samples_f[:, best_idx]
                ml_likelihood = samples_likelihood[best_idx]
                
                return ml_sequence, ml_f, ml_likelihood, samples_sequence, samples_f, samples_likelihood
        
        # Create mock data
        sustain_instance = MockSuStaIn()
        sustain_data = None  # Mock data object
        seq_init = np.random.rand(1, 3)
        f_init = np.array([1.0])
        
        # Run benchmark
        print("Running benchmark...")
        results = benchmark_parallel_mcmc(
            sustain_instance, sustain_data, seq_init, f_init,
            n_iterations=50,  # Small number for quick test
            n_chains_list=[1, 2, 4]
        )
        
        print("✓ Benchmark completed")
        for n_chains, result in results.items():
            print(f"  - {n_chains} chains: {result['total_time']:.2f}s, "
                  f"speedup: {result['speedup']:.2f}x, "
                  f"efficiency: {result['efficiency']:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Benchmarking example failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all examples."""
    print("🚀 Parallel MCMC Examples for SuStaIn")
    print("=" * 50)
    
    # Run examples
    examples = [
        ("Basic Parallel MCMC", example_parallel_mcmc_basic),
        ("Integration Example", example_parallel_mcmc_integration),
        ("Benchmarking Example", example_benchmarking)
    ]
    
    results = []
    for name, func in examples:
        print(f"\n{'='*20} {name} {'='*20}")
        try:
            success = func()
            results.append((name, success))
        except Exception as e:
            print(f"❌ {name} failed with exception: {e}")
            results.append((name, False))
    
    # Summary
    print(f"\n{'='*20} Summary {'='*20}")
    for name, success in results:
        status = "✓ PASSED" if success else "❌ FAILED"
        print(f"  {name}: {status}")
    
    all_passed = all(success for _, success in results)
    print(f"\nOverall: {'🎉 ALL EXAMPLES PASSED' if all_passed else '❌ SOME EXAMPLES FAILED'}")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
