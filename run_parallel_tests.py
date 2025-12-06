"""
Main execution script for Task 3: Parallel Matrix Multiplication
"""

import sys
import time
from datetime import datetime
import numpy as np

from parallel_matrix_mult import ParallelMatrixMultiplication
from parallel_visualization import ParallelVisualizer


def print_header(title):
    """Print formatted header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")


def main():
    """Main execution function"""
    
    start_time = time.time()
    
    print_header("TASK 3: PARALLEL & VECTORIZED MATRIX MULTIPLICATION")
    print(f"Execution started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Author: Yaín René Estrada Domínguez")
    print(f"Course: Big Data - Universidad de Las Palmas de Gran Canaria")
    
    # Initialize benchmark
    benchmark = ParallelMatrixMultiplication()
    
    # =========================================================================
    # PART 1: THREAD SCALING ANALYSIS
    # =========================================================================
    
    print_header("PART 1: THREAD SCALING ANALYSIS")
    print("Testing how performance scales with number of threads...")
    print("-" * 80)
    
    # Test with different matrix sizes
    for size in [500, 1000]:
        print(f"\n>>> Matrix size: {size}x{size}")
        scaling_results = benchmark.thread_scaling_analysis(n=size)
    
    print("\n✓ Thread scaling analysis completed")
    
    # =========================================================================
    # PART 2: ALGORITHM COMPARISON
    # =========================================================================
    
    print_header("PART 2: COMPREHENSIVE ALGORITHM COMPARISON")
    print("Comparing all parallel and vectorized approaches...")
    print("-" * 80)
    
    comparison_results = benchmark.compare_all_approaches(
        sizes=[100, 500, 1000, 1500],
        num_threads=4
    )
    
    print("\n✓ Algorithm comparison completed")
    
    # =========================================================================
    # PART 3: LARGE MATRIX TESTS
    # =========================================================================
    
    print_header("PART 3: LARGE MATRIX PERFORMANCE")
    print("Testing with larger matrices...")
    print("-" * 80)
    
    large_sizes = [2000, 3000]
    
    for n in large_sizes:
        print(f"\n>>> Testing {n}x{n} matrices")
        A = np.random.rand(n, n)
        B = np.random.rand(n, n)
        
        # Sequential baseline
        print("  Sequential NumPy (baseline)...")
        baseline = benchmark.benchmark_algorithm('sequential_numpy', A, B)
        
        # Parallel with threads
        print(f"  Parallel with {benchmark.num_cores} threads...")
        parallel = benchmark.benchmark_algorithm(
            'parallel_threads', A, B, 
            num_threads=benchmark.num_cores,
            baseline_time=baseline.execution_time
        )
        
        # NumPy parallel BLAS
        print("  NumPy parallel BLAS...")
        numpy_parallel = benchmark.benchmark_algorithm(
            'numpy_parallel_blas', A, B,
            baseline_time=baseline.execution_time
        )
        
        print(f"\n  Results:")
        print(f"    Sequential: {baseline.execution_time:.4f}s, {baseline.gflops:.2f} GFlop/s")
        print(f"    Parallel Threads: {parallel.execution_time:.4f}s, {parallel.gflops:.2f} GFlop/s, Speedup: {parallel.speedup:.2f}x")
        print(f"    NumPy BLAS: {numpy_parallel.execution_time:.4f}s, {numpy_parallel.gflops:.2f} GFlop/s, Speedup: {numpy_parallel.speedup:.2f}x")
    
    print("\n✓ Large matrix tests completed")
    
    # =========================================================================
    # PART 4: SAVE AND VISUALIZE
    # =========================================================================
    
    print_header("PART 4: SAVING RESULTS AND GENERATING VISUALIZATIONS")
    
    # Save results
    benchmark.save_results()
    benchmark.print_summary()
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    visualizer = ParallelVisualizer()
    visualizer.generate_all_plots()
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    
    print_header("EXECUTION SUMMARY")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n📊 Results saved in:")
    print("   - results/parallel_results.json")
    
    print("\n📈 Figures saved in:")
    print("   - figures/thread_scaling.png")
    print("   - figures/algorithm_comparison_parallel.png")
    print("   - figures/amdahl_law.png")
    print("   - figures/parallel_summary.txt")
    
    print("\n" + "="*80)
    print("  ✓ ALL TESTS COMPLETED SUCCESSFULLY")
    print("="*80 + "\n")
    
    # Print key findings
    print_header("KEY FINDINGS")
    
    if benchmark.results:
        # Best parallel performance
        parallel_results = [r for r in benchmark.results if r.num_threads > 1]
        
        if parallel_results:
            best = max(parallel_results, key=lambda x: x.speedup)
            print(f"🏆 Best Parallel Speedup:")
            print(f"   - Algorithm: {best.algorithm}")
            print(f"   - Matrix size: {best.matrix_size}x{best.matrix_size}")
            print(f"   - Threads: {best.num_threads}")
            print(f"   - Speedup: {best.speedup:.2f}x")
            print(f"   - Efficiency: {best.efficiency*100:.1f}%")
            print(f"   - Performance: {best.gflops:.2f} GFlop/s")
        
        # Average speedup
        avg_speedup = np.mean([r.speedup for r in parallel_results])
        print(f"\n📊 Average Speedup: {avg_speedup:.2f}x")
        
        # Scalability insights
        print(f"\n💡 Scalability Insights:")
        print(f"   - System has {benchmark.num_cores} CPU cores")
        print(f"   - Parallel implementations tested with 1-{benchmark.num_cores} threads")
        print(f"   - Best efficiency achieved: {max([r.efficiency for r in parallel_results])*100:.1f}%")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
