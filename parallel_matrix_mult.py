"""
Parallel Matrix Multiplication
Task 3 - Big Data Course
Author: Yaín René Estrada Domínguez
Universidad de Las Palmas de Gran Canaria
"""

import numpy as np
import time
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass, asdict
import os
import json


@dataclass
class ParallelResult:
    """Store benchmark results for parallel multiplication"""
    algorithm: str
    matrix_size: int
    num_threads: int
    execution_time: float
    gflops: float
    speedup: float
    efficiency: float
    baseline_time: float
    
    def to_dict(self):
        return asdict(self)


class ParallelMatrixMultiplication:
    """
    Parallel and Vectorized Matrix Multiplication Suite
    """
    
    def __init__(self, results_dir="results"):
        self.results = []
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        # Get number of available cores
        self.num_cores = mp.cpu_count()
        print(f"System has {self.num_cores} CPU cores available")
    
    @staticmethod
    def naive_sequential(A, B):
        """Sequential naive O(n³) multiplication - baseline"""
        n = A.shape[0]
        C = np.zeros((n, n), dtype=np.float64)
        
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    C[i, j] += A[i, k] * B[k, j]
        
        return C
    
    @staticmethod
    def numpy_sequential(A, B):
        """NumPy sequential (BLAS optimized) - baseline"""
        return np.dot(A, B)
    
    @staticmethod
    def _multiply_row_range(args):
        """Helper function for parallel row computation"""
        A, B, start_row, end_row = args
        n = A.shape[1]
        m = B.shape[1]
        C_partial = np.zeros((end_row - start_row, m), dtype=np.float64)
        
        for i in range(start_row, end_row):
            for j in range(m):
                for k in range(n):
                    C_partial[i - start_row, j] += A[i, k] * B[k, j]
        
        return start_row, C_partial
    
    def parallel_rows_threads(self, A, B, num_threads=None, timeout=30):
        """
        Parallel multiplication using Python threading
        Distributes rows across threads
        """
        if num_threads is None:
            num_threads = self.num_cores
        
        n, m = A.shape[0], B.shape[1]
        C = np.zeros((n, m), dtype=np.float64)
        
        print(f"    Starting parallel computation with {num_threads} threads...")
        
        # Calculate rows per thread
        rows_per_thread = n // num_threads
        
        # Prepare arguments for each thread
        tasks = []
        for i in range(num_threads):
            start_row = i * rows_per_thread
            end_row = (i + 1) * rows_per_thread if i < num_threads - 1 else n
            tasks.append((A, B, start_row, end_row))
        
        # Execute in parallel using threads with timeout
        try:
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                future = executor.map(self._multiply_row_range, tasks, timeout=timeout)
                results = list(future)
            
            print(f"    ✓ Computation completed")
            
            # Combine results
            for start_row, C_partial in results:
                end_row = start_row + C_partial.shape[0]
                C[start_row:end_row, :] = C_partial
            
            return C
            
        except TimeoutError:
            print(f"    ✗ Timeout after {timeout}s - skipping this test")
            return None
        except Exception as e:
            print(f"    ✗ Error: {e}")
            return None
    
    def parallel_rows_processes(self, A, B, num_processes=None):
        """
        Parallel multiplication using multiprocessing
        Distributes rows across processes
        """
        if num_processes is None:
            num_processes = self.num_cores
        
        n, m = A.shape[0], B.shape[1]
        C = np.zeros((n, m), dtype=np.float64)
        
        # Calculate rows per process
        rows_per_process = n // num_processes
        
        # Prepare arguments
        tasks = []
        for i in range(num_processes):
            start_row = i * rows_per_process
            end_row = (i + 1) * rows_per_process if i < num_processes - 1 else n
            tasks.append((A, B, start_row, end_row))
        
        # Execute in parallel using processes
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            results = executor.map(self._multiply_row_range, tasks)
        
        # Combine results
        for start_row, C_partial in results:
            end_row = start_row + C_partial.shape[0]
            C[start_row:end_row, :] = C_partial
        
        return C
    
    @staticmethod
    def _multiply_block(args):
        """Helper for block-based parallel multiplication"""
        A, B, i_start, i_end, j_start, j_end = args
        return np.dot(A[i_start:i_end, :], B[:, j_start:j_end])
    
    def parallel_blocks(self, A, B, num_threads=None, block_size=None):
        """
        Block-based parallel multiplication
        Divides matrix into blocks and computes in parallel
        """
        if num_threads is None:
            num_threads = self.num_cores
        
        n = A.shape[0]
        
        if block_size is None:
            # Auto-calculate block size
            block_size = max(32, n // int(np.sqrt(num_threads)))
        
        C = np.zeros((n, n), dtype=np.float64)
        
        # Generate block tasks
        tasks = []
        for i in range(0, n, block_size):
            for j in range(0, n, block_size):
                i_end = min(i + block_size, n)
                j_end = min(j + block_size, n)
                tasks.append((A, B, i, i_end, j, j_end))
        
        # Execute blocks in parallel
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            results = list(executor.map(self._multiply_block, tasks))
        
        # Combine results
        idx = 0
        for i in range(0, n, block_size):
            for j in range(0, n, block_size):
                i_end = min(i + block_size, n)
                j_end = min(j + block_size, n)
                C[i:i_end, j:j_end] = results[idx]
                idx += 1
        
        return C
    
    def vectorized_numpy(self, A, B):
        """
        Vectorized multiplication using NumPy
        Uses SIMD instructions internally via BLAS
        """
        return np.matmul(A, B)
    
    def numpy_parallel_blas(self, A, B):
        """
        NumPy with parallel BLAS
        Uses multithreaded BLAS underneath
        """
        # NumPy/BLAS will use multiple threads automatically
        # if linked to multithreaded BLAS (MKL, OpenBLAS)
        return A @ B
    
    def benchmark_algorithm(self, algorithm_name, A, B, num_threads=None, 
                          baseline_time=None, num_runs=3, warmup=1):
        """
        Benchmark a specific parallel algorithm
        """
        algorithms = {
            'sequential_naive': lambda: self.naive_sequential(A, B),
            'sequential_numpy': lambda: self.numpy_sequential(A, B),
            'parallel_threads': lambda: self.parallel_rows_threads(A, B, num_threads),
            'parallel_processes': lambda: self.parallel_rows_processes(A, B, num_threads),
            'parallel_blocks': lambda: self.parallel_blocks(A, B, num_threads),
            'vectorized_numpy': lambda: self.vectorized_numpy(A, B),
            'numpy_parallel_blas': lambda: self.numpy_parallel_blas(A, B),
        }
        
        if algorithm_name not in algorithms:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")
        
        algo_func = algorithms[algorithm_name]
        n = A.shape[0]
        
        print(f"      Running warmup ({warmup} runs)...", end=" ")
        # Warmup
        for _ in range(warmup):
            try:
                result = algo_func()
                if result is None:
                    print("FAILED")
                    return None
            except Exception as e:
                print(f"ERROR: {e}")
                return None
        print("OK")
        
        # Benchmark
        print(f"      Running benchmark ({num_runs} runs)...", end=" ")
        times = []
        for i in range(num_runs):
            try:
                start = time.perf_counter()
                C = algo_func()
                end = time.perf_counter()
                
                if C is None:
                    print(f"FAILED at run {i+1}")
                    return None
                    
                times.append(end - start)
                print(f".", end="", flush=True)
            except Exception as e:
                print(f"ERROR: {e}")
                return None
        
        print(" OK")
        
        avg_time = np.mean(times)
        
        # Calculate metrics
        flops = 2 * n ** 3  # n³ multiplications + n³ additions
        gflops = (flops / avg_time) / 1e9
        
        # Calculate speedup and efficiency
        if baseline_time is None:
            baseline_time = avg_time
            speedup = 1.0
            efficiency = 1.0
        else:
            speedup = baseline_time / avg_time
            efficiency = speedup / (num_threads if num_threads else 1)
        
        result = ParallelResult(
            algorithm=algorithm_name,
            matrix_size=n,
            num_threads=num_threads if num_threads else 1,
            execution_time=avg_time,
            gflops=gflops,
            speedup=speedup,
            efficiency=efficiency,
            baseline_time=baseline_time
        )
        
        self.results.append(result)
        return result
    
    def thread_scaling_analysis(self, n=1000, max_threads=None):
        """
        Analyze how performance scales with number of threads
        """
        if max_threads is None:
            max_threads = min(8, self.num_cores)  # Limit to 8 to avoid issues
        
        print("\n" + "="*70)
        print(f"THREAD SCALING ANALYSIS (n={n})")
        print("="*70)
        
        # Generate test matrices
        print(f"\n  Generating {n}×{n} test matrices...", end=" ")
        A = np.random.rand(n, n)
        B = np.random.rand(n, n)
        print("OK")
        
        # Get baseline (sequential)
        print("\n  Baseline: Sequential NumPy...")
        baseline = self.benchmark_algorithm('sequential_numpy', A, B)
        
        if baseline is None:
            print("  ✗ Baseline failed - cannot continue")
            return []
            
        baseline_time = baseline.execution_time
        
        print(f"    ✓ Time: {baseline_time:.6f}s, GFlop/s: {baseline.gflops:.3f}")
        
        # Test with different thread counts
        thread_counts = [1, 2, 4]
        if max_threads >= 8:
            thread_counts.append(8)
        
        # Only test threads that are <= available cores
        thread_counts = [t for t in thread_counts if t <= max_threads]
        
        results = []
        
        for num_threads in thread_counts:
            print(f"\n  Testing with {num_threads} thread(s)...")
            
            try:
                # Test threaded implementation
                result = self.benchmark_algorithm(
                    'parallel_threads', A, B, 
                    num_threads=num_threads,
                    baseline_time=baseline_time,
                    num_runs=2,  # Reduced for speed
                    warmup=1
                )
                
                if result is None:
                    print(f"    ✗ Test failed - skipping")
                    continue
                
                results.append(result)
                
                print(f"    ✓ Time: {result.execution_time:.6f}s")
                print(f"    ✓ GFlop/s: {result.gflops:.3f}")
                print(f"    ✓ Speedup: {result.speedup:.2f}x")
                print(f"    ✓ Efficiency: {result.efficiency*100:.1f}%")
                
            except Exception as e:
                print(f"    ✗ Error: {e}")
                continue
        
        return results
    
    def compare_all_approaches(self, sizes=[100, 500, 1000], num_threads=None):
        """
        Compare all parallel and vectorized approaches
        """
        if num_threads is None:
            num_threads = min(4, self.num_cores)
        
        print("\n" + "="*70)
        print("COMPREHENSIVE PARALLEL COMPARISON")
        print("="*70)
        
        all_results = []
        
        for n in sizes:
            print(f"\n{'='*70}")
            print(f"Matrix size: {n}x{n}")
            print(f"{'='*70}")
            
            A = np.random.rand(n, n)
            B = np.random.rand(n, n)
            
            # Determine which algorithms to test
            if n <= 500:
                algorithms = [
                    'sequential_naive',
                    'sequential_numpy',
                    'parallel_threads',
                    'vectorized_numpy',
                    'numpy_parallel_blas'
                ]
            else:
                # Skip naive for large matrices (too slow)
                algorithms = [
                    'sequential_numpy',
                    'parallel_threads',
                    'parallel_blocks',
                    'vectorized_numpy',
                    'numpy_parallel_blas'
                ]
            
            # Get baseline
            baseline = self.benchmark_algorithm('sequential_numpy', A, B)
            baseline_time = baseline.execution_time
            
            print(f"\nBaseline (Sequential NumPy): {baseline_time:.6f}s, {baseline.gflops:.3f} GFlop/s")
            print(f"\n{'-'*70}")
            
            for algo in algorithms:
                if algo == 'sequential_numpy':
                    continue  # Already computed as baseline
                
                try:
                    print(f"\n{algo}:")
                    
                    if 'parallel' in algo or 'blocks' in algo:
                        result = self.benchmark_algorithm(
                            algo, A, B, 
                            num_threads=num_threads,
                            baseline_time=baseline_time
                        )
                    else:
                        result = self.benchmark_algorithm(
                            algo, A, B,
                            baseline_time=baseline_time
                        )
                    
                    all_results.append(result)
                    
                    print(f"  Time: {result.execution_time:.6f}s")
                    print(f"  GFlop/s: {result.gflops:.3f}")
                    print(f"  Speedup: {result.speedup:.2f}x")
                    if result.num_threads > 1:
                        print(f"  Efficiency: {result.efficiency*100:.1f}%")
                    
                except Exception as e:
                    print(f"  Error: {e}")
        
        return all_results
    
    def save_results(self, filename="parallel_results.json"):
        """Save results to JSON"""
        filepath = os.path.join(self.results_dir, filename)
        results_dict = [r.to_dict() for r in self.results]
        
        with open(filepath, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"\nResults saved to {filepath}")
    
    def print_summary(self):
        """Print summary of benchmark results"""
        print("\n" + "="*70)
        print("BENCHMARK SUMMARY")
        print("="*70)
        
        if not self.results:
            print("No results available")
            return
        
        print(f"\nTotal benchmarks: {len(self.results)}")
        
        # Find best speedup
        parallel_results = [r for r in self.results if r.num_threads > 1]
        if parallel_results:
            best = max(parallel_results, key=lambda x: x.speedup)
            print(f"\nBest speedup:")
            print(f"  Algorithm: {best.algorithm}")
            print(f"  Matrix size: {best.matrix_size}")
            print(f"  Threads: {best.num_threads}")
            print(f"  Speedup: {best.speedup:.2f}x")
            print(f"  Efficiency: {best.efficiency*100:.1f}%")
        
        # Performance statistics
        speedups = [r.speedup for r in parallel_results] if parallel_results else []
        if speedups:
            print(f"\nSpeedup statistics:")
            print(f"  Min: {min(speedups):.2f}x")
            print(f"  Max: {max(speedups):.2f}x")
            print(f"  Average: {np.mean(speedups):.2f}x")


def main():
    """Main execution"""
    print("="*70)
    print("PARALLEL MATRIX MULTIPLICATION BENCHMARK")
    print("Task 3 - Big Data Course")
    print("="*70)
    
    benchmark = ParallelMatrixMultiplication()
    
    # Test 1: Thread scaling
    print("\n### TEST 1: THREAD SCALING ANALYSIS ###")
    scaling_results = benchmark.thread_scaling_analysis(n=1000)
    
    # Test 2: Compare all approaches
    print("\n\n### TEST 2: COMPREHENSIVE COMPARISON ###")
    comparison_results = benchmark.compare_all_approaches(
        sizes=[100, 500, 1000, 2000],
        num_threads=4
    )
    
    # Save and summarize
    benchmark.save_results()
    benchmark.print_summary()
    
    print("\n" + "="*70)
    print("BENCHMARK COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
