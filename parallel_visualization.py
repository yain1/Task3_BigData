"""
Visualization for Parallel Matrix Multiplication Results
Task 3 - Big Data Course
"""

import matplotlib.pyplot as plt
import numpy as np
import json
import os


class ParallelVisualizer:
    """Visualization for parallel benchmark results"""
    
    def __init__(self, results_dir="results", output_dir="figures"):
        self.results_dir = results_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        plt.style.use('seaborn-v0_8-darkgrid')
    
    def load_results(self, filename="parallel_results.json"):
        """Load results from JSON"""
        filepath = os.path.join(self.results_dir, filename)
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def plot_thread_scaling(self, results, save=True):
        """
        Plot: Speedup vs Number of Threads
        Shows strong scaling behavior
        """
        # Filter results for thread scaling
        scaling_results = [r for r in results if 'parallel' in r['algorithm']]
        
        if not scaling_results:
            print("No thread scaling data available")
            return
        
        # Group by matrix size
        sizes = sorted(set(r['matrix_size'] for r in scaling_results))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        for size in sizes:
            size_data = [r for r in scaling_results if r['matrix_size'] == size]
            size_data.sort(key=lambda x: x['num_threads'])
            
            threads = [r['num_threads'] for r in size_data]
            speedups = [r['speedup'] for r in size_data]
            efficiencies = [r['efficiency'] * 100 for r in size_data]
            
            ax1.plot(threads, speedups, marker='o', label=f'n={size}', linewidth=2)
            ax2.plot(threads, efficiencies, marker='s', label=f'n={size}', linewidth=2)
        
        # Ideal speedup line
        max_threads = max([r['num_threads'] for r in scaling_results])
        ideal_threads = list(range(1, max_threads + 1))
        ax1.plot(ideal_threads, ideal_threads, 'k--', alpha=0.5, label='Ideal')
        
        ax1.set_xlabel('Number of Threads', fontsize=12)
        ax1.set_ylabel('Speedup', fontsize=12)
        ax1.set_title('Strong Scaling: Speedup vs Threads', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.set_xlabel('Number of Threads', fontsize=12)
        ax2.set_ylabel('Efficiency (%)', fontsize=12)
        ax2.set_title('Parallel Efficiency', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=100, color='k', linestyle='--', alpha=0.5, label='Ideal')
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, 'thread_scaling.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved: {filepath}")
        
        plt.show()
    
    def plot_algorithm_comparison(self, results, save=True):
        """
        Plot: Algorithm Performance Comparison
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Group by matrix size
        sizes = sorted(set(r['matrix_size'] for r in results))
        algorithms = sorted(set(r['algorithm'] for r in results))
        
        # Performance comparison
        x = np.arange(len(algorithms))
        width = 0.15
        
        for i, size in enumerate(sizes):
            gflops = []
            for algo in algorithms:
                matching = [r for r in results 
                           if r['algorithm'] == algo and r['matrix_size'] == size]
                if matching:
                    gflops.append(matching[0]['gflops'])
                else:
                    gflops.append(0)
            
            ax1.bar(x + i*width, gflops, width, label=f'n={size}', alpha=0.8)
        
        ax1.set_xlabel('Algorithm', fontsize=12)
        ax1.set_ylabel('Performance (GFlop/s)', fontsize=12)
        ax1.set_title('Algorithm Performance Comparison', fontsize=14, fontweight='bold')
        ax1.set_xticks(x + width * (len(sizes)-1) / 2)
        ax1.set_xticklabels(algorithms, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Speedup comparison
        for i, size in enumerate(sizes):
            speedups = []
            for algo in algorithms:
                matching = [r for r in results 
                           if r['algorithm'] == algo and r['matrix_size'] == size]
                if matching:
                    speedups.append(matching[0]['speedup'])
                else:
                    speedups.append(0)
            
            ax2.bar(x + i*width, speedups, width, label=f'n={size}', alpha=0.8)
        
        ax2.set_xlabel('Algorithm', fontsize=12)
        ax2.set_ylabel('Speedup', fontsize=12)
        ax2.set_title('Speedup vs Sequential Baseline', fontsize=14, fontweight='bold')
        ax2.set_xticks(x + width * (len(sizes)-1) / 2)
        ax2.set_xticklabels(algorithms, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.axhline(y=1, color='r', linestyle='--', alpha=0.5, label='Baseline')
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, 'algorithm_comparison_parallel.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved: {filepath}")
        
        plt.show()
    
    def plot_amdahl_law(self, results, save=True):
        """
        Plot: Amdahl's Law Analysis
        Shows theoretical vs actual speedup
        """
        parallel_results = [r for r in results if r['num_threads'] > 1]
        
        if not parallel_results:
            print("No parallel results for Amdahl's law")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Get unique thread counts
        thread_counts = sorted(set(r['num_threads'] for r in parallel_results))
        
        # Plot actual speedups
        sizes = set(r['matrix_size'] for r in parallel_results)
        for size in sizes:
            size_data = [r for r in parallel_results if r['matrix_size'] == size]
            size_data.sort(key=lambda x: x['num_threads'])
            
            threads = [r['num_threads'] for r in size_data]
            speedups = [r['speedup'] for r in size_data]
            
            ax.plot(threads, speedups, marker='o', label=f'Actual (n={size})', linewidth=2)
        
        # Plot Amdahl's law for different parallel fractions
        max_threads = max(thread_counts)
        p_values = [0.75, 0.90, 0.95, 0.99]  # Parallel fractions
        threads_range = np.linspace(1, max_threads, 100)
        
        for p in p_values:
            amdahl_speedup = 1 / ((1 - p) + p / threads_range)
            ax.plot(threads_range, amdahl_speedup, '--', alpha=0.6, 
                   label=f"Amdahl's Law (p={p})")
        
        ax.set_xlabel('Number of Threads', fontsize=12)
        ax.set_ylabel('Speedup', fontsize=12)
        ax.set_title("Amdahl's Law: Theoretical vs Actual Speedup", 
                    fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, 'amdahl_law.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved: {filepath}")
        
        plt.show()
    
    def create_summary_table(self, results, save=True):
        """Create summary table of results"""
        print("\n" + "="*100)
        print("PARALLEL BENCHMARK RESULTS SUMMARY")
        print("="*100)
        print(f"{'Algorithm':<25} {'Size':<8} {'Threads':<8} {'Time (s)':<12} "
              f"{'GFlop/s':<12} {'Speedup':<10} {'Efficiency':<12}")
        print("-"*100)
        
        for r in sorted(results, key=lambda x: (x['matrix_size'], x['algorithm'])):
            print(f"{r['algorithm']:<25} {r['matrix_size']:<8} {r['num_threads']:<8} "
                  f"{r['execution_time']:>10.6f}  {r['gflops']:>10.3f}  "
                  f"{r['speedup']:>8.2f}x  {r['efficiency']*100:>10.1f}%")
        
        if save:
            filepath = os.path.join(self.output_dir, 'parallel_summary.txt')
            with open(filepath, 'w') as f:
                f.write("PARALLEL BENCHMARK RESULTS SUMMARY\n")
                f.write("="*100 + "\n")
                f.write(f"{'Algorithm':<25} {'Size':<8} {'Threads':<8} {'Time (s)':<12} "
                       f"{'GFlop/s':<12} {'Speedup':<10} {'Efficiency':<12}\n")
                f.write("-"*100 + "\n")
                
                for r in sorted(results, key=lambda x: (x['matrix_size'], x['algorithm'])):
                    f.write(f"{r['algorithm']:<25} {r['matrix_size']:<8} {r['num_threads']:<8} "
                           f"{r['execution_time']:>10.6f}  {r['gflops']:>10.3f}  "
                           f"{r['speedup']:>8.2f}x  {r['efficiency']*100:>10.1f}%\n")
            
            print(f"\nSaved: {filepath}")
    
    def generate_all_plots(self):
        """Generate all visualization plots"""
        print("\n" + "="*70)
        print("GENERATING VISUALIZATIONS")
        print("="*70)
        
        try:
            results = self.load_results()
            
            print("\n1. Thread scaling plots...")
            self.plot_thread_scaling(results)
            
            print("\n2. Algorithm comparison...")
            self.plot_algorithm_comparison(results)
            
            print("\n3. Amdahl's law analysis...")
            self.plot_amdahl_law(results)
            
            print("\n4. Summary table...")
            self.create_summary_table(results)
            
            print("\n" + "="*70)
            print("ALL VISUALIZATIONS GENERATED")
            print("="*70)
            
        except FileNotFoundError:
            print("No results file found. Run benchmarks first.")


if __name__ == "__main__":
    visualizer = ParallelVisualizer()
    visualizer.generate_all_plots()
