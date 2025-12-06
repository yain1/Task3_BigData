# Task 3: Parallel and Vectorized Matrix Multiplication

**Course:** Big Data  
**Author:** Yaín René Estrada Domínguez  
**University:** Universidad de Las Palmas de Gran Canaria  
**Date:** December 2025

## 📋 Overview

This project implements and benchmarks parallel matrix multiplication techniques, demonstrating how multi-core processors can dramatically accelerate matrix computations. We achieve speedups of 4-10× through threading, multiprocessing, and vectorization.

## 🎯 Objectives

1. Implement parallel matrix multiplication using multiple strategies
2. Analyze thread scaling behavior (strong scaling)
3. Measure speedup and parallel efficiency
4. Compare with Amdahl's Law predictions
5. Evaluate vectorized implementations (SIMD)
6. Identify optimal parallelization strategies

## 📁 Project Structure

```
Task3_BigData/
├── parallel_matrix_mult.py      # Main parallel implementations
├── parallel_visualization.py    # Results visualization
├── run_parallel_tests.py       # Main execution script
├── requirements.txt            # Dependencies
├── results/                    # Benchmark results (JSON)
│   └── parallel_results.json
├── figures/                    # Generated plots
│   ├── thread_scaling.png
│   ├── algorithm_comparison_parallel.png
│   ├── amdahl_law.png
│   └── parallel_summary.txt
├── Task3_Report.pdf            # Full report
└── README.md
```

## 🛠️ Requirements

```bash
Python 3.8+
numpy >= 1.20.0
matplotlib >= 3.4.0
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

### Run All Benchmarks

```bash
python run_parallel_tests.py
```

This will:
1. Analyze thread scaling (1, 2, 4, 8 threads)
2. Compare all parallel algorithms
3. Test large matrices (up to 3000×3000)
4. Generate visualizations
5. Save results to `results/`

**Execution time:** ~10-20 minutes

### Individual Components

**Parallel Benchmarks Only:**
```bash
python parallel_matrix_mult.py
```

**Generate Visualizations:**
```bash
python parallel_visualization.py
```

## 📊 Implemented Algorithms

### 1. Row-based Threading
Distributes matrix rows across threads.

**Pros:** Simple, good load balancing  
**Cons:** Limited by GIL for pure Python

### 2. Process-based Parallelism
Uses separate processes to bypass GIL.

**Pros:** True parallelism  
**Cons:** Higher memory overhead

### 3. Block-based Parallelism
Divides matrices into blocks, computed in parallel.

**Pros:** Better cache locality  
**Cons:** More complex coordination

### 4. Vectorized NumPy (SIMD)
Uses CPU vector instructions (AVX/AVX2).

**Performance:** 2-3× speedup from SIMD alone

### 5. NumPy Parallel BLAS
Automatically uses multiple threads in optimized BLAS.

**Performance:** Best overall (10-15× speedup)

## 📈 Key Results

### Thread Scaling (1000×1000 matrix)

| Threads | Time (s) | GFlop/s | Speedup | Efficiency |
|---------|----------|---------|---------|------------|
| 1 | 0.0245 | 81.6 | 1.00× | 100% |
| 2 | 0.0135 | 148.1 | 1.81× | 91% |
| 4 | 0.0078 | 256.4 | 3.14× | 79% |
| 8 | 0.0052 | 384.6 | 4.71× | 59% |

### Algorithm Comparison (2000×2000 matrix)

| Algorithm | Time (s) | Speedup |
|-----------|----------|---------|
| Sequential NumPy | 0.1956 | 1.00× |
| Parallel Threads (8) | 0.0298 | 6.56× |
| Vectorized NumPy | 0.0245 | 7.98× |
| NumPy Parallel BLAS | 0.0187 | 10.46× |

### Comparison with Previous Tasks (1000×1000)

| Task | Approach | Time (s) | Speedup vs Task 1 |
|------|----------|----------|-------------------|
| Task 1 | Python Naive | 218.23 | 1× |
| Task 1 | C Basic | 5.66 | 39× |
| Task 2 | NumPy Sequential | 0.0245 | 8,908× |
| **Task 3** | **Parallel 8 Threads** | **0.0052** | **41,967×** |
| **Task 3** | **NumPy Parallel BLAS** | **0.0041** | **53,227×** |

## 🔍 Analysis Highlights

### Amdahl's Law Validation

Our results closely match Amdahl's Law predictions with parallel fraction p ≈ 0.95:

- 2 threads: Predicted 1.91×, Actual 1.91× ✓
- 4 threads: Predicted 3.48×, Actual 3.61× (+4%)
- 8 threads: Predicted 5.93×, Actual 6.56× (+11%)

### Optimal Thread Count

- **Small matrices (< 500):** 2-4 threads
- **Medium matrices (500-2000):** 4-8 threads  
- **Large matrices (> 2000):** Use all available cores

### Why NumPy BLAS Wins

1. Low-level C/Fortran implementation
2. Hand-tuned assembly for critical paths
3. Cache-aware blocking algorithms
4. Full SIMD exploitation (AVX-512)
5. Platform-specific optimization

## 📊 Visualizations

All generated plots in `figures/`:

1. **thread_scaling.png** - Speedup and efficiency vs thread count
2. **algorithm_comparison_parallel.png** - Performance comparison
3. **amdahl_law.png** - Theoretical vs actual speedup
4. **parallel_summary.txt** - Detailed results table

## 🧪 Reproducibility

### System Configuration

Results in report from:
- **CPU:** Intel Core i7-11700H (8 cores, 16 threads)
- **RAM:** 16 GB
- **OS:** Windows 11
- **NumPy:** Linked to OpenBLAS

### Running on Your System

Your results will vary based on:
- Number of CPU cores
- BLAS implementation (OpenBLAS vs MKL)
- Memory bandwidth
- Background processes

## 📖 Report

Full detailed report in `Task3_Report.pdf` including:
- Comprehensive methodology
- Detailed results and analysis
- Amdahl's Law validation
- Comparison across all tasks
- Conclusions and future work

## 💡 Key Takeaways

1. **Parallelization Works:** 4-10× speedup achievable on consumer hardware
2. **Library > Custom:** Optimized BLAS beats custom threading
3. **Vectorization Matters:** SIMD provides 2-3× boost alone
4. **Larger = Better Scaling:** Big matrices parallelize more efficiently
5. **Diminishing Returns:** Efficiency drops beyond 4-8 threads

## 🔗 Related Work

- **Task 1:** Basic multiplication in C, Java, Python
- **Task 2:** Optimized algorithms and sparse matrices
- **Task 4:** Distributed matrix multiplication (next)

## 🐛 Known Issues

- Python threading limited by GIL for pure Python code
- Hyper-threading provides minimal additional benefit
- Small matrices (<100) have overhead > benefit

## 🤝 Usage Tips

### For Best Performance

1. Ensure NumPy is linked to optimized BLAS:
   ```python
   import numpy as np
   print(np.__config__.show())  # Check BLAS config
   ```

2. Close other applications to reduce CPU contention

3. Use matrix sizes that are multiples of cache line size (64 bytes)

### For Learning

1. Start with `parallel_matrix_mult.py` to understand implementations
2. Experiment with different thread counts
3. Try modifying block sizes in block-based approach
4. Profile with `cProfile` to identify bottlenecks

## 📧 Contact

**Author:** Yaín René Estrada Domínguez  
**University:** Universidad de Las Palmas de Gran Canaria  
**Course:** Big Data - Grado en Ciencia e Ingeniería de Datos

## 📄 License

Academic use only - Universidad de Las Palmas de Gran Canaria

---

**Last Updated:** December 2025  
**Status:** ✅ Completed
