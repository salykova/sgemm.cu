# High-Performance SGEMM on NVIDIA GPUs

> **Important note:** while the implementation is expected to deliver high performance on all Ada/Ampere/Volta/Turing devices, it was specifically fine-tuned for and tested on NVIDIA RTX 3090 (GA102 chip: RTX 3080, A10, A40, A6000).

## Performance

<p align="center">
  <img src="assets/perf.png" alt="perf" width="85%">
</p>

<p align="center">
  <img src="assets/perf_locked.png" alt="perf" width="85%">
</p>

## Benchmark

>Avoid using WSL for performance measurements. To ensure accurate and reliable results, please use a native Linux environment.

To benchmark the code, run `benchmark.sh` and specify compute capability of your CUDA device. For example, on RTX 3090:

```bash
bash benchmark.sh 86
```

The benchmark settings such as minimum/maximum matrix sizes, step size, number of warm-up iterations etc. can be adjusted in the `benchmark.sh` file.

To visualize benchmark results, please install `matplotlib` and run

```python
python plot_benchmark_data.py benchmark_results
```

## Tests

To test the implementation for correctness, run `test.sh` and specify compute capability of your CUDA device. For example, on RTX 3090:

```bash
bash test.sh 86
```