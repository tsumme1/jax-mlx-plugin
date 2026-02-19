# JAX MLX Plugin

[![PyPI](https://img.shields.io/pypi/v/jax-mlx-plugin)](https://pypi.org/project/jax-mlx-plugin/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A [PJRT](https://openxla.org/xla/pjrt_integration) plugin that lets **JAX run on Apple Silicon GPUs** via [MLX](https://github.com/ml-explore/mlx). Write standard JAX code — the plugin handles compilation to Metal compute kernels automatically.

## Requirements

- Apple Silicon Mac (M1/M2/M3/M4)
- macOS 14.0+ (Sonoma)
- Python 3.11+

## Installation

```bash
pip install jax-mlx-plugin
```

Or from source:

```bash
git clone https://github.com/tsumme1/jax-mlx.git
cd jax-mlx
pip install .
```

## Quick Start

```python
import jax
import jax.numpy as jnp

print(jax.devices())  # [MlxDevice(id=0)]

mlx = jax.devices('mlx')[0]
with jax.default_device(mlx):
    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.sin(x) + jnp.cos(x)
    print(y)  # runs on Metal GPU
```

## What Works

| Category | Details |
|----------|---------|
| **Core ops** | Arithmetic, math, reductions, comparisons, bitwise, type conversion |
| **Autodiff** | `jax.grad`, `value_and_grad`, `jacfwd`, `jacrev`, `hessian` |
| **Transforms** | `jax.jit`, `jax.vmap` |
| **Control flow** | `lax.cond`, `lax.switch`, `lax.while_loop`, `lax.scan`, `lax.fori_loop`, `lax.map`, `lax.associative_scan` |
| **Linear algebra** | `matmul`, `solve`, `inv`, `cholesky`, `qr`, `svd`, `eig`, `eigh`, `triangular_solve`, `slogdet` |
| **Neural networks** | Flax + Optax (CNNs, MLPs, RNNs, Transformers verified) |
| **Convolutions** | `conv_general_dilated` (NHWC/NCHW), pooling (max/min/avg + gradients) |
| **FFT** | `fft`, `ifft`, `rfft`, `irfft`, 2D variants |
| **Distributions** | `jax.random.*` (Threefry PRNG with 64-bit emulation on Metal) |
| **SciPy** | `scipy.special`, `scipy.linalg`, `scipy.stats`, `scipy.signal`, `scipy.ndimage` |

See [ARCHITECTURE.md](docs/ARCHITECTURE.md) for technical details.

## Benchmarks

Four benchmark suites compare JAX-MLX against JAX CPU and native MLX:

| Benchmark | Command |
|-----------|---------|
| CNN (Conv + Pool + Dense) | `python benchmarks/benchmark_cnn.py` |
| MLP (Dense layers) | `python benchmarks/benchmark_mlp.py` |
| RNN (Recurrent) | `python benchmarks/benchmark_rnn.py` |
| Transformer (Attention) | `python benchmarks/benchmark_transformer.py` |

Each also has a `_native.py` variant for direct MLX comparison.

## Testing

```bash
# Exhaustive op coverage (387 ops)
python tests/test_exhaustive.py

# Every op wrapped in lax.while_loop (362 ops)
python tests/test_exhaustive_while.py

# Compilation tier coverage
python tests/test_compilation_coverage.py
```

## Known Limitations

- **Float64** — Not natively supported on Metal; use Float32
- **While loops** — Block kernel fusion for the enclosing graph (segments within are still compiled)
- **LAPACK ops** — LU factorization, slogdet use CPU interpreter fallback

## License

MIT — see [LICENSE](LICENSE).
