"""
MLP Training Benchmark: Native MLX
Full training loop with mx.compile, forward + backward + SGD.

Usage:
  python benchmarks/benchmark_mlp_native.py
"""

import mlx.core as mx
import time
import numpy as np


def forward(x, params):
    """Forward pass through deep wide MLP."""
    w1, b1, w2, b2, w3, b3, w4, b4, w5, b5, w6, b6 = params

    h = mx.matmul(x, w1) + b1
    h = mx.maximum(h, 0)
    h = mx.matmul(h, w2) + b2
    h = mx.maximum(h, 0)
    h = mx.matmul(h, w3) + b3
    h = mx.maximum(h, 0)
    h = mx.matmul(h, w4) + b4
    h = mx.maximum(h, 0)
    h = mx.matmul(h, w5) + b5
    h = mx.maximum(h, 0)
    h = mx.matmul(h, w6) + b6
    return h


def cross_entropy(logits, labels):
    """Numerically stable cross-entropy."""
    max_logits = mx.max(logits, axis=-1, keepdims=True)
    shifted = logits - max_logits
    log_softmax = shifted - mx.log(mx.sum(mx.exp(shifted), axis=-1, keepdims=True))
    indices = mx.expand_dims(labels, -1)
    log_probs = mx.take_along_axis(log_softmax, indices.astype(mx.int32), axis=1)
    return -mx.mean(log_probs)


def loss_fn(params, x, y):
    logits = forward(x, params)
    return cross_entropy(logits, y)


def benchmark_native(batch_size=2048, input_dim=2048, num_warmup=30, num_runs=20):
    print(f"\n{'='*60}")
    print(f"MLP Training Benchmark: Native MLX (functional compiled)")
    print(f"{'='*60}")
    print(f"Config: batch_size={batch_size}, input_dim={input_dim}")
    print(f"Model: Dense(1024→8192→4096→4096→2048→1024→10) + ReLU")

    lr = 0.01

    # Initialize weights (Xavier-like)
    dims = [(input_dim, 8192), (8192, 4096), (4096, 4096), (4096, 2048), (2048, 1024), (1024, 10)]
    params = []
    for d_in, d_out in dims:
        scale = (2.0 / d_in) ** 0.5
        params.append(mx.random.normal((d_in, d_out)) * scale)
        params.append(mx.zeros((d_out,)))
    mx.eval(params)

    # Dummy data
    x = mx.random.normal((batch_size, input_dim))
    y = mx.random.randint(0, 10, (batch_size,))
    mx.eval(x, y)

    # Create compiled train step
    grad_fn = mx.value_and_grad(loss_fn)

    def train_step(params, x, y):
        loss, grads = grad_fn(params, x, y)
        new_params = [p - lr * g for p, g in zip(params, grads)]
        return [loss] + new_params

    compiled_train_step = mx.compile(train_step)

    # Warmup
    print(f"\nWarmup ({num_warmup} runs)...")
    for _ in range(num_warmup):
        result = compiled_train_step(params, x, y)
        mx.eval(result)
        params = result[1:]

    # Timed runs
    print(f"\n{'='*50}")
    print("Timed runs")
    print(f"{'='*50}")
    times = []
    for i in range(num_runs):
        start = time.perf_counter()
        result = compiled_train_step(params, x, y)
        mx.eval(result)
        params = result[1:]
        end = time.perf_counter()
        times.append((end - start) * 1000)
        print(f"  Run {i+1}: {times[-1]:.2f}ms (loss: {float(result[0]):.4f})")

    mean_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)

    print(f"\n  Mean: {mean_time:.2f}ms ± {std_time:.2f}ms  Min: {min_time:.2f}ms")

    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    print(f"Native MLX: {mean_time:.2f}ms ± {std_time:.2f}ms")
    return mean_time


if __name__ == "__main__":
    benchmark_native()
