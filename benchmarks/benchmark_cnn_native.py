"""
CNN Training Benchmark: Native MLX
Full training loop with mx.compile, forward + backward + SGD.

Usage:
  python benchmarks/benchmark_cnn_native.py
"""

import mlx.core as mx
import time
import numpy as np


def conv2d(x, w, b, stride=1, padding=1):
    """Conv2d with bias."""
    out = mx.conv2d(x, w, stride=stride, padding=padding)
    return out + b


def maxpool2d(x):
    """2x2 max pooling."""
    N, H, W, C = x.shape
    x = mx.reshape(x, (N, H // 2, 2, W // 2, 2, C))
    return mx.max(x, axis=(2, 4))


def forward(x, params):
    """Forward pass through CNN."""
    c1w, c1b, c2w, c2b, c3w, c3b, c4w, c4b, c5w, c5b, d1w, d1b, d2w, d2b = params

    # Block 1
    h = mx.maximum(conv2d(x, c1w, c1b), 0)
    h = mx.maximum(conv2d(h, c2w, c2b), 0)
    h = maxpool2d(h)

    # Block 2
    h = mx.maximum(conv2d(h, c3w, c3b), 0)
    h = mx.maximum(conv2d(h, c4w, c4b), 0)
    h = maxpool2d(h)

    # Block 3
    h = mx.maximum(conv2d(h, c5w, c5b), 0)

    # Global average pool + Dense
    h = mx.mean(h, axis=(1, 2))
    h = mx.maximum(mx.matmul(h, d1w) + d1b, 0)
    h = mx.matmul(h, d2w) + d2b
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


def benchmark_native(batch_size=128, image_size=64, num_warmup=30, num_runs=20):
    print(f"\n{'='*60}")
    print(f"CNN Training Benchmark: Native MLX (functional compiled)")
    print(f"{'='*60}")
    print(f"Config: batch_size={batch_size}, image_size={image_size}x{image_size}x3")
    print(f"Model: Conv(64)×2 → Pool → Conv(128)×2 → Pool → Conv(256) → GAP → Dense(128) → Dense(10)")

    lr = 0.01

    # Initialize weights (MLX conv2d format: [out_channels, kH, kW, in_channels])
    scale = (2.0 / (3 * 3 * 3)) ** 0.5
    c1w = mx.random.normal((64, 3, 3, 3)) * scale
    c1b = mx.zeros((64,))

    scale = (2.0 / (3 * 3 * 64)) ** 0.5
    c2w = mx.random.normal((64, 3, 3, 64)) * scale
    c2b = mx.zeros((64,))

    c3w = mx.random.normal((128, 3, 3, 64)) * scale
    c3b = mx.zeros((128,))

    scale = (2.0 / (3 * 3 * 128)) ** 0.5
    c4w = mx.random.normal((128, 3, 3, 128)) * scale
    c4b = mx.zeros((128,))

    c5w = mx.random.normal((256, 3, 3, 128)) * scale
    c5b = mx.zeros((256,))

    scale = (2.0 / 256) ** 0.5
    d1w = mx.random.normal((256, 128)) * scale
    d1b = mx.zeros((128,))

    scale = (2.0 / 128) ** 0.5
    d2w = mx.random.normal((128, 10)) * scale
    d2b = mx.zeros((10,))

    params = [c1w, c1b, c2w, c2b, c3w, c3b, c4w, c4b, c5w, c5b, d1w, d1b, d2w, d2b]
    mx.eval(params)

    # Dummy data
    x = mx.random.normal((batch_size, image_size, image_size, 3))
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
