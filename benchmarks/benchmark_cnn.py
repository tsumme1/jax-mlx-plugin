"""
CNN Training Benchmark: JAX-MLX
Full training loop with forward + backward + SGD.

Usage:
  python benchmarks/benchmark_cnn.py
"""

import os
if "MLX_PJRT_DEBUG" in os.environ:
    del os.environ["MLX_PJRT_DEBUG"]
if "JAX_PLATFORMS" not in os.environ:
    os.environ["JAX_PLATFORMS"] = "mlx"

import jax
import jax.numpy as jnp
from flax import linen as nn
import time
import numpy as np


class SimpleCNN(nn.Module):
    """Deep CNN for benchmarking."""

    @nn.compact
    def __call__(self, x):
        # Block 1
        x = nn.Conv(features=64, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        # Block 2
        x = nn.Conv(features=128, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(features=128, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        # Block 3
        x = nn.Conv(features=256, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.relu(x)

        x = jnp.mean(x, axis=(1, 2))  # Global average pooling
        x = nn.Dense(features=128)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        return x


def loss_fn(params, model, x, y):
    """Cross-entropy loss."""
    logits = model.apply(params, x)
    one_hot = jax.nn.one_hot(y, 10)
    return -jnp.mean(jnp.sum(jax.nn.log_softmax(logits) * one_hot, axis=-1))


def create_train_step(model, lr=0.01):
    @jax.jit
    def train_step(params, x, y):
        loss, grads = jax.value_and_grad(lambda p: loss_fn(p, model, x, y))(params)
        params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
        return params, loss
    return train_step


def benchmark_mlx(batch_size=128, image_size=64, num_warmup=30, num_runs=20):
    """Benchmark CNN training on MLX."""
    try:
        device = jax.devices("mlx")[0]
    except Exception as e:
        print(f"MLX device not available: {e}")
        return None, None

    model = SimpleCNN()
    input_shape = (batch_size, image_size, image_size, 3)

    with jax.default_device(device):
        rng = jax.random.PRNGKey(0)
        params = model.init(rng, jnp.ones(input_shape))
        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, input_shape)
        y = jax.random.randint(key, (batch_size,), 0, 10)
        train_step = create_train_step(model)

        # Warmup
        print(f"Warming up ({num_warmup} steps)...")
        for _ in range(num_warmup):
            params, loss = train_step(params, x, y)
            loss.block_until_ready()

        # Timed runs
        print(f"\n{'='*50}")
        print("Timed runs")
        print(f"{'='*50}")
        times = []
        for i in range(num_runs):
            start = time.perf_counter()
            params, loss = train_step(params, x, y)
            loss.block_until_ready()
            end = time.perf_counter()
            times.append(end - start)
            print(f"  Run {i+1}: {times[-1]*1000:.2f}ms (loss: {float(loss):.4f})")

    mean_time = np.mean(times) * 1000
    std_time = np.std(times) * 1000
    min_time = np.min(times) * 1000
    print(f"\n  Mean: {mean_time:.2f}ms ± {std_time:.2f}ms  Min: {min_time:.2f}ms")
    return mean_time, std_time


def main():
    batch_size = 128
    image_size = 64

    print("CNN Training Benchmark (JAX-MLX)")
    print(f"Config: batch_size={batch_size}, image_size={image_size}x{image_size}x3")
    print(f"Model: Conv(64)×2 → Pool → Conv(128)×2 → Pool → Conv(256) → GAP → Dense(128) → Dense(10)")
    print(f"Task: Forward + Backward pass with SGD\n")

    mean, std = benchmark_mlx(batch_size, image_size)

    if mean is not None:
        print(f"\n{'='*50}")
        print("SUMMARY")
        print(f"{'='*50}")
        print(f"JAX-MLX: {mean:.2f}ms ± {std:.2f}ms")


if __name__ == "__main__":
    main()
