"""
MLP Training Benchmark: JAX-MLX
Full training loop with forward + backward + SGD.

Usage:
  python benchmarks/benchmark_mlp.py
"""

import os
if "MLX_PJRT_DEBUG" in os.environ:
    del os.environ["MLX_PJRT_DEBUG"]
if "JAX_PLATFORMS" not in os.environ:
    os.environ["JAX_PLATFORMS"] = "cpu,mlx"
elif "cpu" not in os.environ["JAX_PLATFORMS"]:
    os.environ["JAX_PLATFORMS"] = "cpu," + os.environ["JAX_PLATFORMS"]

import jax
import jax.numpy as jnp
from flax import linen as nn
import time
import numpy as np


class MLP(nn.Module):
    """Deep wide MLP for benchmarking."""

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=8192)(x)
        x = nn.relu(x)
        x = nn.Dense(features=4096)(x)
        x = nn.relu(x)
        x = nn.Dense(features=4096)(x)
        x = nn.relu(x)
        x = nn.Dense(features=2048)(x)
        x = nn.relu(x)
        x = nn.Dense(features=1024)(x)
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


def benchmark_mlx(batch_size=2048, input_dim=2048, num_warmup=30, num_runs=20):
    """Benchmark MLP training on MLX."""
    try:
        mlx = jax.devices("mlx")[0]
    except Exception as e:
        print(f"MLX device not available: {e}")
        return None, None

    model = MLP()

    try:
        cpu = jax.devices("cpu")[0]
    except Exception:
        cpu = mlx

    with jax.default_device(cpu):
        rng = jax.random.PRNGKey(0)
        params = model.init(rng, jnp.ones((batch_size, input_dim)))
        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, (batch_size, input_dim))
        y = jax.random.randint(key, (batch_size,), 0, 10)

    with jax.default_device(mlx):
        params = jax.device_put(params, mlx)
        x = jax.device_put(x, mlx)
        y = jax.device_put(y, mlx)
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
    batch_size = 2048
    input_dim = 2048

    print("MLP Training Benchmark (JAX-MLX)")
    print(f"Config: batch_size={batch_size}, input_dim={input_dim}")
    print(f"Model: Dense(1024→8192→4096→4096→2048→1024→10) + ReLU")
    print(f"Task: Forward + Backward pass with SGD\n")

    mean, std = benchmark_mlx(batch_size, input_dim)

    if mean is not None:
        print(f"\n{'='*50}")
        print("SUMMARY")
        print(f"{'='*50}")
        print(f"JAX-MLX: {mean:.2f}ms ± {std:.2f}ms")


if __name__ == "__main__":
    main()
