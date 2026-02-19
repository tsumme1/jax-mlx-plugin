"""
RNN (GRU) Training Benchmark: JAX-MLX
2-layer GRU → Dense(relu) → Dense(10) with SGD.
Matches the architecture of native MLX and Julia benchmarks.

Usage:
  python benchmarks/benchmark_rnn.py
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
from flax import nnx
import time
import numpy as np


class GRULayer(nnx.Module):
    """Single GRU layer using lax.scan with full unroll."""

    def __init__(self, input_dim: int, hidden_dim: int, *, rngs: nnx.Rngs):
        self.hidden_dim = hidden_dim
        self.input_proj = nnx.Linear(input_dim, 3 * hidden_dim, rngs=rngs)
        self.hidden_proj = nnx.Linear(hidden_dim, 3 * hidden_dim, use_bias=False, rngs=rngs)

    def __call__(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            h_final: (batch, hidden_dim) — final hidden state
            h_all:   (batch, seq_len, hidden_dim) — all hidden states
        """
        batch, seq_len, _ = x.shape
        H = self.hidden_dim
        h0 = jnp.zeros((batch, H))

        # Pre-compute input projections for all timesteps
        x_proj = jax.vmap(self.input_proj)(x.reshape(-1, x.shape[-1])).reshape(batch, seq_len, 3 * H)
        x_proj = jnp.transpose(x_proj, (1, 0, 2))  # (seq_len, batch, 3*H)

        hidden_proj = self.hidden_proj

        def gru_step(h, x_t):
            hx = hidden_proj(h)
            z = jax.nn.sigmoid(x_t[:, :H] + hx[:, :H])
            r = jax.nn.sigmoid(x_t[:, H:2*H] + hx[:, H:2*H])
            n = jnp.tanh(x_t[:, 2*H:] + r * hx[:, 2*H:])
            h_new = (1 - z) * n + z * h
            return h_new, h_new

        h_final, h_all = jax.lax.scan(gru_step, h0, x_proj, unroll=seq_len)
        # h_all: (seq_len, batch, H) → (batch, seq_len, H)
        h_all = jnp.transpose(h_all, (1, 0, 2))
        return h_final, h_all


class GRUClassifier(nnx.Module):
    """2-layer GRU → Dense(relu) → Dense(10). Matches Julia/native."""

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, *, rngs: nnx.Rngs):
        self.gru1 = GRULayer(input_dim, hidden_dim, rngs=rngs)
        self.gru2 = GRULayer(hidden_dim, hidden_dim, rngs=rngs)
        self.dense1 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.dense2 = nnx.Linear(hidden_dim, num_classes, rngs=rngs)

    def __call__(self, x):
        _, h1_all = self.gru1(x)         # h1_all: (batch, seq_len, hidden)
        h2_final, _ = self.gru2(h1_all)  # h2_final: (batch, hidden)
        h = jax.nn.relu(self.dense1(h2_final))
        return self.dense2(h)


def cross_entropy_loss(model, x, y):
    logits = model(x)
    one_hot = jax.nn.one_hot(y, logits.shape[-1])
    return -jnp.mean(jnp.sum(jax.nn.log_softmax(logits) * one_hot, axis=-1))


import optax


@nnx.jit
def train_step(model, optimizer, x, y):
    loss, grads = nnx.value_and_grad(cross_entropy_loss)(model, x, y)
    optimizer.update(model, grads)
    return loss


def benchmark_rnn(batch_size=128, seq_len=64, input_dim=64, hidden_dim=512,
                  num_classes=10, num_warmup=30, num_runs=20):
    """2-layer GRU training benchmark."""

    try:
        device = jax.devices("mlx")[0]
        device_name = "mlx"
    except Exception:
        device = jax.devices("cpu")[0]
        device_name = "cpu"

    print(f"Device: {device_name}")
    print(f"Config: batch={batch_size}, seq_len={seq_len}, input_dim={input_dim}, "
          f"hidden={hidden_dim}, classes={num_classes}")
    print(f"Model: 2× GRU({input_dim}→{hidden_dim}, {seq_len} steps) → Dense({hidden_dim}, relu) → Dense({num_classes})")
    print(f"Task: Forward + Backward + SGD\n")

    lr = 0.001

    with jax.default_device(device):
        rngs = nnx.Rngs(0)
        model = GRUClassifier(input_dim, hidden_dim, num_classes, rngs=rngs)
        optimizer = nnx.Optimizer(model, optax.sgd(lr), wrt=nnx.Param)

        key = jax.random.PRNGKey(42)
        k1, k2 = jax.random.split(key)
        x = jax.random.normal(k1, (batch_size, seq_len, input_dim))
        y = jax.random.randint(k2, (batch_size,), 0, num_classes)

        # Warmup
        print(f"Warming up ({num_warmup} steps)...")
        for _ in range(num_warmup):
            loss = train_step(model, optimizer, x, y)
            loss.block_until_ready()
        print("Warmup done ✓\n")

        # Timed runs
        print(f"{'='*50}")
        print("Timed runs")
        print(f"{'='*50}")
        times = []
        for i in range(num_runs):
            t0 = time.perf_counter()
            loss = train_step(model, optimizer, x, y)
            loss.block_until_ready()
            elapsed = (time.perf_counter() - t0) * 1000
            times.append(elapsed)
            print(f"  Run {i+1}: {elapsed:.2f}ms (loss: {float(loss):.4f})")

        mean_t = np.mean(times)
        std_t = np.std(times)
        min_t = np.min(times)
        print(f"\n  Mean: {mean_t:.2f}ms ± {std_t:.2f}ms  Min: {min_t:.2f}ms")

    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    print(f"JAX-MLX: {mean_t:.2f}ms ± {std_t:.2f}ms")
    return mean_t, std_t


if __name__ == "__main__":
    print("RNN (GRU) Training Benchmark — JAX-MLX")
    print("Architecture: 2-layer GRU → Dense(relu) → Dense(10)\n")
    benchmark_rnn()
