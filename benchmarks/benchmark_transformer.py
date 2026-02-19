"""
Transformer Encoder Training Benchmark: JAX-MLX
Full training loop with forward + backward + SGD.
Uses multi-head self-attention + FFN blocks.

Usage:
  python benchmarks/benchmark_transformer_train.py
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


def sinusoidal_pos_encoding(seq_len, d_model):
    """Create sinusoidal positional encoding (seq_len, d_model)."""
    pos = np.arange(seq_len)[:, None].astype(np.float32)
    div = np.exp(np.arange(0, d_model, 2).astype(np.float32) * -(np.log(10000.0) / d_model))
    pe = np.zeros((seq_len, d_model), dtype=np.float32)
    pe[:, 0::2] = np.sin(pos * div)
    pe[:, 1::2] = np.cos(pos * div)
    return pe


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention."""
    d_model: int
    num_heads: int

    @nn.compact
    def __call__(self, x):
        batch, seq_len, d = x.shape
        head_dim = self.d_model // self.num_heads

        # QKV projections
        Q = nn.Dense(self.d_model, name='q')(x)
        K = nn.Dense(self.d_model, name='k')(x)
        V = nn.Dense(self.d_model, name='v')(x)

        # Reshape to (batch, seq_len, num_heads, head_dim) for SDPA
        Q = Q.reshape(batch, seq_len, self.num_heads, head_dim)
        K = K.reshape(batch, seq_len, self.num_heads, head_dim)
        V = V.reshape(batch, seq_len, self.num_heads, head_dim)

        # Fused scaled dot-product attention (expects B, T, N, H layout)
        out = jax.nn.dot_product_attention(Q, K, V)  # (B, T, N, H)

        # Concatenate heads
        out = out.reshape(batch, seq_len, self.d_model)
        return nn.Dense(self.d_model, name='out')(out)


class TransformerBlock(nn.Module):
    """Single Transformer encoder block: self-attention + FFN with pre-norm."""
    d_model: int = 256
    num_heads: int = 8
    d_ff: int = 512

    @nn.compact
    def __call__(self, x):
        # Pre-norm self-attention
        y = nn.LayerNorm()(x)
        y = MultiHeadAttention(d_model=self.d_model, num_heads=self.num_heads)(y)
        x = x + y

        # Pre-norm FFN
        y = nn.LayerNorm()(x)
        y = nn.Dense(features=self.d_ff)(y)
        y = nn.gelu(y)
        y = nn.Dense(features=self.d_model)(y)
        x = x + y
        return x


class TransformerClassifier(nn.Module):
    """Transformer encoder stack → mean pooling → classification head."""
    d_model: int = 256
    num_heads: int = 8
    d_ff: int = 512
    num_layers: int = 6
    num_classes: int = 10
    seq_len: int = 64

    @nn.compact
    def __call__(self, x):
        # Input projection
        x = nn.Dense(features=self.d_model)(x)

        # Add positional encoding (constant, computed in numpy)
        pe = self.variable('constants', 'pos_enc',
                           lambda: jnp.array(sinusoidal_pos_encoding(self.seq_len, self.d_model)))
        x = x + pe.value[None, :, :]

        # Transformer encoder layers
        for i in range(self.num_layers):
            x = TransformerBlock(
                d_model=self.d_model,
                num_heads=self.num_heads,
                d_ff=self.d_ff,
                name=f'block_{i}',
            )(x)

        # Mean pooling over sequence dimension
        x = jnp.mean(x, axis=1)

        # Classification head
        x = nn.Dense(features=self.num_classes)(x)
        return x


def benchmark_mlx(batch_size=128, seq_len=64, d_model=256, num_warmup=30, num_runs=20):
    """Benchmark Transformer training on MLX."""
    try:
        mlx = jax.devices("mlx")[0]
    except Exception as e:
        print(f"MLX device not available: {e}")
        return None, None

    model = TransformerClassifier(d_model=d_model, seq_len=seq_len)

    try:
        cpu = jax.devices("cpu")[0]
    except Exception:
        cpu = mlx

    with jax.default_device(cpu):
        rng = jax.random.PRNGKey(0)
        variables = model.init(rng, jnp.ones((batch_size, seq_len, d_model)))
        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, (batch_size, seq_len, d_model))
        y = jax.random.randint(key, (batch_size,), 0, 10)

    with jax.default_device(mlx):
        variables = jax.device_put(variables, mlx)
        x = jax.device_put(x, mlx)
        y = jax.device_put(y, mlx)

        @jax.jit
        def train_step(variables, x, y):
            def loss_fn_inner(variables):
                logits = model.apply(variables, x)
                one_hot = jax.nn.one_hot(y, 10)
                return -jnp.mean(jnp.sum(jax.nn.log_softmax(logits) * one_hot, axis=-1))

            loss, grads = jax.value_and_grad(loss_fn_inner)(variables)
            lr = 0.01
            new_params = jax.tree.map(lambda p, g: p - lr * g,
                                       variables['params'], grads['params'])
            new_variables = dict(variables)
            new_variables['params'] = new_params
            return new_variables, loss

        # Warmup
        print(f"Warming up ({num_warmup} steps)...")
        for _ in range(num_warmup):
            variables, loss = train_step(variables, x, y)
            loss.block_until_ready()

        # Timed runs
        print(f"\n{'='*50}")
        print("Timed runs")
        print(f"{'='*50}")
        times = []
        for i in range(num_runs):
            start = time.perf_counter()
            variables, loss = train_step(variables, x, y)
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
    seq_len = 64
    d_model = 256

    print("Transformer Training Benchmark (JAX-MLX)")
    print(f"Config: batch_size={batch_size}, seq_len={seq_len}, d_model={d_model}")
    print(f"Model: 6× TransformerBlock(d=256, 8 heads, ff=512) → MeanPool → Dense(10)")
    print(f"Task: Forward + Backward pass with SGD\n")

    mean, std = benchmark_mlx(batch_size, seq_len, d_model)

    if mean is not None:
        print(f"\n{'='*50}")
        print("SUMMARY")
        print(f"{'='*50}")
        print(f"JAX-MLX: {mean:.2f}ms ± {std:.2f}ms")


if __name__ == "__main__":
    main()
