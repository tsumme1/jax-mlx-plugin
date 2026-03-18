"""
Cross-Backend Benchmark: JAX-MLX vs JAX-MPS
Tests MLP, CNN, ResNet-18, and Transformer training steps.
Uses flax.linen for maximum backend compatibility.

Usage:
  # JAX-MLX (jax conda env)
  conda run -n jax env BACKEND=mlx python benchmarks/benchmark_compare.py

  # JAX-MPS (jax_mps conda env)
  conda run -n jax_mps env BACKEND=mps python benchmarks/benchmark_compare.py
"""

import os
import sys

backend_name = os.environ.get("BACKEND", "mlx")
if backend_name == "mlx":
    os.environ.setdefault("JAX_PLATFORMS", "mlx")
elif backend_name == "mps":
    os.environ.setdefault("JAX_PLATFORMS", "cpu,mps")
else:
    print(f"Unknown BACKEND={backend_name}. Use 'mlx' or 'mps'.")
    sys.exit(1)

import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
import time
import numpy as np

print(f"Backend:  {jax.default_backend()}")
print(f"JAX:      {jax.__version__}")
print(f"Devices:  {jax.devices()}")
print()

WARMUP = 5
RUNS = 20


def block_all(pytree):
    """Block until ALL arrays in a pytree are ready."""
    leaves = jax.tree.leaves(pytree)
    for leaf in leaves:
        if hasattr(leaf, 'block_until_ready'):
            leaf.block_until_ready()


def bench(name, fn, *args):
    """Benchmark a function, reporting mean/std/min over RUNS iterations."""
    for _ in range(WARMUP):
        out = fn(*args)
        block_all(out)
    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        out = fn(*args)
        block_all(out)
        times.append((time.perf_counter() - t0) * 1000)
    mean = np.mean(times)
    std = np.std(times)
    mn = np.min(times)
    print(f"  {name:45s}  {mean:8.2f}ms ± {std:5.2f}ms  (min: {mn:.2f}ms)")
    return {"name": name, "mean": mean, "std": std, "min": mn}


# ============================================================
# 1. MLP (6-layer Dense)
# ============================================================
class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(8192)(x); x = nn.relu(x)
        x = nn.Dense(4096)(x); x = nn.relu(x)
        x = nn.Dense(4096)(x); x = nn.relu(x)
        x = nn.Dense(2048)(x); x = nn.relu(x)
        x = nn.Dense(1024)(x); x = nn.relu(x)
        x = nn.Dense(10)(x)
        return x


# ============================================================
# 2. CNN (VGG-style)
# ============================================================
class CNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(64, (3, 3), padding='SAME')(x); x = nn.relu(x)
        x = nn.Conv(64, (3, 3), padding='SAME')(x); x = nn.relu(x)
        x = nn.max_pool(x, (2, 2), strides=(2, 2))
        x = nn.Conv(128, (3, 3), padding='SAME')(x); x = nn.relu(x)
        x = nn.Conv(128, (3, 3), padding='SAME')(x); x = nn.relu(x)
        x = nn.max_pool(x, (2, 2), strides=(2, 2))
        x = nn.Conv(256, (3, 3), padding='SAME')(x); x = nn.relu(x)
        x = jnp.mean(x, axis=(1, 2))
        x = nn.Dense(128)(x); x = nn.relu(x)
        x = nn.Dense(10)(x)
        return x


# ============================================================
# 3. ResNet-18
# ============================================================
class ResBlock(nn.Module):
    out_channels: int
    stride: int = 1

    @nn.compact
    def __call__(self, x, train: bool = True):
        residual = x
        y = nn.Conv(self.out_channels, (3, 3), strides=(self.stride, self.stride),
                     padding=((1, 1), (1, 1)), use_bias=False)(x)
        y = nn.BatchNorm(use_running_average=not train)(y)
        y = nn.relu(y)
        y = nn.Conv(self.out_channels, (3, 3), padding=((1, 1), (1, 1)), use_bias=False)(y)
        y = nn.BatchNorm(use_running_average=not train)(y)

        if self.stride != 1 or x.shape[-1] != self.out_channels:
            residual = nn.Conv(self.out_channels, (1, 1), strides=(self.stride, self.stride),
                               padding='VALID', use_bias=False)(residual)
            residual = nn.BatchNorm(use_running_average=not train)(residual)
        return nn.relu(y + residual)


class ResNet18(nn.Module):
    num_classes: int = 1000

    @nn.compact
    def __call__(self, x, train: bool = True):
        x = nn.Conv(64, (7, 7), strides=(2, 2), padding=((3, 3), (3, 3)), use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        # Max pool
        x = jnp.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)), constant_values=-jnp.inf)
        x = nn.max_pool(x, (3, 3), strides=(2, 2), padding='VALID')

        for _ in range(2): x = ResBlock(64)(x, train)
        x = ResBlock(128, stride=2)(x, train)
        x = ResBlock(128)(x, train)
        x = ResBlock(256, stride=2)(x, train)
        x = ResBlock(256)(x, train)
        x = ResBlock(512, stride=2)(x, train)
        x = ResBlock(512)(x, train)

        x = jnp.mean(x, axis=(1, 2))
        return nn.Dense(self.num_classes)(x)


# ============================================================
# 4. Transformer Encoder
# ============================================================
class TransformerBlock(nn.Module):
    d_model: int = 256
    num_heads: int = 8
    d_ff: int = 512

    @nn.compact
    def __call__(self, x):
        y = nn.LayerNorm()(x)
        y = nn.SelfAttention(num_heads=self.num_heads)(y)
        x = x + y
        y = nn.LayerNorm()(x)
        y = nn.Dense(self.d_ff)(y); y = nn.gelu(y)
        y = nn.Dense(self.d_model)(y)
        return x + y


class Transformer(nn.Module):
    d_model: int = 256
    num_layers: int = 6
    num_classes: int = 10

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.d_model)(x)
        for _ in range(self.num_layers):
            x = TransformerBlock(d_model=self.d_model)(x)
        x = jnp.mean(x, axis=1)
        return nn.Dense(self.num_classes)(x)


# ============================================================
# 5. RNN (2-layer GRU via jax.lax.scan)
# ============================================================
class GRULayer(nn.Module):
    """GRU layer using lax.scan (required for AD — while_loop not differentiable)."""
    hidden_dim: int

    @nn.compact
    def __call__(self, x):
        B, T, D = x.shape
        H = self.hidden_dim

        # GRU gate parameters
        gate_in = D + H
        W_z = self.param('W_z', nn.initializers.lecun_normal(), (gate_in, H))
        b_z = self.param('b_z', nn.initializers.zeros, (H,))
        W_r = self.param('W_r', nn.initializers.lecun_normal(), (gate_in, H))
        b_r = self.param('b_r', nn.initializers.zeros, (H,))
        W_h = self.param('W_h', nn.initializers.lecun_normal(), (gate_in, H))
        b_h = self.param('b_h', nn.initializers.zeros, (H,))

        def scan_fn(h_prev, x_t):
            xh = jnp.concatenate([x_t, h_prev], axis=-1)
            z = jax.nn.sigmoid(xh @ W_z + b_z)
            r = jax.nn.sigmoid(xh @ W_r + b_r)
            xrh = jnp.concatenate([x_t, r * h_prev], axis=-1)
            h_hat = jnp.tanh(xrh @ W_h + b_h)
            h_new = (1 - z) * h_prev + z * h_hat
            return h_new, h_new

        h0 = jnp.zeros((B, H))
        x_t = jnp.transpose(x, (1, 0, 2))  # (T, B, D)
        _, all_h = jax.lax.scan(scan_fn, h0, x_t)
        return jnp.transpose(all_h, (1, 0, 2))  # (B, T, H)


class GRUClassifier(nn.Module):
    """2-layer GRU → Dense classifier using lax.scan."""
    hidden_dim: int = 512
    num_classes: int = 10

    @nn.compact
    def __call__(self, x):
        x = GRULayer(self.hidden_dim, name='gru1')(x)
        x = GRULayer(self.hidden_dim, name='gru2')(x)
        h_last = x[:, -1, :]
        h_last = nn.relu(nn.Dense(self.hidden_dim)(h_last))
        return nn.Dense(self.num_classes)(h_last)


# ============================================================
# Training step factory
# ============================================================
def make_train_step(model, num_classes, has_batch_stats=False):
    """Create a JIT-compiled training step using Adam."""

    @jax.jit
    def train_step(params, opt_state, x, y):
        def loss_fn(p):
            logits = model.apply({'params': p}, x)
            one_hot = jax.nn.one_hot(y, num_classes)
            return -jnp.mean(jnp.sum(jax.nn.log_softmax(logits) * one_hot, axis=-1))
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    @jax.jit
    def train_step_bn(variables, opt_state, x, y):
        def loss_fn(params):
            logits, updates = model.apply(
                {'params': params, **{k: variables[k] for k in variables if k != 'params'}},
                x, train=True, mutable=['batch_stats'])
            one_hot = jax.nn.one_hot(y, num_classes)
            return -jnp.mean(jnp.sum(jax.nn.log_softmax(logits) * one_hot, axis=-1)), updates
        (loss, updates), grads = jax.value_and_grad(loss_fn, has_aux=True)(variables['params'])
        opt_updates, new_opt_state = optimizer.update(grads, opt_state, variables['params'])
        new_params = optax.apply_updates(variables['params'], opt_updates)
        new_variables = {'params': new_params}
        new_variables.update(updates)
        return new_variables, new_opt_state, loss

    if has_batch_stats:
        return train_step_bn
    return train_step


# ============================================================
# Global optimizer
# ============================================================
optimizer = optax.adam(1e-3)


# ============================================================
# Run all benchmarks
# ============================================================
results = []


def run_benchmark(title, model, input_shape, num_classes, batch_size,
                  has_batch_stats=False):
    """Initialize and benchmark a model."""
    import gc

    # Clear state from previous benchmarks to avoid interference
    gc.collect()
    jax.clear_caches()

    print(f"\n{'=' * 70}")
    print(title)
    print(f"{'=' * 70}")

    # MPS can't init models (stablehlo.composite unsupported), so init on CPU
    if backend_name == "mps":
        cpu = jax.devices("cpu")[0]
        device = jax.devices("mps")[0]
        with jax.default_device(cpu):
            rng = jax.random.PRNGKey(0)
            dummy = jnp.ones(input_shape)
            if has_batch_stats:
                variables = model.init(rng, dummy, train=True)
            else:
                variables = model.init(rng, dummy)
        # Transfer to MPS
        variables = jax.device_put(variables, device)
    else:
        rng = jax.random.PRNGKey(0)
        dummy = jnp.ones(input_shape)
        if has_batch_stats:
            variables = model.init(rng, dummy, train=True)
        else:
            variables = model.init(rng, dummy)

    opt_state = optimizer.init(variables['params'])
    train_step = make_train_step(model, num_classes, has_batch_stats=has_batch_stats)

    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)
    x = jax.random.normal(k1, input_shape)
    y = jax.random.randint(k2, (batch_size,), 0, num_classes)

    if has_batch_stats:
        r = bench(title, train_step, variables, opt_state, x, y)
    else:
        r = bench(title, train_step, variables['params'], opt_state, x, y)

    results.append(r)


if __name__ == '__main__':
    print(f"{'=' * 70}")
    print(f"Cross-Backend Benchmark — {backend_name.upper()}")
    print(f"{'=' * 70}")

    # 1. MLP
    run_benchmark(
        "MLP  (batch=2048, 2048→8192→...→10, Adam)",
        MLP(), (2048, 2048), 10, 2048)

    # 2. CNN
    run_benchmark(
        "CNN  (batch=128, 64×64×3, Adam)",
        CNN(), (128, 64, 64, 3), 10, 128)

    # 3. ResNet-18
    run_benchmark(
        "ResNet-18  (batch=32, 224×224×3, Adam)",
        ResNet18(), (32, 224, 224, 3), 1000, 32,
        has_batch_stats=True)

    # 4. Transformer
    run_benchmark(
        "Transformer  (batch=128, seq=64, d=256, 6L, Adam)",
        Transformer(), (128, 64, 256), 10, 128)

    # 5. RNN (2-layer GRU)
    run_benchmark(
        "RNN/GRU  (batch=64, seq=128, d=128, h=512, Adam)",
        GRUClassifier(), (64, 128, 128), 10, 64)

    # Summary
    print(f"\n{'=' * 70}")
    print(f"SUMMARY — {backend_name.upper()}")
    print(f"{'=' * 70}")
    for r in results:
        print(f"  {r['name']:50s}  mean={r['mean']:.1f}ms  min={r['min']:.1f}ms")
    print()

