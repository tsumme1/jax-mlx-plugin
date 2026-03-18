"""
Mamba SSM Benchmark: Sequential vs Parallel Scan — JAX-MLX

Compares two implementations of the Mamba selective state-space model:
  1. Sequential: lax.scan (step-by-step recurrence, fully unrolled)
  2. Parallel:   lax.associative_scan (Blelloch parallel prefix)

Both implement the same discretized SSM:
  h[t] = A_bar[t] * h[t-1] + B_bar[t] * x[t]
  y[t] = C[t] · h[t]

The parallel version uses the associative binary operator:
  (a1, b1) ⊕ (a2, b2) = (a2 * a1, a2 * b1 + b2)

Config: batch=32, seq_len=1024, d_model=256, d_state=16
Task: Forward + Backward + SGD

Usage:
  python benchmarks/benchmark_mamba.py
"""

import os
if "MLX_PJRT_DEBUG" in os.environ:
    del os.environ["MLX_PJRT_DEBUG"]

import jax
import jax.numpy as jnp
import jax.lax as lax
from flax import nnx
import optax
import time
import numpy as np


# ── Mamba block (sequential scan via lax.scan, fully unrolled) ──────────────

class MambaBlockSequential(nnx.Module):
    """Simplified Mamba block using sequential lax.scan (fully unrolled)."""

    def __init__(self, d_model: int, d_state: int, *, rngs: nnx.Rngs):
        D, N = d_model, d_state
        self.z_proj  = nnx.Linear(D, D, rngs=rngs)
        self.x_proj  = nnx.Linear(D, D, rngs=rngs)
        self.B_proj  = nnx.Linear(D, N, rngs=rngs)
        self.C_proj  = nnx.Linear(D, N, rngs=rngs)
        self.dt_proj = nnx.Linear(D, D, rngs=rngs)
        self.A_log   = nnx.Param(jnp.log(jnp.broadcast_to(
            jnp.arange(1, N + 1, dtype=jnp.float32), (D, N))))
        self.out_proj = nnx.Linear(D, D, rngs=rngs)
        self.D_ = D
        self.N_ = N

    def __call__(self, x):
        """x: (batch, seq_len, d_model) → (batch, seq_len, d_model)"""
        B, L, D = x.shape
        N = self.N_

        z = jax.nn.sigmoid(self.z_proj(x))
        x_ssm = self.x_proj(x)
        Bm = self.B_proj(x)
        C = self.C_proj(x)
        dt = jax.nn.softplus(self.dt_proj(x))

        A = -jnp.exp(self.A_log[...])
        A_bar = jnp.exp(dt[..., :, None] * A[None, None, :, :])  # (B, L, D, N)
        B_bar = dt[..., :, None] * Bm[..., None, :]              # (B, L, D, N)
        x_db  = x_ssm[..., :, None] * B_bar                      # (B, L, D, N)

        # Sequential scan: time axis → 0 for lax.scan
        a_scan = jnp.transpose(A_bar, (1, 0, 2, 3))              # (L, B, D, N)
        x_scan = jnp.transpose(x_db, (1, 0, 2, 3))
        C_exp  = jnp.broadcast_to(C[..., None, :], (B, L, D, N))
        c_scan = jnp.transpose(C_exp, (1, 0, 2, 3))

        def ssm_step(h, inp):
            a_t, x_t, c_t = inp
            h_new = a_t * h + x_t
            y_t = jnp.sum(h_new * c_t, axis=-1)
            return h_new, y_t

        h0 = jnp.zeros((B, D, N))
        _, y_ssm = lax.scan(ssm_step, h0, (a_scan, x_scan, c_scan),
                            unroll=1)
        y_ssm = jnp.transpose(y_ssm, (1, 0, 2))  # (B, L, D)

        y = y_ssm * z
        return self.out_proj(y)


# ── Mamba block (parallel scan via lax.associative_scan) ────────────────────

class MambaBlockParallel(nnx.Module):
    """Simplified Mamba block using parallel lax.associative_scan."""

    def __init__(self, d_model: int, d_state: int, *, rngs: nnx.Rngs):
        D, N = d_model, d_state
        self.z_proj  = nnx.Linear(D, D, rngs=rngs)
        self.x_proj  = nnx.Linear(D, D, rngs=rngs)
        self.B_proj  = nnx.Linear(D, N, rngs=rngs)
        self.C_proj  = nnx.Linear(D, N, rngs=rngs)
        self.dt_proj = nnx.Linear(D, D, rngs=rngs)
        self.A_log   = nnx.Param(jnp.log(jnp.broadcast_to(
            jnp.arange(1, N + 1, dtype=jnp.float32), (D, N))))
        self.out_proj = nnx.Linear(D, D, rngs=rngs)
        self.D_ = D
        self.N_ = N

    def __call__(self, x):
        """x: (batch, seq_len, d_model) → (batch, seq_len, d_model)"""
        B, L, D = x.shape
        N = self.N_

        z = jax.nn.sigmoid(self.z_proj(x))
        x_ssm = self.x_proj(x)
        Bm = self.B_proj(x)
        C = self.C_proj(x)
        dt = jax.nn.softplus(self.dt_proj(x))

        A = -jnp.exp(self.A_log[...])
        A_bar = jnp.exp(dt[..., :, None] * A[None, None, :, :])
        B_bar = dt[..., :, None] * Bm[..., None, :]
        x_db  = x_ssm[..., :, None] * B_bar

        # Parallel associative scan over axis=1 (seq_len)
        def combine(ab1, ab2):
            a1, b1 = ab1
            a2, b2 = ab2
            return a2 * a1, a2 * b1 + b2

        _, h = lax.associative_scan(combine, (A_bar, x_db), axis=1)

        C_exp = jnp.broadcast_to(C[..., None, :], (B, L, D, N))
        y_ssm = jnp.sum(h * C_exp, axis=-1)

        y = y_ssm * z
        return self.out_proj(y)


# ── Classifier wrapper ─────────────────────────────────────────────────────

class MambaClassifier(nnx.Module):
    """Mamba → Dense(relu) → Dense(num_classes)."""

    def __init__(self, d_model: int, d_state: int, num_classes: int,
                 *, parallel: bool, rngs: nnx.Rngs):
        BlockCls = MambaBlockParallel if parallel else MambaBlockSequential
        self.mamba = BlockCls(d_model, d_state, rngs=rngs)
        self.dense1 = nnx.Linear(d_model, d_model, rngs=rngs)
        self.dense2 = nnx.Linear(d_model, num_classes, rngs=rngs)

    def __call__(self, x):
        h = self.mamba(x)            # (batch, seq_len, d_model)
        h = h[:, -1, :]              # last timestep → (batch, d_model)
        h = jax.nn.relu(self.dense1(h))
        return self.dense2(h)


def cross_entropy_loss(model, x, y):
    logits = model(x)
    one_hot = jax.nn.one_hot(y, logits.shape[-1])
    return -jnp.mean(jnp.sum(jax.nn.log_softmax(logits) * one_hot, axis=-1))


@nnx.jit
def train_step(model, optimizer, x, y):
    loss, grads = nnx.value_and_grad(cross_entropy_loss)(model, x, y)
    optimizer.update(model, grads)
    return loss


# ── Benchmark harness ──────────────────────────────────────────────────────

def benchmark_mamba(variant: str, batch_size=32, seq_len=1024, d_model=256,
                    d_state=16, num_classes=10, num_warmup=10, num_runs=20):
    """Benchmark a single Mamba variant (sequential or parallel)."""

    parallel = (variant == "parallel")

    device = jax.devices()[0]
    cpu = jax.devices("cpu")[0]
    device_name = jax.default_backend()

    scan_type = "lax.associative_scan (parallel)" if parallel else "lax.scan (sequential, unrolled)"

    print(f"  Device: {device_name}")
    print(f"  Scan:   {scan_type}")
    print(f"  Config: batch={batch_size}, seq_len={seq_len}, d_model={d_model}, d_state={d_state}")
    print(f"  Task:   Forward + Backward + SGD\n")

    lr = 0.001

    with jax.default_device(cpu):
        rngs = nnx.Rngs(0)
        model = MambaClassifier(d_model, d_state, num_classes,
                                parallel=parallel, rngs=rngs)
        optimizer = nnx.Optimizer(model, optax.sgd(lr), wrt=nnx.Param)

        key = jax.random.PRNGKey(42)
        k1, k2 = jax.random.split(key)
        x = jax.random.normal(k1, (batch_size, seq_len, d_model))
        y = jax.random.randint(k2, (batch_size,), 0, num_classes)

    # Transfer to target device
    model_state = nnx.state(model)
    model_state = jax.device_put(model_state, device)
    nnx.update(model, model_state)
    opt_state = nnx.state(optimizer)
    opt_state = jax.device_put(opt_state, device)
    nnx.update(optimizer, opt_state)
    x = jax.device_put(x, device)
    y = jax.device_put(y, device)

    # Warmup (includes JIT compilation)
    print(f"  Compiling + warming up ({num_warmup} steps)...")
    t_compile = time.perf_counter()
    for i in range(num_warmup):
        loss = train_step(model, optimizer, x, y)
        loss.block_until_ready()
        if i == 0:
            compile_time = time.perf_counter() - t_compile
            print(f"  First step (compile): {compile_time:.1f}s")
    print("  Warmup done ✓\n")

    # Timed runs
    print(f"  {'─'*40}")
    times = []
    for i in range(num_runs):
        t0 = time.perf_counter()
        loss = train_step(model, optimizer, x, y)
        loss.block_until_ready()
        elapsed = (time.perf_counter() - t0) * 1000
        times.append(elapsed)
        print(f"    Run {i+1:2d}: {elapsed:7.2f}ms  (loss: {float(loss):.4f})")

    mean_t = np.mean(times)
    std_t = np.std(times)
    min_t = np.min(times)
    print(f"\n  Mean: {mean_t:.2f}ms ± {std_t:.2f}ms  Min: {min_t:.2f}ms")

    return mean_t, std_t, min_t


if __name__ == "__main__":
    print("Mamba SSM Benchmark — JAX-MLX")
    print("Sequential (lax.scan) vs Parallel (lax.associative_scan)")
    print(f"{'='*60}\n")

    print(f"{'='*60}")
    print("SEQUENTIAL (lax.scan, fully unrolled)")
    print(f"{'='*60}")
    seq_mean, seq_std, seq_min = benchmark_mamba("sequential")

    print(f"\n{'='*60}")
    print("PARALLEL (lax.associative_scan)")
    print(f"{'='*60}")
    par_mean, par_std, par_min = benchmark_mamba("parallel")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Sequential (lax.scan, unrolled):    {seq_mean:7.2f}ms ± {seq_std:.2f}ms  (min: {seq_min:.2f}ms)")
    print(f"  Parallel   (lax.associative_scan):  {par_mean:7.2f}ms ± {par_std:.2f}ms  (min: {par_min:.2f}ms)")

    if par_mean < seq_mean:
        speedup = seq_mean / par_mean
        print(f"\n  ⚡ Parallel scan is {speedup:.2f}× faster")
    else:
        slowdown = par_mean / seq_mean
        print(f"\n  Sequential scan is {slowdown:.2f}× faster")
