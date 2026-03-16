"""
mlx_repro.py

Minimal reproducible example of MLX backend hanging during compilation 
of `jax.lax.scan` operations involving Neural ODE physics solvers.

This script tests three ODE solvers:
1. Sequential RK4 (unrolled scan)
2. Sequential RK4 (lax.scan length=T)
3. quasi-ELK Parallel RK4 (associative scan)

Run:
  # Works instantly
  JAX_PLATFORMS=cpu python mlx_repro.py

  # Hangs indefinitely during `jax.jit` compilation
  JAX_PLATFORMS=mlx python mlx_repro.py
"""

import time
import jax
import jax.numpy as jnp
from jax import random, lax, vmap
from functools import partial

# Check backend
backend = jax.default_backend()
print(f"Running on JAX backend: {backend}")

# ── 1. Define typical Neural ODE dynamics ──
STATE_DIM = 192
SEQ_LEN = 1000

def silu(x):
    return x * jax.nn.sigmoid(x)

def mlp_fwd(params, x):
    for w, b in params[:-1]:
        x = silu(x @ w + b)
    w, b = params[-1]
    return x @ w + b

def init_mlp(key, layer_sizes):
    keys = random.split(key, len(layer_sizes) - 1)
    params = []
    for k, d_in, d_out in zip(keys, layer_sizes[:-1], layer_sizes[1:]):
        k1, k2 = random.split(k)
        w = random.normal(k1, (d_in, d_out)) * jnp.sqrt(2.0 / d_in)
        b = jnp.zeros((d_out,))
        params.append((w, b))
    return params

# Initialize dummy models
rng = random.PRNGKey(0)
cb_drift = init_mlp(rng, [66, 64])
cb_readout = init_mlp(rng, [66, 2])
mtr_drift = init_mlp(rng, [65, 64])

def f_continuous(state, u1, u2):
    """192-state composed Neural ODE RHS (64×3 sub-systems)"""
    cb_in = jnp.concatenate([state[:64], jnp.array([u1, u2])])
    d_cb = mlp_fwd(cb_drift, cb_in)
    i_col = mlp_fwd(cb_readout, cb_in)
    
    d_m1 = mlp_fwd(mtr_drift, jnp.concatenate([state[64:128], jnp.array([i_col[0]])]))
    d_m2 = mlp_fwd(mtr_drift, jnp.concatenate([state[128:192], jnp.array([i_col[1]])]))
    
    return jnp.concatenate([d_cb, d_m1, d_m2])

# ── 2. Solvers ──
DT = 1e-4

def solve_seq_scan(u0, u1_seq, u2_seq):
    def step(st, inputs):
        v1, v2 = inputs
        k1 = f_continuous(st, v1, v2)
        k2 = f_continuous(st + 0.5 * DT * k1, v1, v2)
        k3 = f_continuous(st + 0.5 * DT * k2, v1, v2)
        k4 = f_continuous(st + DT * k3, v1, v2)
        nxt = st + (DT / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return nxt, nxt
    _, traj = jax.lax.scan(step, u0, (u1_seq, u2_seq))
    return traj

# Use the exact quasi-ELK associative scan implementation for Par RK4
from quasi_elk_repro import parallel_rk4

def solve_par_elk(u0, u1_seq, u2_seq):
    return parallel_rk4(f_continuous, u0, u1_seq, u2_seq, DT, num_iters=2)


# ── 3. Test Harness ──
def main():
    u0 = jnp.zeros(STATE_DIM)
    u1_seq = jnp.ones(SEQ_LEN)
    u2_seq = jnp.zeros(SEQ_LEN)
    
    print("\n[1] Testing Sequential RK4 (lax.scan)")
    t0 = time.time()
    jit_seq = jax.jit(solve_seq_scan)
    try:
        # High likelihood of MLX hanging here
        _ = jit_seq(u0, u1_seq, u2_seq).block_until_ready()
        print(f"    Compiled and ran in {time.time()-t0:.2f}s")
    except Exception as e:
        print(f"    Failed: {e}")
        
    print("\n[2] Testing Parallel RK4 (quasi-ELK associative scan)")
    t0 = time.time()
    jit_par = jax.jit(solve_par_elk)
    try:
        # High likelihood of MLX hanging here
        _ = jit_par(u0, u1_seq, u2_seq).block_until_ready()
        print(f"    Compiled and ran in {time.time()-t0:.2f}s")
    except Exception as e:
        print(f"    Failed: {e}")

if __name__ == "__main__":
    main()
