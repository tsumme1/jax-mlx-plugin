"""
RNN (GRU) Training Benchmark: Native MLX
2-layer GRU → Dense(relu) → Dense(10) with SGD.
Matches the architecture of JAX-MLX and Julia benchmarks.

Usage:
  python benchmarks/benchmark_rnn_native.py
"""

import mlx.core as mx
import time
import numpy as np


def forward(x, params):
    """Forward pass: 2-layer GRU + dense head."""
    (Wi1, bi1, Wh1, bh1, Wi2, bi2, Wh2, bh2, Wd1, bd1, Wd2, bd2) = params

    batch_size, seq_len, input_dim = x.shape
    hidden_dim = Wh1.shape[1] // 3  # Wh1 is (hidden, 3*hidden)

    # --- Layer 1: GRU over input ---
    # Pre-project input: (batch, seq, 3*hidden)
    x_proj1 = mx.matmul(x.reshape(-1, input_dim), Wi1).reshape(batch_size, seq_len, 3 * hidden_dim) + bi1

    h1 = mx.zeros((batch_size, hidden_dim))
    h1_all = []
    for t in range(seq_len):
        x_t = x_proj1[:, t, :]
        hx = mx.matmul(h1, Wh1) + bh1
        z = mx.sigmoid(x_t[:, :hidden_dim] + hx[:, :hidden_dim])
        r = mx.sigmoid(x_t[:, hidden_dim:2*hidden_dim] + hx[:, hidden_dim:2*hidden_dim])
        n = mx.tanh(x_t[:, 2*hidden_dim:] + r * hx[:, 2*hidden_dim:])
        h1 = (1 - z) * n + z * h1
        h1_all.append(h1)

    # Stack layer 1 outputs: (batch, seq_len, hidden)
    h1_seq = mx.stack(h1_all, axis=1)

    # --- Layer 2: GRU over layer 1's hidden states ---
    x_proj2 = mx.matmul(h1_seq.reshape(-1, hidden_dim), Wi2).reshape(batch_size, seq_len, 3 * hidden_dim) + bi2

    h2 = mx.zeros((batch_size, hidden_dim))
    for t in range(seq_len):
        x_t = x_proj2[:, t, :]
        hx = mx.matmul(h2, Wh2) + bh2
        z = mx.sigmoid(x_t[:, :hidden_dim] + hx[:, :hidden_dim])
        r = mx.sigmoid(x_t[:, hidden_dim:2*hidden_dim] + hx[:, hidden_dim:2*hidden_dim])
        n = mx.tanh(x_t[:, 2*hidden_dim:] + r * hx[:, 2*hidden_dim:])
        h2 = (1 - z) * n + z * h2

    # Dense layers
    h = mx.maximum(mx.matmul(h2, Wd1) + bd1, 0)
    logits = mx.matmul(h, Wd2) + bd2
    return logits


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


def benchmark_native(batch_size=128, seq_len=64, input_dim=64, hidden_dim=512,
                     num_classes=10, num_warmup=30, num_runs=20):
    print(f"\n{'='*60}")
    print(f"RNN (GRU) Training Benchmark: Native MLX (functional compiled)")
    print(f"{'='*60}")
    print(f"Config: batch_size={batch_size}, seq_len={seq_len}, input_dim={input_dim}")
    print(f"Model: 2× GRU({input_dim}→{hidden_dim}, {seq_len} steps) → Dense({hidden_dim}, relu) → Dense({num_classes})")

    lr = 0.001

    # Layer 1: input projection + recurrent weights
    scale = (2.0 / input_dim) ** 0.5
    Wi1 = mx.random.normal((input_dim, 3 * hidden_dim)) * scale
    bi1 = mx.zeros((3 * hidden_dim,))
    scale_h = (2.0 / hidden_dim) ** 0.5
    Wh1 = mx.random.normal((hidden_dim, 3 * hidden_dim)) * scale_h
    bh1 = mx.zeros((3 * hidden_dim,))

    # Layer 2: hidden→hidden GRU
    Wi2 = mx.random.normal((hidden_dim, 3 * hidden_dim)) * scale_h
    bi2 = mx.zeros((3 * hidden_dim,))
    Wh2 = mx.random.normal((hidden_dim, 3 * hidden_dim)) * scale_h
    bh2 = mx.zeros((3 * hidden_dim,))

    # Dense layers
    Wd1 = mx.random.normal((hidden_dim, hidden_dim)) * scale_h
    bd1 = mx.zeros((hidden_dim,))
    Wd2 = mx.random.normal((hidden_dim, num_classes)) * scale_h
    bd2 = mx.zeros((num_classes,))

    params = [Wi1, bi1, Wh1, bh1, Wi2, bi2, Wh2, bh2, Wd1, bd1, Wd2, bd2]
    mx.eval(params)

    # Dummy data
    x = mx.random.normal((batch_size, seq_len, input_dim))
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
