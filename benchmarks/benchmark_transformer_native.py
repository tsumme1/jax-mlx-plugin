"""
Transformer Encoder Training Benchmark: Native MLX
Full training loop with mx.compile, forward + backward + SGD.
Manual multi-head attention implementation.

Usage:
  python benchmarks/benchmark_transformer_native.py
"""

import mlx.core as mx
import mlx.nn as mlx_nn
import time
import numpy as np
import math


def layer_norm(x, gamma, beta, eps=1e-5):
    """Layer normalization."""
    mean = mx.mean(x, axis=-1, keepdims=True)
    var = mx.var(x, axis=-1, keepdims=True)
    return gamma * (x - mean) / mx.sqrt(var + eps) + beta


def multi_head_attention(x, Wq, Wk, Wv, Wo, num_heads):
    """Multi-head self-attention using fused SDPA kernel."""
    batch, seq_len, d_model = x.shape
    head_dim = d_model // num_heads

    Q = mx.matmul(x, Wq)
    K = mx.matmul(x, Wk)
    V = mx.matmul(x, Wv)

    Q = Q.reshape(batch, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    K = K.reshape(batch, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    V = V.reshape(batch, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)

    scale = 1.0 / math.sqrt(head_dim)
    out = mx.fast.scaled_dot_product_attention(Q, K, V, scale=scale)

    out = out.transpose(0, 2, 1, 3).reshape(batch, seq_len, d_model)
    return mx.matmul(out, Wo)


def transformer_block(x, Wq, Wk, Wv, Wo,
                      W1, b1, W2, b2,
                      ln1_gamma, ln1_beta, ln2_gamma, ln2_beta,
                      num_heads):
    """Single Transformer encoder block with pre-norm LayerNorm."""
    # Pre-norm self-attention + residual
    y = layer_norm(x, ln1_gamma, ln1_beta)
    y = multi_head_attention(y, Wq, Wk, Wv, Wo, num_heads)
    x = x + y

    # Pre-norm FFN + residual
    y = layer_norm(x, ln2_gamma, ln2_beta)
    y = mlx_nn.gelu(mx.matmul(y, W1) + b1)
    y = mx.matmul(y, W2) + b2
    x = x + y
    return x


def make_forward(num_heads, num_layers, params_per_block):
    def forward(x, params):
        W_in, b_in, W_out, b_out, pos_enc = params[0], params[1], params[2], params[3], params[4]
        block_params_flat = params[5:]

        batch, seq_len, input_dim = x.shape
        d_model = W_in.shape[1]

        x = mx.matmul(x.reshape(-1, input_dim), W_in).reshape(batch, seq_len, d_model) + b_in
        x = x + pos_enc

        for i in range(num_layers):
            start = i * params_per_block
            bp = block_params_flat[start:start + params_per_block]
            x = transformer_block(x, bp[0], bp[1], bp[2], bp[3],
                                  bp[4], bp[5], bp[6], bp[7],
                                  bp[8], bp[9], bp[10], bp[11],
                                  num_heads)

        x = mx.mean(x, axis=1)
        logits = mx.matmul(x, W_out) + b_out
        return logits
    return forward


def cross_entropy(logits, labels):
    """Numerically stable cross-entropy."""
    max_logits = mx.max(logits, axis=-1, keepdims=True)
    shifted = logits - max_logits
    log_softmax = shifted - mx.log(mx.sum(mx.exp(shifted), axis=-1, keepdims=True))
    indices = mx.expand_dims(labels, -1)
    log_probs = mx.take_along_axis(log_softmax, indices.astype(mx.int32), axis=1)
    return -mx.mean(log_probs)


def benchmark_native(batch_size=128, seq_len=64, d_model=256, num_heads=8,
                     d_ff=512, num_layers=6, num_classes=10,
                     num_warmup=30, num_runs=20):
    print(f"\n{'='*60}")
    print(f"Transformer Training Benchmark: Native MLX (functional compiled)")
    print(f"{'='*60}")
    print(f"Config: batch_size={batch_size}, seq_len={seq_len}, d_model={d_model}")
    print(f"Model: {num_layers}× TransformerBlock(d={d_model}, {num_heads} heads, ff={d_ff}) → MeanPool → Dense({num_classes})")

    lr = 0.01
    params_per_block = 12  # Wq, Wk, Wv, Wo, W1, b1, W2, b2, ln1_gamma, ln1_beta, ln2_gamma, ln2_beta

    def init_block(d_model, d_ff):
        scale = (2.0 / d_model) ** 0.5
        Wq = mx.random.normal((d_model, d_model)) * scale
        Wk = mx.random.normal((d_model, d_model)) * scale
        Wv = mx.random.normal((d_model, d_model)) * scale
        Wo = mx.random.normal((d_model, d_model)) * scale
        W1 = mx.random.normal((d_model, d_ff)) * scale
        b1 = mx.zeros((d_ff,))
        W2 = mx.random.normal((d_ff, d_model)) * (2.0 / d_ff) ** 0.5
        b2 = mx.zeros((d_model,))
        ln1_gamma = mx.ones((d_model,))
        ln1_beta = mx.zeros((d_model,))
        ln2_gamma = mx.ones((d_model,))
        ln2_beta = mx.zeros((d_model,))
        return [Wq, Wk, Wv, Wo, W1, b1, W2, b2, ln1_gamma, ln1_beta, ln2_gamma, ln2_beta]

    scale = (2.0 / d_model) ** 0.5
    W_in = mx.random.normal((d_model, d_model)) * scale
    b_in = mx.zeros((d_model,))
    W_out = mx.random.normal((d_model, num_classes)) * scale
    b_out = mx.zeros((num_classes,))

    # Positional encoding
    pos = np.arange(seq_len)[:, None].astype(np.float32)
    div = np.exp(np.arange(0, d_model, 2).astype(np.float32) * -(np.log(10000.0) / d_model))
    pe_np = np.zeros((seq_len, d_model), dtype=np.float32)
    pe_np[:, 0::2] = np.sin(pos * div)
    pe_np[:, 1::2] = np.cos(pos * div)
    pos_enc = mx.array(pe_np)[None, :, :]

    block_params = []
    for _ in range(num_layers):
        block_params.extend(init_block(d_model, d_ff))

    params = [W_in, b_in, W_out, b_out, pos_enc] + block_params
    mx.eval(params)

    x = mx.random.normal((batch_size, seq_len, d_model))
    y = mx.random.randint(0, 10, (batch_size,))
    mx.eval(x, y)

    forward = make_forward(num_heads, num_layers, params_per_block)

    def loss_fn(params, x, y):
        logits = forward(x, params)
        return cross_entropy(logits, y)

    grad_fn = mx.value_and_grad(loss_fn)

    def train_step(params, x, y):
        loss, grads = grad_fn(params, x, y)
        new_params = [p - lr * g for p, g in zip(params, grads)]
        return [loss] + new_params

    compiled_train_step = mx.compile(train_step)

    print(f"\nWarmup ({num_warmup} runs)...")
    for _ in range(num_warmup):
        result = compiled_train_step(params, x, y)
        mx.eval(result)
        params = result[1:]

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
