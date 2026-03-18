"""
Model Benchmarks: ResNet-18, ViT-Base, 5-Stacked LSTM
Uses Flax NNX. Run with JAX_PLATFORMS=mlx or on jax_mps env.
"""
import jax
import jax.numpy as jnp
from jax import random
from flax import nnx
import time

backend = jax.default_backend()
print(f"Backend: {backend}")
print(f"JAX: {jax.__version__}")
print(f"Devices: {jax.devices()}")
print()

WARMUP = 5
RUNS = 20

def bench(name, fn, *args):
    for _ in range(WARMUP):
        out = fn(*args)
        jax.tree.map(lambda x: x.block_until_ready() if hasattr(x, 'block_until_ready') else None, out)
    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        out = fn(*args)
        jax.tree.map(lambda x: x.block_until_ready() if hasattr(x, 'block_until_ready') else None, out)
        times.append((time.perf_counter() - t0) * 1000)
    mean = sum(times) / len(times)
    std = (sum((t - mean)**2 for t in times) / len(times)) ** 0.5
    print(f"  {name:40s}  {mean:8.2f}ms ± {std:5.2f}ms  (min: {min(times):.2f}ms)")
    return mean

# ============================================================
# 1. ResNet-18 (NNX)
# ============================================================
class ResBlock(nnx.Module):
    def __init__(self, in_channels, out_channels, stride=1, *, rngs):
        self.stride = stride
        self.conv1 = nnx.Conv(in_channels, out_channels, kernel_size=(3, 3),
                              strides=(stride, stride), padding=((1,1),(1,1)), use_bias=False, rngs=rngs)
        self.bn1 = nnx.BatchNorm(out_channels, rngs=rngs)
        self.conv2 = nnx.Conv(out_channels, out_channels, kernel_size=(3, 3),
                              padding=((1,1),(1,1)), use_bias=False, rngs=rngs)
        self.bn2 = nnx.BatchNorm(out_channels, rngs=rngs)

        self.use_shortcut = (stride != 1 or in_channels != out_channels)
        if self.use_shortcut:
            self.shortcut = nnx.Conv(in_channels, out_channels, kernel_size=(1, 1),
                                     strides=(stride, stride), padding='VALID', use_bias=False, rngs=rngs)
            self.shortcut_bn = nnx.BatchNorm(out_channels, rngs=rngs)

    def __call__(self, x):
        residual = x
        x = nnx.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        if self.use_shortcut:
            residual = self.shortcut_bn(self.shortcut(residual))
        return nnx.relu(x + residual)


class ResNet18(nnx.Module):
    def __init__(self, num_classes=1000, *, rngs):
        self.conv1 = nnx.Conv(3, 64, kernel_size=(7, 7), strides=(2, 2),
                              padding=((3,3),(3,3)), use_bias=False, rngs=rngs)
        self.bn1 = nnx.BatchNorm(64, rngs=rngs)

        self.layer1 = nnx.List([ResBlock(64, 64, rngs=rngs) for _ in range(2)])
        self.layer2 = nnx.List([ResBlock(64, 128, stride=2, rngs=rngs),
                                ResBlock(128, 128, rngs=rngs)])
        self.layer3 = nnx.List([ResBlock(128, 256, stride=2, rngs=rngs),
                                ResBlock(256, 256, rngs=rngs)])
        self.layer4 = nnx.List([ResBlock(256, 512, stride=2, rngs=rngs),
                                ResBlock(512, 512, rngs=rngs)])
        self.fc = nnx.Linear(512, num_classes, rngs=rngs)

    def __call__(self, x):
        # x: (B, H, W, C) — NNX uses NHWC
        x = nnx.relu(self.bn1(self.conv1(x)))
        # Max pool with explicit padding to match PyTorch/standard ResNet
        x = jnp.pad(x, ((0,0),(1,1),(1,1),(0,0)), constant_values=-jnp.inf)
        x = nnx.max_pool(x, (3, 3), strides=(2, 2), padding='VALID')
        for block in self.layer1: x = block(x)
        for block in self.layer2: x = block(x)
        for block in self.layer3: x = block(x)
        for block in self.layer4: x = block(x)
        x = jnp.mean(x, axis=(1, 2))
        return self.fc(x)


# ============================================================
# 2. ViT-Base (NNX)
# ============================================================
class ViTBlock(nnx.Module):
    def __init__(self, d_model=768, n_heads=12, d_ff=3072, *, rngs):
        self.ln1 = nnx.LayerNorm(d_model, rngs=rngs)
        self.attn = nnx.MultiHeadAttention(num_heads=n_heads,
                                            in_features=d_model,
                                            qkv_features=d_model,
                                            decode=False, rngs=rngs)
        self.ln2 = nnx.LayerNorm(d_model, rngs=rngs)
        self.ff1 = nnx.Linear(d_model, d_ff, rngs=rngs)
        self.ff2 = nnx.Linear(d_ff, d_model, rngs=rngs)

    def __call__(self, x):
        r = x
        x = self.ln1(x)
        x = self.attn(x)
        x = x + r
        r = x
        x = self.ln2(x)
        x = nnx.gelu(self.ff1(x))
        x = self.ff2(x)
        return x + r


class ViTBase(nnx.Module):
    def __init__(self, num_classes=1000, d_model=768, n_layers=12,
                 n_heads=12, patch_size=16, *, rngs):
        self.patch_embed = nnx.Conv(3, d_model, kernel_size=(patch_size, patch_size),
                                     strides=(patch_size, patch_size), padding='VALID', rngs=rngs)
        self.cls_token = nnx.Param(jax.random.normal(rngs(), (1, 1, d_model)) * 0.02)
        # We'll lazily size pos_embed
        self.pos_embed = nnx.Param(jax.random.normal(rngs(), (1, 197, d_model)) * 0.02)  # 196 patches + 1 cls
        self.blocks = nnx.List([ViTBlock(d_model, n_heads, rngs=rngs) for _ in range(n_layers)])
        self.ln = nnx.LayerNorm(d_model, rngs=rngs)
        self.head = nnx.Linear(d_model, num_classes, rngs=rngs)

    def __call__(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x.reshape(B, -1, x.shape[-1])  # (B, N, D)
        cls = jnp.broadcast_to(self.cls_token[...], (B, 1, x.shape[-1]))
        x = jnp.concatenate([cls, x], axis=1)
        x = x + self.pos_embed[...]
        for block in self.blocks:
            x = block(x)
        x = self.ln(x[:, 0])
        return self.head(x)


# ============================================================
# 3. 5-Stacked LSTM (NNX)
# ============================================================
class StackedLSTM(nnx.Module):
    def __init__(self, input_dim=128, hidden_dim=512, n_layers=5,
                 num_classes=10, *, rngs):
        layers = []
        for i in range(n_layers):
            in_d = input_dim if i == 0 else hidden_dim
            layers.append(nnx.LSTMCell(in_d, hidden_dim, rngs=rngs))
        self.layers = nnx.List(layers)
        self.hidden_dim = hidden_dim
        self.head = nnx.Linear(hidden_dim, num_classes, rngs=rngs)

    def __call__(self, x):
        B, T, D = x.shape
        for lstm_cell in self.layers:
            carry = lstm_cell.initialize_carry((B,), rngs=nnx.Rngs(0))
            outputs = []
            for t in range(T):
                carry, y = lstm_cell(carry, x[:, t, :])
                outputs.append(y)
            x = jnp.stack(outputs, axis=1)
        return self.head(x[:, -1, :])


# ============================================================
# Run benchmarks
# ============================================================
if __name__ == '__main__':
    # Determine target device (mlx, mps, or cpu)
    target = jax.devices()[0]
    cpu = jax.devices("cpu")[0]
    print(f"Target device: {target}")

    # Initialize on CPU to avoid stablehlo.composite issues on MPS
    with jax.default_device(cpu):
        key = random.key(42)
        rngs = nnx.Rngs(0)

    # --- ResNet-18 ---
    print("=" * 70)
    print("ResNet-18 (batch=32, 224×224, ImageNet)")
    print("=" * 70)
    with jax.default_device(cpu):
        resnet = ResNet18(rngs=rngs)
        key, k1, k2 = random.split(key, 3)
        x_rn = random.normal(k1, (32, 224, 224, 3))
        y_rn = random.randint(k2, (32,), 0, 1000)
    graphdef, params, rest = nnx.split(resnet, nnx.Param, ...)
    params = jax.device_put(params, target)
    rest = jax.device_put(rest, target)
    x_rn = jax.device_put(x_rn, target)
    y_rn = jax.device_put(y_rn, target)

    @jax.jit
    def resnet_step(params, rest, x, y):
        model = nnx.merge(graphdef, params, rest)
        def loss_fn(model):
            logits = model(x)
            return jnp.mean(jax.nn.log_softmax(logits) * jax.nn.one_hot(y, 1000)) * -1
        loss, grads = nnx.value_and_grad(loss_fn)(model)
        _, new_params, new_rest = nnx.split(model, nnx.Param, ...)
        new_params = jax.tree.map(lambda w, g: w - 0.01 * g, new_params, grads)
        return new_params, new_rest, loss

    try:
        bench("ResNet-18 fwd+bwd+SGD", resnet_step, params, rest, x_rn, y_rn)
    except Exception as e:
        print(f"  ResNet-18: ERROR - {e}")

    # --- ViT-Base ---
    print()
    print("=" * 70)
    print("ViT-Base (batch=32, 224×224, patch=16, d=768, 12L, 12H)")
    print("=" * 70)
    with jax.default_device(cpu):
        vit = ViTBase(rngs=nnx.Rngs(1))
        key, k1, k2 = random.split(key, 3)
        x_vit = random.normal(k1, (32, 224, 224, 3))
        y_vit = random.randint(k2, (32,), 0, 1000)
    vit_gd, vit_state = nnx.split(vit)
    vit_state = jax.device_put(vit_state, target)
    x_vit = jax.device_put(x_vit, target)
    y_vit = jax.device_put(y_vit, target)

    @jax.jit
    def vit_step(state, x, y):
        model = nnx.merge(vit_gd, state)
        def loss_fn(model):
            logits = model(x)
            return jnp.mean(jax.nn.log_softmax(logits) * jax.nn.one_hot(y, 1000)) * -1
        loss, grads = nnx.value_and_grad(loss_fn)(model)
        _, new_state = nnx.split(model)
        new_state = jax.tree.map(lambda w, g: w - 0.0001 * g, new_state, grads)
        return new_state, loss

    try:
        bench("ViT-Base fwd+bwd+SGD", vit_step, vit_state, x_vit, y_vit)
    except Exception as e:
        print(f"  ViT-Base: ERROR - {e}")

    # --- 5-Stacked LSTM ---
    print()
    print("=" * 70)
    print("5-Stacked LSTM (batch=64, seq=256, input=128, hidden=512)")
    print("=" * 70)
    with jax.default_device(cpu):
        lstm = StackedLSTM(rngs=nnx.Rngs(2))
        key, k1, k2 = random.split(key, 3)
        x_lstm = random.normal(k1, (64, 256, 128))
        y_lstm = random.randint(k2, (64,), 0, 10)
    lstm_gd, lstm_state = nnx.split(lstm)
    lstm_state = jax.device_put(lstm_state, target)
    x_lstm = jax.device_put(x_lstm, target)
    y_lstm = jax.device_put(y_lstm, target)

    @jax.jit
    def lstm_step(state, x, y):
        model = nnx.merge(lstm_gd, state)
        def loss_fn(model):
            logits = model(x)
            return jnp.mean(jax.nn.log_softmax(logits) * jax.nn.one_hot(y, 10)) * -1
        loss, grads = nnx.value_and_grad(loss_fn)(model)
        _, new_state = nnx.split(model)
        new_state = jax.tree.map(lambda w, g: w - 0.001 * g, new_state, grads)
        return new_state, loss

    try:
        bench("5-LSTM fwd+bwd+SGD", lstm_step, lstm_state, x_lstm, y_lstm)
    except Exception as e:
        print(f"  5-LSTM: ERROR - {e}")

    print()
    print("=" * 70)
    print(f"Done — {backend}")
    print("=" * 70)
