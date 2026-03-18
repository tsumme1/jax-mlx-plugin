"""
Native MLX Model Benchmarks: ResNet-18, ViT-Base, 5-Stacked LSTM
Uses mlx.nn. Run in the jax conda env (has mlx installed).
"""
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import time
import math

print(f"MLX: {mx.__version__}")
print()

WARMUP = 5
RUNS = 20

def bench(name, fn, *args):
    for _ in range(WARMUP):
        out = fn(*args)
        mx.eval(out)
    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        out = fn(*args)
        mx.eval(out)
        times.append((time.perf_counter() - t0) * 1000)
    mean = sum(times) / len(times)
    std = (sum((t - mean)**2 for t in times) / len(times)) ** 0.5
    print(f"  {name:40s}  {mean:8.2f}ms ± {std:5.2f}ms  (min: {min(times):.2f}ms)")
    return mean


# ============================================================
# 1. ResNet-18
# ============================================================
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm(out_channels)

        self.shortcut = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                      stride=stride, bias=False)
            self.shortcut_bn = nn.BatchNorm(out_channels)

    def __call__(self, x):
        residual = x
        x = nn.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        if self.shortcut is not None:
            residual = self.shortcut_bn(self.shortcut(residual))
        return nn.relu(x + residual)


class ResNet18(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm(64)

        self.layer1 = [ResBlock(64, 64) for _ in range(2)]
        self.layer2 = [ResBlock(64, 128, stride=2)] + [ResBlock(128, 128)]
        self.layer3 = [ResBlock(128, 256, stride=2)] + [ResBlock(256, 256)]
        self.layer4 = [ResBlock(256, 512, stride=2)] + [ResBlock(512, 512)]
        self.fc = nn.Linear(512, num_classes)

    def __call__(self, x):
        # x: (B, H, W, C) — MLX uses NHWC
        x = nn.relu(self.bn1(self.conv1(x)))
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)(x)
        for block in self.layer1: x = block(x)
        for block in self.layer2: x = block(x)
        for block in self.layer3: x = block(x)
        for block in self.layer4: x = block(x)
        x = mx.mean(x, axis=(1, 2))  # global avg pool
        return self.fc(x)


# ============================================================
# 2. ViT-Base
# ============================================================
class ViTBlock(nn.Module):
    def __init__(self, d_model=768, n_heads=12, d_ff=3072):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiHeadAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)

    def __call__(self, x):
        r = x
        x = self.ln1(x)
        x = self.attn(x, x, x)
        x = x + r
        r = x
        x = self.ln2(x)
        x = nn.gelu(self.ff1(x))
        x = self.ff2(x)
        return x + r


class ViTBase(nn.Module):
    def __init__(self, num_classes=1000, d_model=768, n_layers=12,
                 n_heads=12, patch_size=16, img_size=224):
        super().__init__()
        n_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(3, d_model, kernel_size=patch_size,
                                      stride=patch_size, bias=True)
        self.cls_token = mx.random.normal((1, 1, d_model)) * 0.02
        self.pos_embed = mx.random.normal((1, n_patches + 1, d_model)) * 0.02
        self.blocks = [ViTBlock(d_model, n_heads) for _ in range(n_layers)]
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def __call__(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)  # (B, H', W', D)
        x = x.reshape(B, -1, x.shape[-1])  # (B, N, D)
        cls = mx.broadcast_to(self.cls_token, (B, 1, x.shape[-1]))
        x = mx.concatenate([cls, x], axis=1)
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        x = self.ln(x[:, 0])
        return self.head(x)


# ============================================================
# 3. 5-Stacked LSTM
# ============================================================
class StackedLSTM(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=512, n_layers=5, num_classes=10):
        super().__init__()
        self.lstm_layers = []
        for i in range(n_layers):
            in_d = input_dim if i == 0 else hidden_dim
            self.lstm_layers.append(nn.LSTM(in_d, hidden_dim))
        self.head = nn.Linear(hidden_dim, num_classes)
        self.hidden_dim = hidden_dim


    def __call__(self, x):
        for lstm in self.lstm_layers:
            x, _ = lstm(x)  # LSTM returns (hidden, cell)
        return self.head(x[:, -1, :])


# ============================================================
# Run benchmarks
# ============================================================

# --- ResNet-18 ---
print("=" * 70)
print("ResNet-18 (batch=32, 224×224, ImageNet)")
print("=" * 70)
resnet = ResNet18()
mx.eval(resnet.parameters())
x_rn = mx.random.normal((32, 224, 224, 3))
y_rn = mx.random.randint(0, 1000, (32,))
loss_and_grad_fn = nn.value_and_grad(resnet, lambda m, x, y:
    mx.mean(nn.losses.cross_entropy(m(x), y)))
optimizer = optim.SGD(learning_rate=0.01)

def resnet_step(x, y):
    loss, grads = loss_and_grad_fn(resnet, x, y)
    optimizer.update(resnet, grads)
    return loss

resnet_step_compiled = mx.compile(resnet_step)

try:
    bench("ResNet-18 fwd+bwd+SGD (eager)", resnet_step, x_rn, y_rn)
    bench("ResNet-18 fwd+bwd+SGD (compiled)", resnet_step_compiled, x_rn, y_rn)
except Exception as e:
    print(f"  ResNet-18: ERROR - {e}")

# --- ViT-Base ---
print()
print("=" * 70)
print("ViT-Base (batch=32, 224×224, patch=16, d=768, 12L, 12H)")
print("=" * 70)
vit = ViTBase()
mx.eval(vit.parameters())
x_vit = mx.random.normal((32, 224, 224, 3))
y_vit = mx.random.randint(0, 1000, (32,))
vit_loss_and_grad = nn.value_and_grad(vit, lambda m, x, y:
    mx.mean(nn.losses.cross_entropy(m(x), y)))
vit_opt = optim.SGD(learning_rate=0.0001)

def vit_step(x, y):
    loss, grads = vit_loss_and_grad(vit, x, y)
    vit_opt.update(vit, grads)
    return loss

vit_step_compiled = mx.compile(vit_step)

try:
    bench("ViT-Base fwd+bwd+SGD (eager)", vit_step, x_vit, y_vit)
    bench("ViT-Base fwd+bwd+SGD (compiled)", vit_step_compiled, x_vit, y_vit)
except Exception as e:
    print(f"  ViT-Base: ERROR - {e}")

# --- 5-Stacked LSTM ---
print()
print("=" * 70)
print("5-Stacked LSTM (batch=64, seq=256, input=128, hidden=512)")
print("=" * 70)
lstm = StackedLSTM()
mx.eval(lstm.parameters())
x_lstm = mx.random.normal((64, 256, 128))
y_lstm = mx.random.randint(0, 10, (64,))
lstm_loss_and_grad = nn.value_and_grad(lstm, lambda m, x, y:
    mx.mean(nn.losses.cross_entropy(m(x), y)))
lstm_opt = optim.SGD(learning_rate=0.001)

def lstm_step(x, y):
    loss, grads = lstm_loss_and_grad(lstm, x, y)
    lstm_opt.update(lstm, grads)
    return loss

lstm_step_compiled = mx.compile(lstm_step)

try:
    bench("5-LSTM fwd+bwd+SGD (eager)", lstm_step, x_lstm, y_lstm)
    bench("5-LSTM fwd+bwd+SGD (compiled)", lstm_step_compiled, x_lstm, y_lstm)
except Exception as e:
    print(f"  5-LSTM: ERROR - {e}")

print()
print("=" * 70)
print("Done — Native MLX")
print("=" * 70)
