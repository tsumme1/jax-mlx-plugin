# Bug: JIT-compiled ResNet fails with broadcast_shapes error

## Summary

When JIT-compiling a ResNet18 model via `torchax.compile()` (which uses `jax.jit`), the forward pass fails with a broadcast shape mismatch on the MLX backend. The same model works correctly on the JAX CPU backend.

## Error

```
ValueError: [broadcast_shapes] Shapes (32,20,112,224) and (32,20,112,112) cannot be broadcast.
```

The error occurs in the ResNet `BasicBlock.forward()` at the residual addition `out += self.shortcut(x)`, where the main branch and shortcut branch produce tensors with different spatial dimensions.

## Root Cause

The strided `Conv2d` (stride=2) in the shortcut connection appears to produce incorrect output spatial dimensions during JIT tracing on the MLX backend. Specifically:
- **Expected**: input `(32, C, 112, 112)` with stride-2 conv → output `(32, C_out, 56, 56)` for both main and shortcut paths
- **Actual on MLX**: one path produces `(32, 20, 112, 224)` — the width dimension is doubled instead of halved, suggesting the stride is being applied incorrectly (or the padding is wrong) during the JIT trace

This does **not** happen:
- In eager mode (both MLX and CPU backends work fine)
- In JIT mode on the JAX CPU backend (works correctly)

## Reproducer

```python
import numpy as np
import torch
import torch.nn as nn
import jax
import jax.numpy as jnp

# Must be run with MLX as the default JAX backend
# JAX_ENABLE_X64=1 python repro.py

import torchax
from torchax import tensor as tx
from torchax import interop

env = torchax.default_env()

# Minimal ResNet with stride-2 conv in shortcut
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, track_running_stats=False)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, track_running_stats=False)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes, track_running_stats=False)
            )

    def forward(self, x):
        import torch.nn.functional as F
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # <-- FAILS HERE on MLX JIT
        out = F.relu(out)
        return out

# Create block with stride=2 (triggers shortcut with strided conv)
block = BasicBlock(20, 40, stride=2)
jm = interop.JittableModule(block)

# Convert params/buffers to torchax tensors
for k, v in list(jm.params.items()):
    jm.params[k] = tx.Tensor(jnp.array(v.detach().numpy()), env)
for k, v in list(jm.buffers.items()):
    if isinstance(v, torch.Tensor):
        jm.buffers[k] = tx.Tensor(jnp.array(v.detach().numpy()), env)

# This triggers JIT compilation and fails on MLX
x = tx.Tensor(jnp.array(np.random.randn(4, 20, 16, 16).astype(np.float32)), env)
out = jm(x)  # ValueError: broadcast_shapes
```

## Environment

- macOS (Apple Silicon)
- JAX 0.4.30
- torch 2.10.0 (CPU)
- torchax 0.0.12
- jax-mlx-plugin (local build)

## Workaround

Use JAX CPU backend: `JAX_PLATFORMS=cpu`
