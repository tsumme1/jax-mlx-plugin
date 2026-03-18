---
description: Resume ResNet benchmark comparison between jax-mlx and jax-mps, including fixing the conv stride-2 broadcast_shapes bug
---

# Context

We are comparing jax-mlx vs jax-mps performance on the ResNet18 benchmark from the jax-mps repo.

## Current State

### Bug: `broadcast_shapes: (256,16,16,128) vs (256,15,15,128)`

- The jax-mps ResNet18 benchmark (`/tmp/jax-mps/examples/resnet/main.py`) **works on CPU** (5/5 steps, loss=1.774, ~8s/step) but **crashes on MLX on step 2**.
- The error is `ValueError: [broadcast_shapes] Shapes (256,16,16,128) and (256,15,15,128) cannot be broadcast`.
- This occurs inside a compiled function during the training step (backward pass of ResNet18 with stride-2 convolutions).
- **Root cause**: MLX's `conv_general` produces different spatial output dimensions than XLA expects for certain convolutions with asymmetric padding and stride > 1.
- The mismatch happens ONLY on step 2 (step 1 succeeds) — likely because Flax NNX retraces the JIT-compiled function after optimizer state initialization.

### Fix Attempt: Post-hoc Shape Correction

A post-hoc shape correction was added to `jax_mlx_pjrt.cpp` (after the `conv_general` call, around line 7747):
- Computes XLA's expected output spatial dimensions using the formula:
  ```
  dilated_dim = (input_dim - 1) * lhs_dilation + 1
  eff_kernel = (kernel_dim - 1) * rhs_dilation + 1
  output_dim = floor((dilated_dim + pad_lo + pad_hi - eff_kernel) / stride) + 1
  ```
- If MLX's actual output shape differs from XLA's expected shape, slices/pads the result.
- This applies to ALL convolutions (not just lhs_dilation > 1) since the mismatch also occurs for strided forward convs.
- The fix was just built and needs testing.

### What Hasn't Worked

1. **vjp approach**: Tried reconstructing forward conv and using `mx::vjp` for input gradients. Failed because MLX's own backward pass has the same shape bug — went in circles.
2. **lhs_dilation-only fix**: The mismatch also occurs for convs with `lhs_dil=[1,1]` (forward convs with stride=2 and asymmetric padding), so restricting the fix to `lhs_dilation > 1` doesn't catch all cases.
3. **Post-hoc slice after conv_general**: Added slice/pad after `conv_general` to fix output shape. **This doesn't work** because inside `mx::compile` trace, the subsequent operations (e.g., residual add) see the pre-slice shape and fail at eval time.

### Key Insight for Next Attempt

The fix must **adjust padding BEFORE calling `conv_general`**, not fix the output shape after. Specifically:
- Compute XLA's expected output shape from the formula above
- Work backwards to determine what `pad_hi` values MLX's `conv_general` needs to produce the correct output
- Adjust `pad_hi` (or `pad_lo`) before the call so MLX produces the right shape from the start
- This ensures the traced shapes in `mx::compile` are correct throughout the computation graph

### MPS Status

- jax-mps **WORKS** on this benchmark when initializing on CPU (bypasses `stablehlo.composite`).
- Use `JAX_PLATFORMS=mps,cpu` with `/tmp/resnet_cpu_init.py`
- **MPS Results**: 5/5 steps, loss 2.37→2.03, **0.226s/step**
- jax-mps handles ResNet18 stride-2 convolutions correctly — confirming the bug is MLX-specific.

## Key Files

- Plugin source: `/Users/thomas/Documents/jax-mlx-plugin/src/jax_mlx_pjrt.cpp`
- Conv handler: around line 7500-7800 (search for `stablehlo.convolution`)
- Post-hoc shape fix: around line 7747 (search for "Post-hoc shape correction")
- jax-mps ResNet benchmark: `/tmp/jax-mps/examples/resnet/main.py` (cloned from https://github.com/tillahoffmann/jax-mps)
- Weight gradient optimization (works): around line 7655 (search for "WEIGHT GRADIENT OPTIMIZATION")

## How to Run

### Build the plugin
// turbo
```bash
conda run -n jax --no-capture-output python -m pip install -e . 2>&1 | grep -E "(Built target|error:)" | tail -3
```
Working directory: `/Users/thomas/Documents/jax-mlx-plugin`

### Copy built dylib to conda env
// turbo
```bash
cp /Users/thomas/Documents/jax-mlx-plugin/src/jax_mlx/libmlx_pjrt_plugin.dylib /Users/thomas/miniforge3/envs/jax/lib/python3.13/site-packages/jax_mlx/libmlx_pjrt_plugin.dylib
```

### Run ResNet benchmark on MLX
// turbo
```bash
conda run -n jax --no-capture-output env JAX_PLATFORMS=mlx PYTHONPATH=/tmp/jax-mps/examples/resnet python /tmp/jax-mps/examples/resnet/main.py --steps 10
```

### Run ResNet benchmark on CPU (reference, working)
// turbo
```bash
conda run -n jax --no-capture-output env JAX_PLATFORMS=cpu PYTHONPATH=/tmp/jax-mps/examples/resnet python /tmp/jax-mps/examples/resnet/main.py --steps 5
```

### Run ResNet benchmark on MPS (currently broken - stablehlo.composite)
```bash
conda run -n jax_mps --no-capture-output env JAX_PLATFORMS=mps PYTHONPATH=/tmp/jax-mps/examples/resnet python /tmp/jax-mps/examples/resnet/main.py --steps 10
```

## Notes

- The `jax` conda env is for MLX, the `jax_mps` conda env is for MPS.
- CIFAR-10 data is cached at `~/.cache/cifar10/`.
- The jax-mps repo is cloned at `/tmp/jax-mps`.
- If `/tmp/jax-mps` is missing, reclone: `git clone --depth 1 https://github.com/tillahoffmann/jax-mps.git /tmp/jax-mps`
