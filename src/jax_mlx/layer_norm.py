"""
LayerNorm lowering for MLX.

Routes JAX LayerNorm through MLX's fused mlx::fast::layer_norm kernel.
Both forward and backward passes use MLX's fused kernels via custom_call.

Usage:
    After importing jax_mlx, flax.nnx.LayerNorm and flax.linen.LayerNorm
    automatically use the MLX fused kernel. No code changes needed.
"""

import jax
import jax.numpy as jnp
from jax._src import core
from functools import partial


# ---- Pure JAX fallback implementations ----

def _layer_norm_impl(x, weight, bias, eps):
    """Pure JAX fallback for layer norm."""
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    normalized = (x - mean) / jnp.sqrt(var + eps)
    return normalized * weight + bias


# ---- Forward Primitive ----
mlx_layer_norm_p = core.Primitive('mlx_layer_norm')
mlx_layer_norm_p.multiple_results = False


def _layer_norm_abstract(x, weight, bias, *, eps):
    return core.ShapedArray(x.shape, x.dtype)


mlx_layer_norm_p.def_abstract_eval(_layer_norm_abstract)
mlx_layer_norm_p.def_impl(
    lambda x, w, b, **kw: _layer_norm_impl(x, w, b, kw.get('eps', 1e-5))
)


# ---- Backward Primitive ----
mlx_layer_norm_bwd_p = core.Primitive('mlx_layer_norm_bwd')
mlx_layer_norm_bwd_p.multiple_results = True


def _layer_norm_bwd_abstract(x, weight, bias, grad_out, *, eps):
    return (
        core.ShapedArray(x.shape, x.dtype),       # dx
        core.ShapedArray(weight.shape, weight.dtype),  # dweight
        core.ShapedArray(bias.shape, bias.dtype),      # dbias
    )


mlx_layer_norm_bwd_p.def_abstract_eval(_layer_norm_bwd_abstract)


def _layer_norm_bwd_impl(x, weight, bias, grad_out, *, eps):
    """Pure JAX fallback backward."""
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    std = jnp.sqrt(var + eps)
    normalized = (x - mean) / std
    
    # dbias = sum(grad_out) over all dims except last
    dbias = jnp.sum(grad_out, axis=tuple(range(grad_out.ndim - 1)))
    # dweight = sum(grad_out * normalized) over all dims except last
    dweight = jnp.sum(grad_out * normalized, axis=tuple(range(grad_out.ndim - 1)))
    
    # dx: standard layer_norm backward
    N = x.shape[-1]
    dx_hat = grad_out * weight
    dx = (1.0 / std) * (dx_hat - jnp.mean(dx_hat, axis=-1, keepdims=True)
                         - normalized * jnp.mean(dx_hat * normalized, axis=-1, keepdims=True))
    return dx, dweight, dbias


mlx_layer_norm_bwd_p.def_impl(
    lambda x, w, b, g, **kw: _layer_norm_bwd_impl(x, w, b, g, eps=kw.get('eps', 1e-5))
)


# ---- MLIR Lowering Registration ----

def _register_mlx_layer_norm_lowering():
    """Register MLIR lowering for both forward and backward LayerNorm primitives."""
    from jax._src.interpreters import mlir

    def layer_norm_lowering(ctx, x, weight, bias, *, eps):
        aval_out = ctx.avals_out[0]
        result_type = mlir.aval_to_ir_type(aval_out)
        backend_config = f'{{"eps": {eps}}}'
        result = mlir.custom_call(
            call_target_name='mlx_layer_norm',
            result_types=[result_type],
            operands=[x, weight, bias],
            backend_config=backend_config,
        )
        return result.results

    def layer_norm_bwd_lowering(ctx, x, weight, bias, grad_out, *, eps):
        aval_dx, aval_dw, aval_db = ctx.avals_out
        result_types = [
            mlir.aval_to_ir_type(aval_dx),
            mlir.aval_to_ir_type(aval_dw),
            mlir.aval_to_ir_type(aval_db),
        ]
        backend_config = f'{{"eps": {eps}}}'
        result = mlir.custom_call(
            call_target_name='mlx_layer_norm_bwd',
            result_types=result_types,
            operands=[x, weight, bias, grad_out],
            backend_config=backend_config,
        )
        return result.results

    # Register forward lowering
    mlir.register_lowering(mlx_layer_norm_p, layer_norm_lowering, platform='METAL')
    try:
        mlir.register_lowering(mlx_layer_norm_p, layer_norm_lowering, platform='mlx')
    except NotImplementedError:
        pass

    # Register backward lowering
    mlir.register_lowering(mlx_layer_norm_bwd_p, layer_norm_bwd_lowering, platform='METAL')
    try:
        mlir.register_lowering(mlx_layer_norm_bwd_p, layer_norm_bwd_lowering, platform='mlx')
    except NotImplementedError:
        pass

    # Fallback lowerings for non-MLX platforms
    mlir.register_lowering(
        mlx_layer_norm_p,
        mlir.lower_fun(
            lambda x, w, b, eps=1e-5: _layer_norm_impl(x, w, b, eps),
            multiple_results=False,
        ),
    )
    mlir.register_lowering(
        mlx_layer_norm_bwd_p,
        mlir.lower_fun(
            lambda x, w, b, g, eps=1e-5: _layer_norm_bwd_impl(x, w, b, g, eps=eps),
            multiple_results=True,
        ),
    )


# ---- custom_vjp wrapper ----

def _mlx_layer_norm_with_grad(x, weight, bias, eps):
    """Dispatch to primitive and supply custom gradients via fused backward."""

    @jax.custom_vjp
    def _fwd(x, w, b):
        return mlx_layer_norm_p.bind(x, w, b, eps=eps)

    def _fwd_rule(x, w, b):
        out = _fwd(x, w, b)
        return out, (x, w, b)

    def _bwd_rule(res, g):
        x, w, b = res
        dx, dw, db = mlx_layer_norm_bwd_p.bind(x, w, b, g, eps=eps)
        return dx, dw, db

    _fwd.defvjp(_fwd_rule, _bwd_rule)
    return _fwd(x, weight, bias)


# ---- Monkey-patching ----

def patch_flax_layer_norm():
    """
    Monkey-patch flax LayerNorm to use MLX's fused kernel.
    Patches both flax.nnx.LayerNorm and flax.linen.LayerNorm.
    """
    try:
        _patch_nnx_layer_norm()
    except Exception as e:
        pass

    try:
        _patch_linen_layer_norm()
    except Exception as e:
        pass


def _patch_nnx_layer_norm():
    """Patch flax.nnx.LayerNorm.__call__ to use MLX fused layer_norm."""
    import flax.nnx as nnx

    _original_call = nnx.LayerNorm.__call__

    def _mlx_layer_norm_call(self, x):
        # Only intercept simple cases: last-axis normalization
        feature_axes = getattr(self, 'feature_axes', -1)
        # Normalize to tuple
        if isinstance(feature_axes, int):
            feature_axes = (feature_axes,)
        # Check if it's just the last axis
        if feature_axes != (-1,) and feature_axes != (x.ndim - 1,):
            return _original_call(self, x)

        eps = getattr(self, 'epsilon', 1e-5)
        
        # Get scale and bias arrays from Param objects
        scale = self.scale
        if hasattr(scale, 'raw_value'):
            scale = scale.raw_value
        elif hasattr(scale, '__getitem__'):
            try:
                scale = scale[...]
            except Exception:
                scale = scale
        
        bias_obj = getattr(self, 'bias', None)
        if bias_obj is None:
            return _original_call(self, x)
        bias_val = bias_obj
        if hasattr(bias_val, 'raw_value'):
            bias_val = bias_val.raw_value
        elif hasattr(bias_val, '__getitem__'):
            try:
                bias_val = bias_val[...]
            except Exception:
                bias_val = bias_val

        return _mlx_layer_norm_with_grad(x, scale, bias_val, eps)

    nnx.LayerNorm.__call__ = _mlx_layer_norm_call


def _patch_linen_layer_norm():
    """Patch flax.linen.LayerNorm.__call__ to use MLX fused layer_norm."""
    import flax.linen as nn

    _original_call = nn.LayerNorm.__call__

    def _mlx_layer_norm_call(self, x):
        # Only intercept simple cases: last-axis normalization
        feature_axes = getattr(self, 'feature_axes', (-1,))
        if feature_axes != (-1,) and feature_axes != (x.ndim - 1,):
            return _original_call(self, x)

        eps = getattr(self, 'epsilon', 1e-6)
        use_bias = getattr(self, 'use_bias', True)
        use_scale = getattr(self, 'use_scale', True)

        if not use_bias or not use_scale:
            return _original_call(self, x)

        # Initialize params through flax's param mechanism
        feature_shape = (x.shape[-1],)
        scale = self.param('scale', nn.initializers.ones, feature_shape)
        bias = self.param('bias', nn.initializers.zeros, feature_shape)

        return _mlx_layer_norm_with_grad(x, scale, bias, eps)

    nn.LayerNorm.__call__ = _mlx_layer_norm_call
