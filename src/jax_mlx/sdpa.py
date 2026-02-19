"""
SDPA (Scaled Dot-Product Attention) lowering for MLX.

Routes jax.nn.dot_product_attention through MLX's fused
mlx::fast::scaled_dot_product_attention kernel when running on the MLX platform.
Both forward and backward passes use MLX's fused kernels via custom_call.

Usage:
    After importing jax_mlx, jax.nn.dot_product_attention automatically uses
    the MLX fused kernel. No code changes needed.
"""

import jax
import jax.numpy as jnp
from jax._src import core
from functools import partial


def _mlx_sdpa_impl(query, key, value, scale):
    """Pure JAX fallback implementation for abstract eval and non-MLX platforms."""
    # query: (B, T, N, H), key: (B, S, K, H), value: (B, S, K, H)
    q = jnp.transpose(query, (0, 2, 1, 3))  # (B, N, T, H)
    k = jnp.transpose(key, (0, 2, 1, 3))    # (B, K, S, H)
    v = jnp.transpose(value, (0, 2, 1, 3))  # (B, K, S, H)

    logits = jnp.matmul(q, jnp.swapaxes(k, -2, -1))  # (B, N, T, S)
    logits = logits * jnp.array(scale, dtype=logits.dtype)
    weights = jax.nn.softmax(logits, axis=-1)
    out = jnp.matmul(weights, v)  # (B, N, T, H)

    return jnp.transpose(out, (0, 2, 1, 3))  # (B, T, N, H)


# ---- Forward Primitive ----
mlx_sdpa_p = core.Primitive('mlx_sdpa')
mlx_sdpa_p.multiple_results = False


def mlx_sdpa_abstract(query, key, value, *, scale):
    """Abstract evaluation: output has same shape and dtype as query."""
    return core.ShapedArray(query.shape, query.dtype)


mlx_sdpa_p.def_abstract_eval(mlx_sdpa_abstract)
mlx_sdpa_p.def_impl(lambda q, k, v, **kw: _mlx_sdpa_impl(q, k, v, kw.get('scale', 1.0)))


# ---- Backward Primitive ----
mlx_sdpa_bwd_p = core.Primitive('mlx_sdpa_bwd')
mlx_sdpa_bwd_p.multiple_results = True


def mlx_sdpa_bwd_abstract(query, key, value, grad_out, *, scale):
    """Abstract evaluation: returns (dQ, dK, dV) with same shapes as inputs."""
    return (
        core.ShapedArray(query.shape, query.dtype),
        core.ShapedArray(key.shape, key.dtype),
        core.ShapedArray(value.shape, value.dtype),
    )


mlx_sdpa_bwd_p.def_abstract_eval(mlx_sdpa_bwd_abstract)


def _mlx_sdpa_bwd_impl(query, key, value, grad_out, *, scale):
    """Pure JAX fallback backward implementation."""
    q = jnp.transpose(query, (0, 2, 1, 3))   # (B, N, T, H)
    k = jnp.transpose(key, (0, 2, 1, 3))     # (B, N, S, H)
    v = jnp.transpose(value, (0, 2, 1, 3))   # (B, N, S, H)
    g = jnp.transpose(grad_out, (0, 2, 1, 3))  # (B, N, T, H)

    logits = jnp.matmul(q, jnp.swapaxes(k, -2, -1)) * scale
    weights = jax.nn.softmax(logits, axis=-1)

    dv = jnp.matmul(jnp.swapaxes(weights, -2, -1), g)
    d_weights = jnp.matmul(g, jnp.swapaxes(v, -2, -1))
    d_scores = weights * (d_weights - jnp.sum(d_weights * weights, axis=-1, keepdims=True))
    d_scores = d_scores * scale

    dq = jnp.matmul(d_scores, k)
    dk = jnp.matmul(jnp.swapaxes(d_scores, -2, -1), q)

    return (
        jnp.transpose(dq, (0, 2, 1, 3)),
        jnp.transpose(dk, (0, 2, 1, 3)),
        jnp.transpose(dv, (0, 2, 1, 3)),
    )


mlx_sdpa_bwd_p.def_impl(
    lambda q, k, v, g, **kw: _mlx_sdpa_bwd_impl(q, k, v, g, scale=kw.get('scale', 1.0))
)


# ---- MLIR Lowering Registration ----

def _register_mlx_sdpa_lowering():
    """Register MLIR lowering for both forward and backward SDPA primitives."""
    from jax._src.interpreters import mlir

    def mlx_sdpa_lowering(ctx, query, key, value, *, scale):
        """Emit a custom_call targeting mlx_sdpa (forward)."""
        aval_out = ctx.avals_out[0]
        result_type = mlir.aval_to_ir_type(aval_out)
        backend_config = f'{{"scale": {scale}}}'
        result = mlir.custom_call(
            call_target_name='mlx_sdpa',
            result_types=[result_type],
            operands=[query, key, value],
            backend_config=backend_config,
        )
        return result.results

    def mlx_sdpa_bwd_lowering(ctx, query, key, value, grad_out, *, scale):
        """Emit a custom_call targeting mlx_sdpa_bwd (backward via mx::vjp)."""
        aval_dq, aval_dk, aval_dv = ctx.avals_out
        result_types = [
            mlir.aval_to_ir_type(aval_dq),
            mlir.aval_to_ir_type(aval_dk),
            mlir.aval_to_ir_type(aval_dv),
        ]
        backend_config = f'{{"scale": {scale}}}'
        result = mlir.custom_call(
            call_target_name='mlx_sdpa_bwd',
            result_types=result_types,
            operands=[query, key, value, grad_out],
            backend_config=backend_config,
        )
        return result.results

    # Register forward lowering
    mlir.register_lowering(mlx_sdpa_p, mlx_sdpa_lowering, platform='METAL')
    try:
        mlir.register_lowering(mlx_sdpa_p, mlx_sdpa_lowering, platform='mlx')
    except NotImplementedError:
        pass

    # Register backward lowering
    mlir.register_lowering(mlx_sdpa_bwd_p, mlx_sdpa_bwd_lowering, platform='METAL')
    try:
        mlir.register_lowering(mlx_sdpa_bwd_p, mlx_sdpa_bwd_lowering, platform='mlx')
    except NotImplementedError:
        pass

    # Fallback lowerings for non-MLX platforms
    mlir.register_lowering(
        mlx_sdpa_p,
        mlir.lower_fun(
            lambda q, k, v, scale=1.0: _mlx_sdpa_impl(q, k, v, scale),
            multiple_results=False,
        ),
    )
    mlir.register_lowering(
        mlx_sdpa_bwd_p,
        mlir.lower_fun(
            lambda q, k, v, g, scale=1.0: _mlx_sdpa_bwd_impl(q, k, v, g, scale=scale),
            multiple_results=True,
        ),
    )


# ---- custom_vjp wrapper ----

def mlx_dot_product_attention(query, key, value, scale):
    """
    Scaled dot-product attention using MLX's fused kernel.
    Both forward and backward use fused custom_call primitives.

    Args:
        query: (B, T, N, H)
        key: (B, S, K, H)
        value: (B, S, K, H)
        scale: float, typically 1/sqrt(H)
    Returns:
        output: (B, T, N, H)
    """
    scale_f = float(scale) if not isinstance(scale, float) else scale
    return _mlx_sdpa_with_grad(query, key, value, scale_f)


def _mlx_sdpa_with_grad(query, key, value, scale):
    """Dispatch to primitive and supply custom gradients via fused backward."""

    @jax.custom_vjp
    def _fwd(q, k, v):
        return mlx_sdpa_p.bind(q, k, v, scale=scale)

    def _fwd_rule(q, k, v):
        out = _fwd(q, k, v)
        return out, (q, k, v)

    def _bwd_rule(res, g):
        q, k, v = res
        # Route backward through mlx_sdpa_bwd primitive -> mx::vjp on SDPA
        dq, dk, dv = mlx_sdpa_bwd_p.bind(q, k, v, g, scale=scale)
        return dq, dk, dv

    _fwd.defvjp(_fwd_rule, _bwd_rule)
    return _fwd(query, key, value)


def patch_jax_dot_product_attention():
    """
    Monkey-patch jax.nn.dot_product_attention to use MLX's fused SDPA
    when running on the MLX platform.
    """
    from jax._src.nn import functions as nn_functions
    import jax.nn as jnn

    _original = nn_functions.dot_product_attention

    def _mlx_dot_product_attention(
        query, key, value, bias=None, mask=None, *,
        scale=None, is_causal=False,
        query_seq_lengths=None, key_value_seq_lengths=None,
        local_window_size=None, implementation=None,
        return_residual=False,
    ):
        import jax.numpy as jnp

        # Only use MLX SDPA for simple cases
        if (bias is not None or mask is not None or is_causal or
                query_seq_lengths is not None or key_value_seq_lengths is not None or
                local_window_size is not None or return_residual):
            return _original(
                query, key, value, bias, mask,
                scale=scale, is_causal=is_causal,
                query_seq_lengths=query_seq_lengths,
                key_value_seq_lengths=key_value_seq_lengths,
                local_window_size=local_window_size,
                implementation=implementation,
                return_residual=return_residual,
            )

        # Handle 3D inputs (T, N, H) -> add batch dim
        output_rank = max(query.ndim, 4)
        if query.ndim == 3:
            query = jnp.expand_dims(query, 0)
            key = jnp.expand_dims(key, 0)
            value = jnp.expand_dims(value, 0)

        B, T, N, H = query.shape
        _, S, K, _ = key.shape

        # Only handle standard MHA (N==K) for now
        # GQA/MQA falls back to original
        if N != K:
            if output_rank == 3:
                query = query.squeeze(0)
                key = key.squeeze(0)
                value = value.squeeze(0)
            return _original(
                query, key, value, bias, mask,
                scale=scale, is_causal=is_causal,
                query_seq_lengths=query_seq_lengths,
                key_value_seq_lengths=key_value_seq_lengths,
                local_window_size=local_window_size,
                implementation=implementation,
                return_residual=return_residual,
            )

        if scale is None:
            scale = 1.0 / (H ** 0.5)

        out = mlx_dot_product_attention(query, key, value, scale)

        if output_rank == 3:
            out = out.squeeze(0)
        return out

    # Patch both the module function and the public API
    nn_functions.dot_product_attention = _mlx_dot_product_attention
    jnn.dot_product_attention = _mlx_dot_product_attention
