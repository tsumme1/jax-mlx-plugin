"""
quasi_elk_repro.py

Minimal reproducible quasi-ELK solver (Parallel Scalar Kalman Filter) 
adapted for dual-driver explicit RK4 parallelization using jax.lax.associative_scan.
"""

import jax
import jax.numpy as jnp
from jax import vmap, lax
from typing import NamedTuple, Optional
from functools import partial

# ═══════════════════════════════════════════════════════════════════════════════
# Parallel Scalar Kalman Filter (ELK)
# ═══════════════════════════════════════════════════════════════════════════════

class ScalarParams(NamedTuple):
    initial_mean: jnp.ndarray
    dynamics_weights: jnp.ndarray
    dynamics_bias: jnp.ndarray
    emission_noises: jnp.ndarray

class PosteriorScalarFilter(NamedTuple):
    filtered_means: Optional[jnp.ndarray] = None
    filtered_covariances: Optional[jnp.ndarray] = None

class FilterMessageScalar(NamedTuple):
    A: jnp.ndarray
    b: jnp.ndarray
    C: jnp.ndarray
    J: jnp.ndarray
    eta: jnp.ndarray

def _initialize_filtering_messages(params, emissions):
    num_timesteps = emissions.shape[0]

    def _first_message(params, y):
        m = params.initial_mean
        sigma2 = params.emission_noises[0]
        S = jnp.ones(1) + sigma2
        A = jnp.zeros(1)
        b = m + (y - m) / S
        C = jnp.ones(1) - (S ** -1)
        eta = jnp.zeros(1)
        J = jnp.ones(1)
        return A, b, C, J, eta

    @partial(vmap, in_axes=(None, 0, 0))
    def _generic_message(params, y, t):
        F = params.dynamics_weights[t]
        b = params.dynamics_bias[t]
        sigma2 = params.emission_noises[t + 1]
        K = 1 / (1 + sigma2)
        eta = F * K * (y - b)
        J = (F ** 2) * K
        A = F - K * F
        b = b + K * (y - b)
        C = 1 - K
        return A, b, C, J, eta

    A0, b0, C0, J0, eta0 = _first_message(params, emissions[0])
    At, bt, Ct, Jt, etat = _generic_message(
        params, emissions[1:], jnp.arange(len(emissions) - 1))

    return FilterMessageScalar(
        A=jnp.concatenate([A0, At]), b=jnp.concatenate([b0, bt]),
        C=jnp.concatenate([C0, Ct]), J=jnp.concatenate([J0, Jt]),
        eta=jnp.concatenate([eta0, etat]))

def parallel_scalar_filter(params, emissions):
    @vmap
    def _operator(elem1, elem2):
        A1, b1, C1, J1, eta1 = elem1
        A2, b2, C2, J2, eta2 = elem2
        denom = C1 * J2 + 1
        A = (A1 * A2) / denom
        b = A2 * (C1 * eta2 + b1) / denom + b2
        C = C1 * (A2 ** 2) / denom + C2
        eta = A1 * (eta2 - J2 * b1) / denom + eta1
        J = J2 * (A1 ** 2) / denom + J1
        return FilterMessageScalar(A, b, C, J, eta)

    initial_messages = _initialize_filtering_messages(params, emissions)
    
    # This associative scan is the known trigger for Apple Silicon MLX GPU hangs
    final_messages = lax.associative_scan(_operator, initial_messages)
    return PosteriorScalarFilter(
        filtered_means=final_messages.b,
        filtered_covariances=final_messages.C)

# ═══════════════════════════════════════════════════════════════════════════════
# Parallel RK4 using ELK 
# ═══════════════════════════════════════════════════════════════════════════════

def parallel_rk4(f_continuous, initial_state, drivers_1, drivers_2, dt, num_iters=2, sigmasq=1e8):
    DIM = initial_state.shape[0]
    T = drivers_1.shape[0]
    
    def f_discrete(y, u1, u2):
        k1 = f_continuous(y, u1, u2)
        k2 = f_continuous(y + 0.5 * dt * k1, u1, u2)
        k3 = f_continuous(y + 0.5 * dt * k2, u1, u2)
        k4 = f_continuous(y + dt * k3, u1, u2)
        return y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    states_guess = jnp.zeros((T, DIM))

    def _step(states, _):
        # NOTE: Only taking the diagonal of the Jacobian to match ELK
        fs = vmap(f_discrete)(states[:-1], drivers_1[1:], drivers_2[1:])

        def diag_jac(z, u1, u2):
            def diag_jvp(i):
                tangent = jnp.zeros(DIM).at[i].set(1.0)
                _, jvp_val = jax.jvp(lambda z_: f_discrete(z_, u1, u2), (z,), (tangent,))
                return jvp_val[i]
            return vmap(diag_jvp)(jnp.arange(DIM))

        Jdiags = vmap(diag_jac)(states[:-1], drivers_1[1:], drivers_2[1:])
        As = Jdiags
        bs = fs - As * states[:-1]
        
        f0 = f_discrete(initial_state, drivers_1[0], drivers_2[0])

        params = ScalarParams(
            initial_mean=f0,
            dynamics_weights=As,
            dynamics_bias=bs,
            emission_noises=jnp.ones(T) * sigmasq)

        post = jax.vmap(
            parallel_scalar_filter,
            in_axes=(ScalarParams(0, 1, 1, None), 1),
            out_axes=1,
        )(params, states)
        
        new_states = post.filtered_means
        return new_states, new_states

    final_states, _ = lax.scan(_step, states_guess, None, length=num_iters, unroll=1)
    return final_states
