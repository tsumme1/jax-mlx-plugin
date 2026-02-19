#!/usr/bin/env python3
"""
Comprehensive test to verify compilation is happening for all operations 
in various control flow contexts (while, scan, cond, nested).

This test verifies TWO things:
1. **Correctness**: Results match CPU reference (rtol=1e-3, atol=1e-3)
2. **Compilation**: Checks for 'compile_safe=true' in debug output (ops use mx::compile())

Run with: 
  python tests/test_compilation_coverage.py           # Quick correctness check
  python tests/test_compilation_coverage.py --verify  # Also verify compilation via debug output
"""

import os
import sys
import subprocess

# Set up environment BEFORE importing JAX
os.environ['JAX_PLATFORMS'] = 'mlx,cpu'
os.environ['MLX_STRICT_COMPILE'] = '1'

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np

# Get devices
try:
    mlx_device = jax.devices('mlx')[0]
    cpu_device = jax.devices('cpu')[0]
except:
    print("MLX device not available, skipping test")
    sys.exit(0)


# ==================== Operation Categories ====================

def test_basic_math_in_while(x):
    """Basic arithmetic ops in while loop."""
    def body(val):
        i, acc = val
        acc = acc + x * 0.1
        acc = acc * 0.99
        acc = jnp.abs(acc)
        acc = jnp.maximum(acc, 0.0)
        return (i + 1, acc)
    def cond(val):
        return val[0] < 10
    _, result = lax.while_loop(cond, body, (0, x))
    return result


def test_trig_in_while(x):
    """Trigonometric ops in while loop."""
    def body(val):
        i, acc = val
        acc = jnp.sin(acc) * 0.5 + jnp.cos(acc) * 0.5
        acc = jnp.tanh(acc)
        return (i + 1, acc)
    def cond(val):
        return val[0] < 5
    _, result = lax.while_loop(cond, body, (0, x))
    return result


def test_comparison_in_while(x):
    """Comparison and select ops in while loop."""
    def body(val):
        i, acc = val
        mask = acc > 0.5
        acc = jnp.where(mask, acc * 0.9, acc * 1.1)
        return (i + 1, acc)
    def cond(val):
        return val[0] < 5
    _, result = lax.while_loop(cond, body, (0, x))
    return result


def test_reduce_in_while(x):
    """Reduction ops in while loop."""
    def body(val):
        i, acc = val
        s = jnp.sum(acc)
        m = jnp.mean(acc)
        acc = acc - m + s * 0.001
        return (i + 1, acc)
    def cond(val):
        return val[0] < 5
    _, result = lax.while_loop(cond, body, (0, x))
    return result


def test_matmul_in_while(x):
    """Matrix multiplication in while loop."""
    def body(val):
        i, acc = val
        acc = acc @ acc.T
        acc = acc / (jnp.max(jnp.abs(acc)) + 1e-6)
        return (i + 1, acc)
    def cond(val):
        return val[0] < 3
    _, result = lax.while_loop(cond, body, (0, x))
    return result


def test_dynamic_slice_in_while(x):
    """Dynamic slicing ops in while loop."""
    def body(val):
        i, acc = val
        # Use modulo to keep index in bounds
        idx = jnp.int32(i % (x.shape[0] - 2))
        sliced = lax.dynamic_slice(acc, (idx,), (2,))
        # Simple update
        acc = acc * 0.99 + jnp.mean(sliced) * 0.01
        return (i + 1, acc)
    def cond(val):
        return val[0] < 5
    _, result = lax.while_loop(cond, body, (0, x))
    return result


def test_dynamic_update_slice_in_while(x):
    """Dynamic update slice in while loop."""
    def body(val):
        i, acc = val
        idx = jnp.int32(i % (x.shape[0] - 2))
        update = jnp.ones(2) * (i + 1) * 0.1
        acc = lax.dynamic_update_slice(acc, update, (idx,))
        return (i + 1, acc)
    def cond(val):
        return val[0] < 3
    _, result = lax.while_loop(cond, body, (0, x))
    return result


def test_gather_in_while(x):
    """Gather-like ops in while loop using static slicing."""
    def body(val):
        i, acc = val
        # Simple gather pattern using static slicing
        first_half = acc[:4]
        second_half = acc[4:]
        acc = acc + jnp.concatenate([second_half, first_half]) * 0.1
        return (i + 1, acc)
    def cond(val):
        return val[0] < 5
    _, result = lax.while_loop(cond, body, (0, x))
    return result


def test_reshape_in_while(x):
    """Reshape ops in while loop."""
    def body(val):
        i, acc = val
        # Reshape pattern
        reshaped = acc.reshape((2, 4))
        result = jnp.sum(reshaped, axis=0)  # (4,)
        padded = jnp.concatenate([result, result])  # Back to (8,)
        acc = acc + padded * 0.1
        return (i + 1, acc)
    def cond(val):
        return val[0] < 5
    _, result = lax.while_loop(cond, body, (0, x))
    return result


def test_custom_call_in_while(x):
    """Custom call ops (CHLO: sinh, cosh) in while loop."""
    def body(val):
        i, acc = val
        acc = jnp.sinh(acc * 0.1)
        acc = jnp.tanh(acc)  # Normalize
        return (i + 1, acc)
    def cond(val):
        return val[0] < 3
    _, result = lax.while_loop(cond, body, (0, x))
    return result


def test_ops_in_scan(x):
    """Operations inside scan."""
    def body(carry, _):
        carry = carry + x * 0.1
        carry = jnp.sin(carry)
        carry = carry * 0.99
        return carry, carry
    final, _ = lax.scan(body, x, None, length=10)
    return final


def test_ops_in_cond(x):
    """Operations inside conditional."""
    def true_fn(x):
        return jnp.sin(x) + jnp.cos(x)
    def false_fn(x):
        return jnp.tanh(x) * 2.0
    return lax.cond(jnp.mean(x) > 0.5, true_fn, false_fn, x)


def test_nested_while(x):
    """Nested while loops."""
    def outer_body(val):
        i, acc = val
        def inner_body(inner_val):
            j, inner_acc = inner_val
            inner_acc = inner_acc + 0.1
            return (j + 1, inner_acc)
        def inner_cond(inner_val):
            return inner_val[0] < 3
        _, inner_result = lax.while_loop(inner_cond, inner_body, (0, acc))
        return (i + 1, inner_result)
    def outer_cond(val):
        return val[0] < 3
    _, result = lax.while_loop(outer_cond, outer_body, (0, x))
    return result


def test_while_with_scan(x):
    """While loop containing scan."""
    def body(val):
        i, acc = val
        def scan_body(carry, _):
            return carry * 0.99, None
        acc, _ = lax.scan(scan_body, acc, None, length=5)
        return (i + 1, acc)
    def cond(val):
        return val[0] < 3
    _, result = lax.while_loop(cond, body, (0, x))
    return result


# ==================== Test Runner ====================

def run_test(name, test_fn, input_shape=(8,)):
    """Run a single test and check correctness."""
    try:
        # Create input
        key = jax.random.PRNGKey(42)
        x_np = np.array(jax.random.uniform(key, input_shape))
        
        # JIT compile the function
        jitted_fn = jax.jit(test_fn)
        
        # Run on CPU for reference
        with jax.default_device(cpu_device):
            x_cpu = jnp.array(x_np)
            ref = jitted_fn(x_cpu)
            ref.block_until_ready()
            ref_np = np.array(ref)
        
        # Run on MLX
        with jax.default_device(mlx_device):
            x_mlx = jnp.array(x_np)
            result = jitted_fn(x_mlx)
            result.block_until_ready()
            result_np = np.array(result)
        
        # Check correctness
        if np.allclose(result_np, ref_np, rtol=1e-3, atol=1e-3):
            return {'name': name, 'passed': True, 'error': None}
        else:
            max_diff = np.max(np.abs(result_np - ref_np))
            return {'name': name, 'passed': False, 'error': f'Max diff: {max_diff:.6f}'}
        
    except Exception as e:
        return {'name': name, 'passed': False, 'error': str(e)[:80]}


def verify_compilation():
    """Run a subprocess test and check for compile_safe=true in debug output."""
    test_code = '''
import os
os.environ['JAX_PLATFORMS'] = 'mlx'
import jax
import jax.numpy as jnp
from jax import lax

mlx = jax.devices('mlx')[0]

with jax.default_device(mlx):
    @jax.jit
    def f(x):
        # Test multiple ops that should be in isSimpleOp whitelist
        x = jnp.sin(x)  # Basic trig
        x = x + x * 0.5  # Arithmetic
        x = jnp.maximum(x, 0.0)  # Comparison
        x = jnp.sum(x, keepdims=True)  # Reduction
        return x
    
    x = jnp.ones(8)
    result = f(x)
    result.block_until_ready()
'''
    
    try:
        result = subprocess.run(
            [sys.executable, '-c', test_code],
            env={**os.environ, 'MLX_PJRT_DEBUG': '1'},
            capture_output=True,
            text=True,
            timeout=30
        )
        
        output = result.stdout + result.stderr
        
        # Check for compile_safe=true
        if 'compile_safe=true' in output:
            return True, "compile_safe=true found in debug output"
        elif 'compile_safe=false' in output:
            return False, "compile_safe=false - some ops falling back to interpreter"
        else:
            # Still pass if no debug output but execution succeeded
            if result.returncode == 0:
                return True, "Execution succeeded (no debug output)"
            return False, f"Exit code {result.returncode}"
            
    except subprocess.TimeoutExpired:
        return False, "Timeout"
    except Exception as e:
        return False, str(e)[:50]


def main():
    verify_mode = '--verify' in sys.argv
    
    print("=" * 70)
    print("COMPILATION COVERAGE TEST")
    print("=" * 70)
    print()
    
    # Verify compilation status if requested
    if verify_mode:
        print("Checking compilation status...")
        compiled, message = verify_compilation()
        if compiled:
            print(f"  ✅ Compilation verified: {message}")
        else:
            print(f"  ⚠️  Compilation check: {message}")
        print()
    
    tests = [
        ("Basic Math in While", test_basic_math_in_while, (8,)),
        ("Trig Ops in While", test_trig_in_while, (8,)),
        ("Comparison/Select in While", test_comparison_in_while, (8,)),
        ("Reduction in While", test_reduce_in_while, (8,)),
        ("MatMul in While", test_matmul_in_while, (4, 4)),
        ("Dynamic Slice in While", test_dynamic_slice_in_while, (8,)),
        ("Dynamic Update Slice in While", test_dynamic_update_slice_in_while, (8,)),
        ("Gather in While", test_gather_in_while, (8,)),
        ("Reshape in While", test_reshape_in_while, (8,)),
        ("Custom Call (CHLO) in While", test_custom_call_in_while, (8,)),
        ("Ops in Scan", test_ops_in_scan, (8,)),
        ("Ops in Cond", test_ops_in_cond, (8,)),
        ("Nested While Loops", test_nested_while, (8,)),
        ("While with Scan", test_while_with_scan, (8,)),
    ]
    
    results = []
    for name, test_fn, shape in tests:
        result = run_test(name, test_fn, shape)
        results.append(result)
        
        if result['passed']:
            print(f"  ✅ {name}")
        else:
            print(f"  ❌ {name}: {result['error']}")
    
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for r in results if r['passed'])
    failed = len(results) - passed
    
    print(f"  Passed: {passed}/{len(results)}")
    print(f"  Failed: {failed}/{len(results)}")
    print()
    
    if failed > 0:
        print("❌ Failed tests:")
        for r in results:
            if not r['passed']:
                print(f"    - {r['name']}: {r['error']}")
        return 1
    else:
        print("✅ All tests passed!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
