# Architecture

## Overview

The JAX MLX plugin implements the PJRT (Portable JAX Runtime) C API, allowing JAX to dispatch computations to Apple's MLX framework on Apple Silicon. The plugin consists of a C++ core (~6,400 lines) that handles graph execution, a Python parser for MLIR bytecode, and a lightweight type system bridging JAX and MLX data types.

## System Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                          JAX Frontend                                │
│  jax.numpy / jax.lax / jax.scipy / Flax / Optax                     │
└────────────────────────────┬─────────────────────────────────────────┘
                             │ StableHLO bytecode
                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│                       PJRT Plugin Layer                               │
│                                                                       │
│  ┌─────────────────┐   ┌──────────────────┐   ┌──────────────────┐  │
│  │  Python Parser   │   │  C++ Executor    │   │  Compilation     │  │
│  │  (parser.py)     │──▶│  (pjrt.cpp)      │──▶│  Engine          │  │
│  │                  │   │                  │   │  (mx::compile)   │  │
│  │  MLIR → Dict     │   │  SSA Interpreter │   │  Metal Kernels   │  │
│  └─────────────────┘   └──────────────────┘   └──────────────────┘  │
│                                                                       │
│  ┌─────────────────┐   ┌──────────────────┐   ┌──────────────────┐  │
│  │  Buffer Mgmt     │   │  Type System     │   │  Linalg          │  │
│  │  (MLXBuffer)     │   │  (64-bit emu)    │   │  Fast Paths      │  │
│  └─────────────────┘   └──────────────────┘   └──────────────────┘  │
└────────────────────────────┬─────────────────────────────────────────┘
                             │ MLX C++ API
                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    MLX Framework (Metal GPU)                          │
│  Lazy evaluation · Kernel fusion · Unified memory                    │
└──────────────────────────────────────────────────────────────────────┘
```

## Key Files

| File | Purpose |
|------|---------|
| `src/jax_mlx_pjrt.cpp` | Main PJRT C API implementation, graph interpreter, compilation engine (~6,400 lines) |
| `src/mlx_mlir_parser.h` | Lightweight StableHLO MLIR parser — extracts ops, attributes, shapes |
| `src/mlx_pjrt_types.h` | Buffer, device, executable type definitions |
| `src/jax_mlx/parser.py` | Python MLIR bytecode parser — deserializes portable artifacts to dict representation |
| `src/jax_mlx/plugin.py` | JAX plugin registration and device discovery |
| `src/jax_mlx/__init__.py` | Package metadata (version, etc.) |

## Execution Flow

1. **JAX compiles** a function to StableHLO bytecode
2. **Python parser** deserializes bytecode into a dictionary graph representation (ops, inputs, outputs, attributes)
3. **C++ executor** receives the graph and selects an execution tier
4. **Operations** are mapped to MLX C++ primitives via O(1) enum-based dispatch
5. **`mx.compile()`** fuses eligible operations into Metal GPU kernels
6. **Results** are returned via Apple Silicon's unified memory (zero-copy)

## Multi-Tier Execution Engine

The plugin uses a tiered strategy to maximize performance while maintaining full compatibility:

```
                    ┌─────────────────────┐
                    │  JAX StableHLO Graph │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Linalg Pattern      │──── SOLVE/INV pattern ──▶ Direct mx::linalg
                    │  Detection           │
                    └──────────┬──────────┘
                               │ No pattern
                    ┌──────────▼──────────┐
                    │  Compile Safety      │──── All ops simple ──▶ Tier 1: mx::compile()
                    │  Check               │                        (fused Metal kernels)
                    └──────────┬──────────┘
                               │ Has complex ops
                    ┌──────────▼──────────┐
                    │  Control Flow        │──── Has while/scan ──▶ Tier 2: Segmented
                    │  Detection           │                        (compile segments,
                    └──────────┬──────────┘                         interpret loops)
                               │ No loops
                    ┌──────────▼──────────┐
                    │  Tier 3: Full SSA    │
                    │  Interpreter         │
                    └─────────────────────┘
```

### Tier 1: Compiled Execution (`mx::compile`)

For graphs containing only "simple" operations (arithmetic, math, linear algebra, structural), the entire graph is compiled into fused Metal GPU kernels via `mx::compile()`. This is the fastest path, enabling full kernel fusion.

**Compile-safe operations include:**
- Arithmetic: `add`, `subtract`, `multiply`, `divide`, `negate`, `abs`, `floor`, `ceil`, `sign`
- Math: `exp`, `log`, `sqrt`, `rsqrt`, `tanh`, `sin`, `cos`, `power`, `log1p`, `round`
- Linear algebra: `dot`, `dot_general`, `convolution`
- Structural: `broadcast_in_dim`, `reshape`, `transpose`, `slice`, `reduce`, `concatenate`, `pad`, `iota`, `select`, `dynamic_slice`, `dynamic_update_slice`, `gather`, `scatter`
- Logic: `compare`, `constant`, `and`, `or`, `xor`, `not`, `shift_left`, `shift_right_logical`
- Recursive: `func.call` (nested function compilation inside fused kernels)

### Tier 2: Segmented Execution

For graphs containing while loops or scan operations, the graph is partitioned into segments at control-flow boundaries. Computational segments are compiled with `mx::compile()` while control-flow segments run through the interpreter. This provides kernel fusion for the majority of the graph (e.g., heavy matrix operations inside a loop body) while safely handling dynamic iteration.

**Key features:**
- Lazy compilation — segments are compiled on first execution
- Recursive subgraph input capture — ensures constants from outer scopes propagate correctly into nested while loops
- Universal fallback — if any segment fails to compile, execution seamlessly reverts to the interpreter

### Tier 3: Full SSA Interpreter

The fallback path processes all operations sequentially using an unordered map for O(1) value lookup. Supports all StableHLO operations including LAPACK custom calls routed to CPU. This path is always correct but does not benefit from kernel fusion.

## Memory Model

MLX uses **unified memory** on Apple Silicon — CPU and GPU share the same physical memory. This eliminates explicit host-to-device transfers:

- **Host → Device**: Data is copied into an `mlx::core::array` and lazily materialized on GPU when needed
- **Device → Host**: Results are read directly from the array's data pointer via `mlx::core::eval()`
- **Zero-copy**: No explicit DMA transfers between CPU and GPU address spaces

## Deferred Execution and Cross-Graph Fusion

The plugin implements a deferred execution model (Phases 10-13) that enables kernel fusion **across** independent JAX JIT boundaries:

1. **Dispatch deferral**: When JAX calls `Execute`, the plugin queues the graph and returns placeholder buffers immediately
2. **Batch accumulation**: Multiple dispatches are accumulated into a batch
3. **Materialization**: When results are needed (e.g., `ToHostBuffer`), all deferred graphs are merged into a single monolithic graph
4. **Cross-graph wiring**: Output-to-input dependencies between graphs are resolved by ID remapping, allowing MLX to fuse kernels from separate JIT compilations
5. **Lifetime management**: A quadruple ref-count elevation strategy (executable, inputs, outputs, placeholders) ensures all memory remains valid through the deferral boundary

## 64-Bit Integer Emulation

Metal GPUs lack native 64-bit integer support. The plugin handles this via selective expansion:
- 64-bit values are split into pairs of 32-bit values
- Bitwise operations (used by Threefry PRNG) operate on the expanded representation
- Results are recombined transparently

## Linalg Fast Paths

The executor detects high-level patterns in the StableHLO graph and bypasses the general interpreter:
- **`linalg.solve`**: Detected as a sequence of LU factorization + permutation loops + triangular solves → replaced with a single `mx::linalg::solve` call
- **`linalg.inv`**: Similar pattern detection → `mx::linalg::inv`
- Custom LAPACK FFI calls (`getrf`, `geqrf`, `householder_product`) are intercepted and mapped to native MLX functions

## Supported Operations (387 tested)

| Category | Operations |
|----------|------------|
| Element-wise | add, subtract, multiply, divide, exp, log, sin, cos, tanh, abs, sign, floor, ceil, round, power, sqrt, rsqrt, negate, clamp, log1p, expm1 |
| Reductions | reduce_sum, reduce_max, reduce_min, reduce_prod, reduce_mean, reduce_and, reduce_or, argmax, argmin |
| Linear algebra | dot, dot_general, triangular_solve, cholesky, qr, lu |
| Convolutions | conv_general_dilated (NHWC, NCHW) |
| Shape | reshape, transpose, broadcast_in_dim, slice, concatenate, pad, reverse, gather, scatter, dynamic_slice, dynamic_update_slice, sort, iota |
| Control flow | cond, while_loop, scan, fori_loop |
| RNG | uniform, normal, bit_generator (Threefry) |
| Comparison | eq, ne, lt, le, gt, ge |
| Bitwise | and, or, xor, not, shift_left, shift_right_logical |
| Type | convert, bitcast_convert |
| FFT | fft, ifft, rfft, irfft |
| Pooling | reduce_window, select_and_scatter |
| Other | custom_call (LAPACK FFI), real, imag, complex, is_finite, select, clamp |

## Testing

| Suite | Coverage | Command |
|-------|----------|---------|
| `test_exhaustive.py` | 387 JAX operations | `python tests/test_exhaustive.py` |
| `test_exhaustive_while.py` | 362 ops wrapped in `lax.while_loop` | `python tests/test_exhaustive_while.py` |
| `test_compilation_coverage.py` | Verifies backend compilation coverage for op categories | `python tests/test_compilation_coverage.py` |

## Performance Benchmarks

| Benchmark | What it measures | Command |
|-----------|-----------------|---------|
| `benchmark_cnn.py` | LeNet-5 CNN training (JAX MLX vs CPU) | `python benchmarks/benchmark_cnn.py` |
| `benchmark_outer_compile.py` | Kernel fusion impact with control flow | `python benchmarks/benchmark_outer_compile.py` |
| `benchmark_mlx_native.py` | Native MLX Python performance comparison | `python benchmarks/benchmark_mlx_native.py` |
| `benchmark_new_compiled_ops.py` | Compiled vs interpreted op performance | `python benchmarks/benchmark_new_compiled_ops.py` |

## Design Decisions

### Why Segmented Compilation?

Pure whole-graph compilation via `mx::compile()` cannot handle `stablehlo.while` or `stablehlo.scan` operations because while loop conditions require runtime evaluation (`mx::eval()`), which is incompatible with the MLX tracer. The segmented approach partitions the graph at control-flow boundaries, compiling each computational segment independently. This provides kernel fusion for the majority of ops while safely interpreting control flow.

### Why Deferred Execution?

JAX emits one StableHLO graph per JIT call. Without deferral, each graph is compiled and executed independently, missing fusion opportunities across calls. The deferred model accumulates graphs and merges them at materialization time, enabling MLX to fuse Metal kernels across JIT boundaries. This is particularly beneficial for training loops where forward/backward/update passes are separate JIT calls.

### Why O(1) Enum Dispatch?

The interpreter maps StableHLO operation names to an `OpType` enum at graph load time. During execution, a `switch` statement on the enum provides O(1) dispatch instead of repeated string comparisons. This eliminates per-op string overhead in the hot path.

## Configuration

| Variable | Description |
|----------|-------------|
| `MLX_PJRT_DEBUG=1` | Verbose debug logging |
| `MLX_NO_COMPILE=1` | Disable `mx.compile()` kernel fusion |
| `MLX_NO_COMPILE_AGGRESSIVE=1` | Disable segmented compilation |
| `MLX_NO_MEGA_COMPILE=1` | Disable deferred multi-graph fusion |
| `MLX_STRICT_COMPILE=1` | Error on compilation failures instead of fallback |
| `MLX_TIMING=1` | Execution timing output |
